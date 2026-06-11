"""Wall-clock benchmark: dense (legacy) vs compacted trace_nograd.

Runs both implementations on the same checkpoint and rays at several iteration
budgets K, with cuda.synchronize() around each timing. Verifies identical hit
results, then plots (left) active rays per iter and (right) measured wall-clock
gain.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from lip_tracer.config import TraceConfig
from lip_tracer.data import load_views, make_deterministic_rays
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import (
    ray_sphere_exit, trace_nograd, _needs_bracket,
)


def build_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    architecture = ckpt.get("architecture", "cpl")
    if architecture == "neus":
        hidden = ckpt["f"]["layers.0.weight"].shape[0]
    elif "head_weight" in ckpt["f"]:
        hidden = ckpt["f"]["head_weight"].shape[0]
    else:
        hidden = next(v.shape[1] for k, v in ckpt["f"].items()
                      if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
                      and not k.startswith("encoder"))
    group_size = ckpt.get("group_size", 2)
    activation = ckpt.get("activation", "groupsort")
    input_encoding = ckpt.get("input_encoding", "pe")
    multires = ckpt.get("multires", 6)
    if architecture == "neus":
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("layers.") and k.endswith(".weight")))
    else:
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("net.") and k.endswith(".weight")
                                      and "_u" not in k))
    print(f"  arch={architecture} hidden={hidden} depth={depth} act={activation} "
          f"enc={input_encoding} mr={multires}", flush=True)
    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_encoding,
                   multires=multires, architecture=architecture).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f


@torch.no_grad()
def trace_nograd_dense(f, o: Tensor, d: Tensor, cfg: TraceConfig, count_active=False):
    """Original (pre-patch) dense sphere trace. Kept here only for benchmarking.

    Re-evaluates f on the full B every iter regardless of active count.
    """
    B = o.shape[0]
    t_far_ray = ray_sphere_exit(o, d, cfg.bsphere_radius) if cfg.bsphere_radius > 0 \
        else torch.full((B,), cfg.t_far, device=o.device)
    t = torch.zeros(B, device=o.device)
    converged = torch.zeros(B, dtype=torch.bool, device=o.device)
    use_bracket = _needs_bracket(f)
    prev_t = torch.zeros(B, device=o.device)
    prev_sdf = torch.zeros(B, device=o.device)
    have_prev = torch.zeros(B, dtype=torch.bool, device=o.device)
    bracketed = torch.zeros(B, dtype=torch.bool, device=o.device)
    bracket_lo = torch.zeros(B, device=o.device)
    bracket_hi = torch.zeros(B, device=o.device)
    active_log = []
    for _ in range(cfg.iters):
        escaped = t >= t_far_ray
        active = ~(converged | escaped)
        if count_active:
            active_log.append(int(active.sum().item()))
        if not active.any():
            break
        sdf = f(o + t.unsqueeze(-1) * d)
        crossing = active & have_prev & (prev_sdf * sdf < 0) if use_bracket else torch.zeros_like(converged)
        bracket_lo = torch.where(crossing, prev_t, bracket_lo)
        bracket_hi = torch.where(crossing, t, bracket_hi)
        bracketed = bracketed | crossing
        converged = converged | (sdf.abs() < cfg.eps) | bracketed
        keep_prev = active & ~(converged | escaped)
        prev_t = torch.where(keep_prev, t, prev_t)
        prev_sdf = torch.where(keep_prev, sdf, prev_sdf)
        have_prev = have_prev | keep_prev
        t = t + torch.where(converged | escaped, torch.zeros_like(t), sdf)
    sdf = f(o + t.unsqueeze(-1) * d)
    hit = (converged | (sdf.abs() < cfg.eps) | bracketed) & (t < t_far_ray) & (t >= 0)
    return t, hit, active_log


def time_fn(fn, repeats: int) -> float:
    """Median of repeats, cuda-synced."""
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", default="16,32")
    ap.add_argument("--down", type=int, default=2)
    ap.add_argument("--Ks", default="24,64,128,256")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--bsphere", type=float, default=1.5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=131072,
                    help="rays per chunk — limits peak memory for full-B f(x)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}", flush=True)
    f = build_model(Path(args.pt), device)

    views = load_views(Path(args.scene), down=1)
    rays = make_deterministic_rays(views, down=args.down, device=device)
    o_all, d_all, vi_all = rays["o"], rays["d"], rays["vi"]
    view_ids = [int(v) for v in args.views.split(",")]
    mask = torch.zeros_like(vi_all, dtype=torch.bool)
    for v in view_ids:
        mask |= vi_all == v
    o_full = o_all[mask].to(device)
    d_full = d_all[mask].to(device)
    B_full = o_full.shape[0]
    print(f"total rays: {B_full:,}  (views {view_ids}, down={args.down})", flush=True)

    Ks = [int(k) for k in args.Ks.split(",")]
    results = {}
    active_curve = None

    # Use chunks to keep dense full-B f(x) within GPU memory.
    def run_chunks(trace_fn, K, count_active=False):
        active_chunks = []
        for s in range(0, B_full, args.chunk):
            o = o_full[s:s+args.chunk]
            d = d_full[s:s+args.chunk]
            cfg = TraceConfig(iters=K, eps=args.eps,
                              bsphere_radius=args.bsphere, newton_steps=0)
            if trace_fn is trace_nograd:
                _, _, hit = trace_fn(f, o, d, cfg=cfg)
                active_chunks.append(None)
            else:
                _, hit, alog = trace_fn(f, o, d, cfg, count_active=count_active)
                active_chunks.append(alog)
        return active_chunks

    # warmup
    cfg_warm = TraceConfig(iters=8, eps=args.eps,
                           bsphere_radius=args.bsphere, newton_steps=0)
    o_w = o_full[:args.chunk]; d_w = d_full[:args.chunk]
    for _ in range(2):
        trace_nograd(f, o_w, d_w, cfg=cfg_warm)
        trace_nograd_dense(f, o_w, d_w, cfg_warm)

    print(f"\n{'K':>5}  {'dense (s)':>10}  {'compact (s)':>11}  {'speedup':>8}",
          flush=True)
    print(" " + "-" * 42, flush=True)
    for K in Ks:
        t_dense = time_fn(lambda: run_chunks(trace_nograd_dense, K), args.repeats)
        # Run dense once more with active counting (not in timing)
        if active_curve is None or K == max(Ks):
            active_chunks = run_chunks(trace_nograd_dense, K, count_active=True)
            # sum active across chunks per iter
            max_len = max(len(a) for a in active_chunks)
            tot = np.zeros(max_len, dtype=np.int64)
            for a in active_chunks:
                for i, v in enumerate(a):
                    tot[i] += v
            active_curve = tot
        t_compact = time_fn(lambda: run_chunks(trace_nograd, K), args.repeats)
        speedup = t_dense / max(t_compact, 1e-9)
        results[K] = (t_dense, t_compact, speedup)
        print(f"  {K:>3}  {t_dense:>10.3f}  {t_compact:>11.3f}  {speedup:>7.2f}x",
              flush=True)

    # Correctness check at largest K
    K = max(Ks)
    cfg_v = TraceConfig(iters=K, eps=args.eps,
                        bsphere_radius=args.bsphere, newton_steps=0)
    hit_d = []
    hit_c = []
    for s in range(0, B_full, args.chunk):
        o = o_full[s:s+args.chunk]; d = d_full[s:s+args.chunk]
        _, hd, _ = trace_nograd_dense(f, o, d, cfg_v)
        _, _, hc = trace_nograd(f, o, d, cfg=cfg_v)
        hit_d.append(hd.cpu()); hit_c.append(hc.cpu())
    hit_d = torch.cat(hit_d); hit_c = torch.cat(hit_c)
    agree = (hit_d == hit_c).float().mean().item()
    print(f"\nhit agreement: {agree*100:.4f}%  "
          f"(dense={int(hit_d.sum())} compact={int(hit_c.sum())})", flush=True)

    # ---- plot ----
    apc = active_curve
    iters_axis = np.arange(len(apc))
    Ks_arr = np.array(Ks)
    dense_t = np.array([results[k][0] for k in Ks])
    compact_t = np.array([results[k][1] for k in Ks])
    speedup = dense_t / np.maximum(compact_t, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    ax.plot(iters_axis, apc, lw=1.7, color="C0")
    ax.set_xlabel("sphere-tracing iteration $k$")
    ax.set_ylabel("active rays")
    ax.set_title(f"Active rays per iteration (B = {B_full:,})")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, B_full * 1.02)
    for thr, color in zip((0.5, 0.9, 0.99), ("C2", "C1", "C3")):
        target = B_full * (1 - thr)
        idx = np.where(apc <= target)[0]
        if len(idx):
            k = int(idx[0])
            ax.axvline(k, color=color, lw=0.8, ls="--", alpha=0.7)
            ax.text(k + 2, B_full * 0.92, f"{int(thr*100)}% done @ k={k}",
                    color=color, fontsize=8.5)

    ax = axes[1]
    width = 0.36
    x = np.arange(len(Ks))
    bars1 = ax.bar(x - width/2, dense_t * 1e3, width, color="C3",
                   label="dense (legacy)")
    bars2 = ax.bar(x + width/2, compact_t * 1e3, width, color="C0",
                   label="compacted")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel("wall-clock per pass (ms)")
    ax.set_title("Measured trace cost vs iteration budget")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for xi, sp, dt, ct in zip(x, speedup, dense_t, compact_t):
        ax.text(xi, max(dt, ct) * 1e3 * 1.02, f"{sp:.1f}×",
                ha="center", fontsize=9.5, color="0.15", fontweight="bold")

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    fig.suptitle(f"DTU scan65 — checkpoint 270k — views {view_ids} — {gpu}",
                 y=1.02, fontsize=10.5)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    np.savez(out.with_suffix(".npz"),
             active_per_iter=apc, B=B_full, Ks=Ks_arr,
             dense_s=dense_t, compact_s=compact_t, speedup=speedup,
             gpu=gpu, views=np.array(view_ids))
    print(f"saved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)
    print(f"saved {out.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
