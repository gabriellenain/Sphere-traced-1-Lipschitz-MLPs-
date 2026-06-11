"""Wall-clock benchmark: dense vs compacted trace_nograd, swept over batch
size B at several iteration budgets K.

Right panel of the resulting figure plots wall-clock time vs B for each K,
which reveals whether per-pass cost is dominated by fixed launch/sync overhead
(flat at small B) or by linear-in-B compute (slope-1 in log-log).
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
    """Original (pre-patch) dense sphere trace; kept for benchmarking only."""
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
    ap.add_argument("--Bs", default="8192,16384,32768,65536,131072,262144,524288")
    ap.add_argument("--Ks", default="24,64,128,256")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--bsphere", type=float, default=1.5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
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
    o_pool = o_all[mask].to(device)
    d_pool = d_all[mask].to(device)
    pool = o_pool.shape[0]
    print(f"ray pool: {pool:,}  (views {view_ids}, down={args.down})", flush=True)

    Bs = [int(b) for b in args.Bs.split(",")]
    Ks = [int(k) for k in args.Ks.split(",")]
    B_max = max(Bs)
    if B_max > pool:
        raise SystemExit(f"requested B_max={B_max:,} exceeds ray pool {pool:,}")

    g = torch.Generator(device=device).manual_seed(args.seed)
    perm = torch.randperm(pool, generator=g, device=device)[:B_max]
    o_big = o_pool[perm].contiguous()
    d_big = d_pool[perm].contiguous()

    def slice_rays(B):
        return o_big[:B], d_big[:B]

    # warmup at small B, mid K
    o_w, d_w = slice_rays(min(Bs))
    cfg_warm = TraceConfig(iters=8, eps=args.eps,
                           bsphere_radius=args.bsphere, newton_steps=0)
    for _ in range(2):
        trace_nograd(f, o_w, d_w, cfg=cfg_warm)
        trace_nograd_dense(f, o_w, d_w, cfg_warm)

    # active-rays curve: take at B_max, K_max, dense (so we see the full schedule)
    K_max = max(Ks)
    cfg_full = TraceConfig(iters=K_max, eps=args.eps,
                           bsphere_radius=args.bsphere, newton_steps=0)
    o_full, d_full = slice_rays(B_max)
    _, _, alog = trace_nograd_dense(f, o_full, d_full, cfg_full, count_active=True)
    active_curve = np.asarray(alog, dtype=np.int64)

    # main sweep
    dense_grid = np.zeros((len(Bs), len(Ks)))
    compact_grid = np.zeros((len(Bs), len(Ks)))
    print(f"\n{'B':>8}  {'K':>4}  {'dense (ms)':>11}  {'compact (ms)':>13}  "
          f"{'speedup':>8}", flush=True)
    print(" " + "-" * 56, flush=True)
    for bi, B in enumerate(Bs):
        o, d = slice_rays(B)
        for ki, K in enumerate(Ks):
            cfg = TraceConfig(iters=K, eps=args.eps,
                              bsphere_radius=args.bsphere, newton_steps=0)
            t_dense = time_fn(lambda: trace_nograd_dense(f, o, d, cfg),
                              args.repeats)
            t_compact = time_fn(lambda: trace_nograd(f, o, d, cfg=cfg),
                                args.repeats)
            dense_grid[bi, ki] = t_dense
            compact_grid[bi, ki] = t_compact
            sp = t_dense / max(t_compact, 1e-9)
            print(f"  {B:>6}  {K:>3}  {t_dense*1e3:>10.3f}  "
                  f"{t_compact*1e3:>12.3f}  {sp:>7.2f}x", flush=True)

    # ---- plot ----
    Bs_arr = np.array(Bs)
    Ks_arr = np.array(Ks)
    iters_axis = np.arange(len(active_curve))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    # left: active rays per iter at (B_max, K_max)
    ax = axes[0]
    ax.plot(iters_axis, active_curve, lw=1.7, color="C0")
    ax.set_xlabel("sphere-tracing iteration $k$")
    ax.set_ylabel("active rays")
    ax.set_title(f"Active rays per iteration (B = {B_max:,}, K = {K_max})")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(active_curve) - 1)
    ax.set_ylim(0, B_max * 1.02)

    # right: ms vs B per K
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for ki, K in enumerate(Ks):
        c = cmap(ki / max(len(Ks) - 1, 1))
        ax.plot(Bs_arr, compact_grid[:, ki] * 1e3, "o-", color=c,
                lw=1.6, ms=4.5, label=f"compact K={K}")
        ax.plot(Bs_arr, dense_grid[:, ki] * 1e3, "s--", color=c,
                lw=1.2, ms=4, alpha=0.7, label=f"dense K={K}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size $B$ (rays)")
    ax.set_ylabel("wall-clock per pass (ms)")
    ax.set_title("Per-pass cost vs batch size")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)

    # slope-1 reference line anchored at the largest-K compact point at smallest B
    ref_x = np.array([Bs_arr[0], Bs_arr[-1]], dtype=float)
    anchor_ms = compact_grid[0, -1] * 1e3
    ref_y = anchor_ms * ref_x / ref_x[0]
    ax.plot(ref_x, ref_y, color="0.4", lw=0.8, ls=":", label="_linear-in-B")
    ax.text(ref_x[-1], ref_y[-1], "  ∝ B",
            color="0.4", fontsize=8, va="center")

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    fig.suptitle(f"DTU scan65 — views {view_ids} — {gpu}",
                 y=1.02, fontsize=10.5)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    np.savez(out.with_suffix(".npz"),
             active_per_iter=active_curve, B_max=B_max, K_max=K_max,
             Bs=Bs_arr, Ks=Ks_arr,
             dense_s=dense_grid, compact_s=compact_grid,
             gpu=gpu, views=np.array(view_ids))
    print(f"saved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)
    print(f"saved {out.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
