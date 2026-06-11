"""Training-time wall-clock benchmark: dense vs compacted trace_idr.

Counterpart to analysis/bench_trace_compaction.py (which measures inference / trace_nograd).
This one runs trace_idr forward + a dummy backward at training-relevant batch
sizes B and iteration budgets K, so the measured cost includes everything that
actually happens in a training step's trace path:
  * the no-grad sphere-trace loop (compacted vs dense),
  * the IDR implicit-gradient re-eval of f(x*),
  * normals via autograd,
  * backward through the IDR correction back into theta.

Plots two panels: active-rays-per-iter at the largest (B,K), and grouped bar
chart of dense vs compacted wall-clock per (B,K) cell with speedup labels.
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
    ray_sphere_exit, trace_idr, _needs_bracket, _refine_bracketed_roots,
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
    f.train()  # training mode — what we want to time
    return f


def trace_idr_dense(
    f, o: Tensor, d: Tensor, cfg: TraceConfig,
    collect_eik: bool = False, diff_normal: bool = False,
    count_active: bool = False,
):
    """Dense (uncompacted) IDR trace. Same semantics as trace_idr but the
    no-grad loop evaluates f on the full B every iter — no active.nonzero()
    indexing. Used only for benchmarking the compaction speedup.
    """
    B = o.shape[0]
    eik_buf = None  # not needed in the bench (collect_eik defaults False)

    if cfg.bsphere_radius > 0:
        t_far_ray = ray_sphere_exit(o.detach(), d.detach(), cfg.bsphere_radius)
    else:
        t_far_ray = torch.full((B,), cfg.t_far, device=o.device)

    active_log: list[int] = []
    with torch.no_grad():
        t = torch.zeros(B, device=o.device)
        sdf = torch.zeros(B, device=o.device)
        sdf_min = torch.full((B,), float("inf"), device=o.device)
        converged = torch.zeros(B, dtype=torch.bool, device=o.device)
        escaped = torch.zeros(B, dtype=torch.bool, device=o.device)
        use_bracket = _needs_bracket(f)
        prev_t = torch.zeros(B, device=o.device)
        prev_sdf = torch.zeros(B, device=o.device)
        have_prev = torch.zeros(B, dtype=torch.bool, device=o.device)
        bracketed = torch.zeros(B, dtype=torch.bool, device=o.device)
        bracket_lo = torch.zeros(B, device=o.device)
        bracket_hi = torch.zeros(B, device=o.device)
        for _ in range(cfg.iters):
            escaped = t >= t_far_ray
            active = ~(converged | escaped)
            if count_active:
                active_log.append(int(active.sum().item()))
            if not active.any():
                break
            sdf = f(o + t.unsqueeze(-1) * d)             # full B
            sdf_min = torch.minimum(sdf_min, sdf)
            if use_bracket:
                crossing = active & have_prev & (prev_sdf * sdf < 0)
                bracket_lo = torch.where(crossing, prev_t, bracket_lo)
                bracket_hi = torch.where(crossing, t, bracket_hi)
                bracketed = bracketed | crossing
                new_conv = active & ((sdf.abs() < cfg.eps) | bracketed)
            else:
                new_conv = active & (sdf.abs() < cfg.eps)
            converged = converged | new_conv
            keep_prev = active & ~(converged | escaped)
            prev_t = torch.where(keep_prev, t, prev_t)
            prev_sdf = torch.where(keep_prev, sdf, prev_sdf)
            have_prev = have_prev | keep_prev
            step = torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
            t = t + step

        if use_bracket and bracketed.any():
            t = torch.where(
                bracketed,
                _refine_bracketed_roots(f, o, d, bracket_lo, bracket_hi, bracketed, cfg.newton_steps),
                t,
            )
        hit = converged & (t < t_far_ray) & (t >= 0)
        hit_bg = (~hit) & (t >= t_far_ray - cfg.eps)

        # Newton refinement on hits — mirror trace_idr's no-grad Newton block
        eps_fd = 1e-3
        for _ in range(cfg.newton_steps):
            x = o + t.unsqueeze(-1) * d
            fval = f(x)
            ddir = (f(x + eps_fd * d) - fval) / eps_fd
            ddir = ddir.abs().clamp(min=1e-6)
            t = t - torch.where(hit & ~bracketed, fval / ddir, torch.zeros_like(t))

    # Normals
    x_star = (o + t.unsqueeze(-1) * d).detach()
    xr = x_star.requires_grad_(True)
    with torch.enable_grad():
        n_raw = torch.autograd.grad(f(xr).sum(), xr, create_graph=diff_normal)[0]
    if not diff_normal:
        n_raw = n_raw.detach()
    n_geo = n_raw.detach()

    n_dot_d = (n_geo * d).sum(-1)
    sign = torch.where(n_dot_d != 0, n_dot_d.sign(), torch.ones_like(n_dot_d))
    n_dot_d_safe = sign * n_dot_d.abs().clamp(min=1e-3)

    with torch.enable_grad():
        f_star = f(x_star)
    correction = torch.where(hit, f_star / n_dot_d_safe.detach(), torch.zeros_like(f_star))
    x_theta = x_star + correction.unsqueeze(-1) * (-d)

    return x_theta, t, hit, n_raw, sdf_min, hit_bg, active_log


def run_step(trace_fn, f, o, d, cfg, dense=False, count_active=False):
    """Forward trace + dummy loss backward — mirrors training step's trace cost."""
    for p in f.parameters():
        if p.grad is not None:
            p.grad = None
    if dense:
        x_theta, t, hit, n_raw, sdf_min, hit_bg, alog = trace_fn(
            f, o, d, cfg, collect_eik=False, diff_normal=False, count_active=count_active,
        )
    else:
        x_theta, t, hit, eik, n_raw, sdf_min, hit_bg = trace_fn(
            f, o, d, cfg, collect_eik=False, diff_normal=False,
        )
        alog = []
    # x_theta carries gradient into theta via IDR; sum().backward() is a
    # cheap stand-in for the photo/ncc/etc. losses that ultimately backprop
    # through x_theta.
    loss = x_theta.sum() + 0.0 * sdf_min.sum()
    loss.backward()
    return alog


def time_fn(fn, repeats: int) -> float:
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def sample_rays(rays_o: Tensor, rays_d: Tensor, B: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    N = rays_o.shape[0]
    idx = torch.randint(0, N, (B,), generator=gen, device=rays_o.device)
    return rays_o[idx].contiguous(), rays_d[idx].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", default="0,16,32,48",
                    help="comma-sep view ids to sample rays from")
    ap.add_argument("--Bs", default="4096,16384",
                    help="comma-sep batch sizes to bench")
    ap.add_argument("--Ks", default="64,128,256",
                    help="comma-sep iter budgets to bench")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--bsphere", type=float, default=1.5)
    ap.add_argument("--newton-steps", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}", flush=True)
    f = build_model(Path(args.pt), device)

    views = load_views(Path(args.scene), down=1)
    rays = make_deterministic_rays(views, down=2, device=device)
    o_all, d_all, vi_all = rays["o"], rays["d"], rays["vi"]
    view_ids = [int(v) for v in args.views.split(",")]
    mask = torch.zeros_like(vi_all, dtype=torch.bool)
    for v in view_ids:
        mask |= vi_all == v
    o_pool = o_all[mask].to(device).contiguous()
    d_pool = d_all[mask].to(device).contiguous()
    print(f"ray pool: {o_pool.shape[0]:,}  (views {view_ids})", flush=True)

    Bs = [int(b) for b in args.Bs.split(",")]
    Ks = [int(k) for k in args.Ks.split(",")]
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # warmup at largest (B, K)
    B_max, K_max = max(Bs), max(Ks)
    o_w, d_w = sample_rays(o_pool, d_pool, B_max, gen)
    cfg_warm = TraceConfig(iters=8, eps=args.eps, bsphere_radius=args.bsphere,
                           newton_steps=args.newton_steps, sdf_min_beta=0.0)
    for _ in range(2):
        run_step(trace_idr,       f, o_w, d_w, cfg_warm, dense=False)
        run_step(trace_idr_dense, f, o_w, d_w, cfg_warm, dense=True)

    results: dict[tuple[int, int], tuple[float, float, float]] = {}
    active_curve = None

    print(f"\n{'B':>7} {'K':>5}  {'dense (s)':>10}  {'compact (s)':>11}  "
          f"{'speedup':>8}", flush=True)
    print(" " + "-" * 52, flush=True)

    for B in Bs:
        # fix rays per B (so dense and compact see identical inputs) but resample
        # per-B (so different B sizes aren't correlated)
        o_b, d_b = sample_rays(o_pool, d_pool, B, gen)
        for K in Ks:
            cfg = TraceConfig(iters=K, eps=args.eps, bsphere_radius=args.bsphere,
                              newton_steps=args.newton_steps, sdf_min_beta=0.0)
            t_dense   = time_fn(lambda: run_step(trace_idr_dense, f, o_b, d_b, cfg, dense=True),
                                args.repeats)
            t_compact = time_fn(lambda: run_step(trace_idr,       f, o_b, d_b, cfg, dense=False),
                                args.repeats)
            speedup = t_dense / max(t_compact, 1e-9)
            results[(B, K)] = (t_dense, t_compact, speedup)
            print(f" {B:>7} {K:>5}  {t_dense:>10.4f}  {t_compact:>11.4f}  "
                  f"{speedup:>7.2f}x", flush=True)

            # capture active curve at largest (B, K) — dense path with logging
            if B == B_max and K == K_max:
                alog = run_step(trace_idr_dense, f, o_b, d_b, cfg,
                                dense=True, count_active=True)
                active_curve = np.array(alog, dtype=np.int64)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # left: active rays at (B_max, K_max)
    ax = axes[0]
    if active_curve is not None and len(active_curve) > 0:
        iters_axis = np.arange(len(active_curve))
        ax.plot(iters_axis, active_curve, lw=1.7, color="C0")
        ax.set_ylim(0, B_max * 1.02)
    ax.set_xlabel("sphere-tracing iteration $k$")
    ax.set_ylabel("active rays")
    ax.set_title(f"Active rays per iter (B = {B_max:,}, K = {K_max})")
    ax.grid(alpha=0.3)

    # right: grouped bars — for each B, dense vs compact across Ks
    ax = axes[1]
    n_groups = len(Bs) * len(Ks)
    labels = [f"B={B}\nK={K}" for B in Bs for K in Ks]
    dense_t = np.array([results[(B, K)][0] for B in Bs for K in Ks])
    compact_t = np.array([results[(B, K)][1] for B in Bs for K in Ks])
    speedup = dense_t / np.maximum(compact_t, 1e-9)
    x = np.arange(n_groups)
    width = 0.4
    ax.bar(x - width/2, dense_t * 1e3, width, color="C3", label="dense")
    ax.bar(x + width/2, compact_t * 1e3, width, color="C0", label="compacted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("wall-clock per train-step trace (ms)")
    ax.set_title("trace_idr fwd+bwd cost  —  dense vs compacted")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for xi, sp, dt, ct in zip(x, speedup, dense_t, compact_t):
        ax.text(xi, max(dt, ct) * 1e3 * 1.02, f"{sp:.2f}×",
                ha="center", fontsize=9, color="0.15", fontweight="bold")
    # vertical separators between B groups
    for gi in range(1, len(Bs)):
        ax.axvline(gi * len(Ks) - 0.5, color="0.8", lw=0.8)

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    np.savez(out.with_suffix(".npz"),
             active_per_iter=active_curve, B_max=B_max, K_max=K_max,
             Bs=np.array(Bs), Ks=np.array(Ks),
             dense_s=dense_t.reshape(len(Bs), len(Ks)),
             compact_s=compact_t.reshape(len(Bs), len(Ks)),
             speedup=speedup.reshape(len(Bs), len(Ks)),
             gpu=gpu, views=np.array(view_ids))
    print(f"\nsaved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)
    print(f"saved {out.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
