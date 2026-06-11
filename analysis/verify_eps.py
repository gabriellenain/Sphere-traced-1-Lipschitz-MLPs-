#!/usr/bin/env python3
"""Minimal ε verification for a trained Lipschitz-SDF checkpoint.

For a sample of rays from a few training views:
  (1) Distribution of |f(p_*)|  at hit  -> should peak just below ε
  (2) Distribution of ‖∇f(p_*)‖ at hit  -> bounds localization = ε/‖∇f‖
  (3) Hit rate vs ε on a log sweep      -> pick the plateau, not the cliff
Outputs PNG + a one-screen text summary.

Usage:
  python verify_eps.py <run_dir> [--ckpt checkpoint_step_020000.pt]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sphere_traced_screen_mesh import load_run                      # reuse loader
from plot_sphere_trace_steps import rays_for_view                   # reuse ray builder
from lip_tracer.sphere_tracing import trace_nograd
import lip_tracer.data as data_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--views", default="0,80,160,240",
                    help="comma-list of view indices to sample rays from")
    ap.add_argument("--n-rays", type=int, default=20000,
                    help="random rays sampled per view")
    ap.add_argument("--eps-sweep", default="3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2",
                    help="ε values for the sweep")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <run_dir>/eps_check")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f, cfg, scene = load_run(args.run_dir, args.ckpt, device)
    data_mod.SCENE = scene
    views = data_mod.load_views(scene, down=1)
    nV = int(views["images"].shape[0])
    view_list = [int(s) % nV for s in args.views.split(",")]
    eps_sweep = sorted(float(s) for s in args.eps_sweep.split(","))

    rng = np.random.default_rng(0)
    out_dir = args.out_dir or (args.run_dir / "eps_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ckpt:    {args.run_dir/'ckpt'/args.ckpt}")
    print(f"  scene:   {scene}")
    print(f"  views:   {view_list}")
    print(f"  cfg.eps: {cfg.eps:g}   cfg.iters: {cfg.iters}   cfg.newton: {cfg.newton_steps}")
    print(f"  eps sweep: {eps_sweep}")

    # --- collect random rays from selected views ---
    o_all, d_all = [], []
    for vi in view_list:
        o, d, _, _, H, W = rays_for_view(views, vi, device)
        idx = torch.from_numpy(rng.choice(o.shape[0], args.n_rays, replace=False)).to(device)
        o_all.append(o[idx]); d_all.append(d[idx])
    origins = torch.cat(o_all, dim=0)
    dirs    = torch.cat(d_all, dim=0)
    n = origins.shape[0]
    print(f"  total rays: {n:,}")

    # --- run tracer at the cfg.eps used in training -------------------------
    with torch.no_grad():
        _, t, hit = trace_nograd(f, origins, dirs, cfg)
    pts = origins + t.unsqueeze(-1) * dirs
    f_vals = torch.full((n,), float("nan"), device=device)
    grad_norms = torch.full((n,), float("nan"), device=device)
    if hit.any():
        p_hit = pts[hit]
        with torch.no_grad():
            f_vals[hit] = f(p_hit).squeeze(-1)
        # autograd for ‖∇f‖
        with torch.enable_grad():
            x = p_hit.detach().clone().requires_grad_(True)
            g = torch.autograd.grad(f(x).sum(), x)[0]
            grad_norms[hit] = g.norm(dim=-1)

    f_abs = f_vals[hit].abs().cpu().numpy()
    g_n   = grad_norms[hit].cpu().numpy()
    print("\n=== |f(p_*)| at hit (n={}) ===".format(int(hit.sum())))
    if len(f_abs):
        print(f"  mean {f_abs.mean():.3e}  p50 {np.median(f_abs):.3e}  "
              f"p90 {np.percentile(f_abs, 90):.3e}  max {f_abs.max():.3e}")
        print(f"  fraction ≤ cfg.eps: {(f_abs <= cfg.eps).mean():.3f}")
    print("\n=== ‖∇f(p_*)‖ at hit ===")
    if len(g_n):
        print(f"  mean {g_n.mean():.3f}  p50 {np.median(g_n):.3f}  "
              f"p10 {np.percentile(g_n, 10):.3f}  p90 {np.percentile(g_n, 90):.3f}")
        loc_bound = cfg.eps / np.clip(g_n, 1e-6, None)
        print(f"  localization bound ε/‖∇f‖:  mean {loc_bound.mean():.3e}  "
              f"p90 {np.percentile(loc_bound, 90):.3e}")

    # --- ε sweep: hit rate as a function of ε at fixed iter budget ---------
    print("\n=== hit rate vs ε sweep (iters={}) ===".format(cfg.iters))
    hit_curve = []
    for eps_val in eps_sweep:
        cfg_eps = replace(cfg, eps=float(eps_val))
        with torch.no_grad():
            _, _, h = trace_nograd(f, origins, dirs, cfg_eps)
        rate = float(h.float().mean().item())
        hit_curve.append((eps_val, rate))
        print(f"  ε={eps_val:.1e}  hit_rate={rate:.4f}")

    # --- save PNG ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if len(f_abs):
        axes[0].hist(np.log10(f_abs.clip(1e-12)), bins=80, color="#00c8ff", alpha=0.8)
        axes[0].axvline(np.log10(cfg.eps), color="red", ls="--", lw=1, label=f"cfg.eps={cfg.eps:g}")
        axes[0].set_xlabel("log10 |f(p_*)|"); axes[0].set_ylabel("count")
        axes[0].set_title(f"|f| at hit  (n={len(f_abs):,})"); axes[0].legend()
    if len(g_n):
        axes[1].hist(g_n, bins=80, color="#ff9900", alpha=0.8)
        axes[1].axvline(1.0, color="red", ls="--", lw=1, label="‖∇f‖=1 (ideal SDF)")
        axes[1].set_xlabel("‖∇f(p_*)‖"); axes[1].set_ylabel("count")
        axes[1].set_title("Gradient norm at hit"); axes[1].legend()
    eps_x = np.asarray([e for e, _ in hit_curve])
    rate_y = np.asarray([r for _, r in hit_curve])
    axes[2].semilogx(eps_x, rate_y, "o-", color="#aaffaa", lw=2, ms=8)
    axes[2].axvline(cfg.eps, color="red", ls="--", lw=1, label=f"cfg.eps={cfg.eps:g}")
    axes[2].set_xlabel("ε"); axes[2].set_ylabel("hit rate")
    axes[2].set_title("Hit rate vs ε  (iters={})".format(cfg.iters))
    axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3); axes[2].legend()
    out_png = out_dir / f"eps_check_{Path(args.ckpt).stem}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
