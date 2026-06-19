#!/usr/bin/env python3
"""ICLR figure: does the *frozen 6-neighbour table* leave surface pixels with
ZERO valid source views ("dead coverage")?

The photometric/ZNCC loss compares each reference-view surface hit against a
per-view list of ``n_alt`` neighbour cameras that is precomputed once from
camera geometry alone (``view_selection`` = arccos / nearest), BEFORE any
per-pixel visibility test (lip_tracer/loss.py). A hit point only supervises
the surface if it is in-frame + front-facing in at least one of those neighbours
(loss.py: ``in_frame & not_occl & cos_ok``). When all n_alt miss, the point
gets no photometric gradient and is silently unsupervised.

This script makes that visible. For one (or a few) reference views it:
  1. sphere-traces the run's SDF to get surface hit points + normals,
  2. for the SAME ``n_alt`` and the SAME view-selection score used in training,
     counts, per hit pixel, how many of its neighbours see it (in-frame +
     front-facing, cos>cos_thresh),
  3. paints the reference image by that count and flags count==0 (dead) in red,
  4. contrasts arccos vs nearest vs the all-cameras ceiling.

Coverage here is in-frame + front-facing only (no occlusion trace), so it is an
UPPER bound on real coverage: the reported dead-pixel fraction is a LOWER bound.

Usage:
    python analysis/neighbour_coverage.py <run_dir> [--views 16,400,800]
        [--ckpt checkpoint_latest.pt] [--down 4] [--n-alt 6]
        [--selection arccos|nearest|both] [--out <png>]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# repo root (for lip_tracer.*) and this dir (for diagnose_holes) on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the traced-surface + coverage machinery already validated in the hole study.
from diagnose_holes import (
    _load_model, _make_rays, _trace_with_diagnostics, _normals_at, _angular_coverage,
)


def _neighbour_table(views: dict, n_alt: int, selection: str) -> torch.Tensor:
    """(V, n_alt) neighbour indices for the requested view-selection score."""
    from lip_tracer.data import precompute_alt_cameras, precompute_alt_cameras_arccos
    if selection == "arccos":
        return precompute_alt_cameras_arccos(views, n_alt)
    return precompute_alt_cameras(views, n_alt)


def _coverage_for_view(f, cfg, v, vi: int, down: int, steps: int,
                       cos_thr: float, device: str):
    """Trace reference view vi; return (Hd, Wd, rgb, hit2d, pts, normals)."""
    c2w = v["c2w"][vi].numpy(); K = v["K"][vi].numpy()
    H, W = v["H"], v["W"]
    img = v["images"][vi].numpy()

    o, d, Hd, Wd = _make_rays(c2w, K, H, W, down, device)
    hit, t_out, _, _ = _trace_with_diagnostics(f, o, d, cfg.trace, steps, device)
    hit = hit.numpy().astype(bool)

    pts = (o.cpu() + t_out.unsqueeze(-1) * d.cpu())          # (Hd*Wd, 3)
    normals = _normals_at(f, pts, device)
    # orient toward the reference camera (so front-facing test is well defined)
    cam_dir = torch.tensor(c2w[:3, 3], dtype=torch.float32) - pts
    normals[(cam_dir * normals).sum(-1) < 0] *= -1
    return Hd, Wd, img[:Hd, :Wd], hit, pts, normals


def _count_valid(pts, normals, hit, neigh_idx, v, cos_thr, H, W):
    """Per hit pixel: #neighbours that see it in-frame + front-facing. NaN off-surface."""
    c2ws = [v["c2w"][int(j)] for j in neigh_idx]
    Ks   = [v["K"][int(j)].numpy() for j in neigh_idx]
    cnt_hit, _ = _angular_coverage(pts[hit], normals[hit], c2ws, Ks, H, W, cos_thr)
    cnt = np.full(pts.shape[0], np.nan)
    cnt[hit] = cnt_hit
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--views", default="16", help="comma list of reference views")
    ap.add_argument("--ckpt", default="checkpoint_latest.pt")
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--n-alt", type=int, default=None, help="default: run cfg n_alt")
    ap.add_argument("--selection", choices=["arccos", "nearest", "both"], default="both")
    ap.add_argument("--steps", type=int, default=None, help="trace iters (default: run cfg)")
    ap.add_argument("--cos-thr", type=float, default=None, help="default: run cfg cos_thresh")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    f, cfg = _load_model(args.run_dir, args.device, args.ckpt)
    from lip_tracer.data import load_views
    v = load_views(cfg.scene, down=1)            # full-res cams; rays downsampled in _make_rays
    nview = v["c2w"].shape[0]
    H, W = v["H"], v["W"]

    n_alt   = args.n_alt   if args.n_alt   is not None else getattr(cfg.train, "n_alt", 6)
    cos_thr = args.cos_thr if args.cos_thr is not None else getattr(cfg.train, "cos_thresh", 0.1)
    steps   = args.steps   if args.steps   is not None else cfg.trace.iters
    sels = ["arccos", "nearest"] if args.selection == "both" else [args.selection]
    cfg_sel = getattr(cfg.train, "view_selection", "nearest")

    tables = {s: _neighbour_table(v, n_alt, s) for s in sels}
    all_idx = list(range(nview))

    views = [min(int(x), nview - 1) for x in args.views.split(",")]
    print(f"[cov] run={args.run_dir.name}  cfg view_selection={cfg_sel}  "
          f"n_alt={n_alt}  cos_thr={cos_thr}  steps={steps}  views={views}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    ncol = 1 + len(sels) + 1                       # RGB | per-selection | all-cam ceiling
    fig, ax = plt.subplots(len(views), ncol, figsize=(3.5 * ncol, 3.3 * len(views)),
                           facecolor="white", squeeze=False)
    cmap = plt.get_cmap("viridis", n_alt + 1)
    norm = BoundaryNorm(np.arange(-0.5, n_alt + 1.5, 1.0), cmap.N)

    for r, vi in enumerate(views):
        Hd, Wd, rgb, hit, pts, normals = _coverage_for_view(
            f, cfg, v, vi, args.down, steps, cos_thr, args.device)
        n_surf = int(hit.sum())

        a = ax[r][0]
        a.imshow(rgb); a.set_title(f"view {vi}\n{n_surf:,} surface px"); a.axis("off")

        def _panel(col, idx, title):
            cnt = _count_valid(pts, normals, hit, idx, v, cos_thr, H, W)
            cnt2d = cnt.reshape(Hd, Wd)
            dead = hit.reshape(Hd, Wd) & (cnt2d == 0)
            dead_frac = float(dead.sum()) / max(n_surf, 1)
            mean_cov = float(np.nanmean(cnt)) if n_surf else float("nan")
            a = ax[r][col]
            a.imshow(rgb)
            im = a.imshow(np.where(np.isnan(cnt2d), np.nan, cnt2d), cmap=cmap, norm=norm,
                          alpha=0.85, interpolation="nearest")
            # dead surface pixels in solid red on top
            red = np.zeros((Hd, Wd, 4)); red[..., 0] = 1.0; red[..., 3] = dead.astype(float)
            a.imshow(red, interpolation="nearest")
            a.set_title(f"{title}\nDEAD={dead_frac:.1%}  mean={mean_cov:.2f}/{len(idx)}")
            a.axis("off")
            if r == 0 and col == 1:
                fig.colorbar(im, ax=ax[r][col:col + len(sels)], fraction=0.025,
                             pad=0.02, label=f"# valid views (of {n_alt})")
            print(f"[cov] v{vi:>4} {title:<22} dead={dead_frac:.1%}  mean_valid={mean_cov:.2f}")
            return dead_frac

        col = 1
        for s in sels:
            _panel(col, tables[s][vi], f"{s} (n_alt={n_alt})")
            col += 1
        # all-cameras ceiling: best possible if every camera were a candidate
        _panel(col, torch.tensor(all_idx), f"ALL {nview} cams (ceiling)")

    fig.suptitle(
        f"Dead photometric coverage — {args.run_dir.name}\n"
        f"frozen {n_alt}-neighbour table ({'/'.join(sels)}) vs all-cam ceiling   ·   "
        f"in-frame+front-facing only (no occlusion ⇒ DEAD is a lower bound)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = args.out or (args.run_dir / f"neighbour_coverage_{'_'.join(map(str, views))}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[cov] saved → {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
