"""Visualize GT depth-unprojected points overlaid on Lego train view 20.

Filters to only keep points with good multi-view visibility across training cameras,
which is the same criterion that makes them useful for sfm_sdf_loss and free_space_loss.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lip_tracer.config import BLENDER_SCENE
from lip_tracer.data import load_blender_gt_points, load_blender_views


def project_pts_batch(pts: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor):
    """Project (N,3) world pts into all V cameras at once.

    Returns:
        uv    (V, N, 2)  pixel coordinates
        front (V, N)     bool, positive camera-space z
    """
    w2c = torch.linalg.inv(c2w)             # (V,4,4)
    R = w2c[:, :3, :3]                      # (V,3,3)
    t = w2c[:, :3, 3]                       # (V,3)
    xc = torch.einsum("vij,nj->vni", R, pts) + t[:, None, :]  # (V,N,3)
    z = xc[..., 2]
    uvh = torch.einsum("vij,vnj->vni", K, xc)                  # (V,N,3)
    uv = uvh[..., :2] / uvh[..., 2:3].clamp_min(1e-6)
    return uv, z > 1e-4


def visibility_filter(pts: torch.Tensor, views: dict, min_views: int = 3,
                      chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only points visible (in bounds + foreground) in >= min_views train cameras.

    Returns:
        pts_keep   (M, 3)  filtered points
        vis_count  (M,)    int, how many cameras saw each kept point
    """
    N = pts.shape[0]
    H, W = views["H"], views["W"]
    masks = views["masks"]   # (V, H, W) bool
    K     = views["K"]       # (V, 3, 3)
    c2w   = views["c2w"]     # (V, 4, 4)

    counts = torch.zeros(N, dtype=torch.long)

    for i in range(0, N, chunk):
        p = pts[i:i+chunk]                           # (c, 3)
        uv, front = project_pts_batch(p, K, c2w)    # (V,c,2), (V,c)

        u = uv[..., 0]   # (V, c)
        v = uv[..., 1]

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & front  # (V, c)

        # sample foreground mask at projected pixel
        ui = u.round().long().clamp(0, W - 1)
        vi = v.round().long().clamp(0, H - 1)
        V_ = masks.shape[0]
        c_ = p.shape[0]
        vi_ = torch.arange(V_)[:, None].expand(V_, c_)
        in_fg = masks[vi_, vi, ui]   # (V, c)

        visible = (in_bounds & in_fg).sum(dim=0)   # (c,)
        counts[i:i+chunk] = visible

    keep = counts >= min_views
    print(f"  visibility filter (>={min_views} views): {keep.sum()}/{N} pts kept  "
          f"[median vis={counts.float().median():.1f}  max={counts.max()}]")
    return pts[keep], counts[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=BLENDER_SCENE)
    ap.add_argument("--view", type=int, default=20)
    ap.add_argument("--n-pts", type=int, default=30000)
    ap.add_argument("--min-views", type=int, default=3,
                    help="min train cameras that must see a point (supervision quality)")
    ap.add_argument("--out", type=Path, default=Path("outputs/lego_colmap_overlay_v20.png"))
    ap.add_argument("--alpha", type=float, default=0.6, help="point transparency")
    ap.add_argument("--s", type=float, default=0.8, help="marker size")
    args = ap.parse_args()

    print("Loading GT points …")
    pts = load_blender_gt_points(args.scene, n_pts=args.n_pts)

    print("Loading train views …")
    views = load_blender_views(args.scene, split="train", down=1)
    V = views["images"].shape[0]
    if args.view >= V:
        raise ValueError(f"--view {args.view} out of range (0..{V-1})")

    print(f"Filtering by visibility across {V} train cameras …")
    pts_clean, vis_count = visibility_filter(pts, views, min_views=args.min_views)

    # project clean pts into the chosen view
    K   = views["K"][args.view]
    c2w = views["c2w"][args.view]
    H, W = views["H"], views["W"]
    img = views["images"][args.view].numpy()

    w2c = torch.linalg.inv(c2w)
    xc = pts_clean @ w2c[:3, :3].T + w2c[:3, 3]
    z = xc[:, 2]
    uvh = xc @ K.T
    uv = uvh[:, :2] / uvh[:, 2:3].clamp_min(1e-6)
    u, v_ = uv[:, 0].numpy(), uv[:, 1].numpy()

    front = (z > 1e-4).numpy()
    in_bounds = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H) & front
    u_v, v_v = u[in_bounds], v_[in_bounds]
    vis_v = vis_count.numpy()[in_bounds].astype(float)
    print(f"  {in_bounds.sum()} / {len(pts_clean)} cleaned pts visible in view {args.view}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.imshow(img)
    sc = ax.scatter(u_v, v_v, c=vis_v, cmap="YlOrRd",
                    s=args.s, alpha=args.alpha, linewidths=0,
                    vmin=args.min_views, vmax=V)
    plt.colorbar(sc, ax=ax, fraction=0.03, label="# train views seeing pt")
    ax.set_title(
        f"Supervision pts (≥{args.min_views} views) on train view {args.view}\n"
        f"{in_bounds.sum()} pts shown  |  {len(pts_clean)} total kept"
    )
    ax.axis("off")

    ax2 = axes[1]
    ax2.imshow(img)
    ax2.set_title(f"Train view {args.view} (clean)")
    ax2.axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
