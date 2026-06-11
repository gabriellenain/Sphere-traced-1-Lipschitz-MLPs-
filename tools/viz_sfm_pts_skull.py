"""Visualize skull (DTU scan65) point clouds overlaid on a chosen view.

Three panels:
  1. Sparse COLMAP (~10k pts) colored by multi-view visibility
  2. Dense aligned-depth (~100k pts) colored by multi-view visibility
  3. Reference image
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SKULL_SCENE = Path("/scratch/_projets_/willow/1-lip-tracer/data/DTU/scan65")

from lip_tracer.data import load_colmap_points, load_views
from lip_tracer.geomvs import load_dtu_gt_points, load_aligned_depths


def visibility_count(pts: torch.Tensor, views: dict, chunk: int = 4096) -> torch.Tensor:
    H, W   = views["H"], views["W"]
    masks  = views["masks"]
    w2c    = torch.linalg.inv(views["c2w"])
    R, t   = w2c[:, :3, :3], w2c[:, :3, 3]
    K      = views["K"]
    N      = pts.shape[0]
    counts = torch.zeros(N, dtype=torch.long)
    for i in range(0, N, chunk):
        p   = pts[i:i+chunk]
        xc  = torch.einsum("vij,nj->vni", R, p) + t[:, None, :]
        z   = xc[..., 2]
        uvh = torch.einsum("vij,vnj->vni", K, xc)
        uv  = uvh[..., :2] / uvh[..., 2:3].clamp_min(1e-6)
        u_, v_ = uv[..., 0], uv[..., 1]
        in_bounds = (u_ >= 0) & (u_ < W) & (v_ >= 0) & (v_ < H) & (z > 1e-4)
        ui  = u_.round().long().clamp(0, W - 1)
        vi  = v_.round().long().clamp(0, H - 1)
        V_  = masks.shape[0]; c_ = p.shape[0]
        vi_ = torch.arange(V_)[:, None].expand(V_, c_)
        counts[i:i+chunk] = (in_bounds & masks[vi_, vi, ui]).sum(dim=0)
    return counts


def project_visible(pts: torch.Tensor, vis_counts: torch.Tensor,
                    K: torch.Tensor, c2w: torch.Tensor, H: int, W: int):
    """Return (u, v, vis_color) for pts that project inside the view."""
    w2c = torch.linalg.inv(c2w)
    xc  = pts @ w2c[:3, :3].T + w2c[:3, 3]
    uvh = xc @ K.T
    uv  = uvh[:, :2] / uvh[:, 2:3].clamp_min(1e-6)
    u, v_ = uv[:, 0].numpy(), uv[:, 1].numpy()
    m = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H) & (xc[:, 2] > 1e-4).numpy()
    return u[m], v_[m], vis_counts.numpy()[m].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",     type=Path, default=SKULL_SCENE)
    ap.add_argument("--view",      type=int,  default=16)
    ap.add_argument("--n-dense",   type=int,  default=100_000)
    ap.add_argument("--min-views", type=int,  default=3)
    ap.add_argument("--out",       type=Path,
                    default=Path("outputs/skull_pts_overlay.png"))
    ap.add_argument("--alpha",    type=float, default=0.6)
    ap.add_argument("--s-sparse", type=float, default=4.0)
    ap.add_argument("--s-dense",  type=float, default=0.5)
    args = ap.parse_args()

    print("Loading views …")
    views = load_views(args.scene)
    V = views["images"].shape[0]
    if args.view >= V:
        raise ValueError(f"--view {args.view} out of range (0..{V-1})")
    img = views["images"][args.view].numpy()
    K, c2w = views["K"][args.view], views["c2w"][args.view]
    H, W   = views["H"], views["W"]

    # --- sparse COLMAP ---
    print("Loading sparse COLMAP points …")
    sp = load_colmap_points(args.scene)
    sp_vis = visibility_count(sp, views)
    sp_keep = sp_vis >= args.min_views
    sp_c, sp_vc = sp[sp_keep], sp_vis[sp_keep]
    print(f"  {sp_keep.sum()}/{len(sp)} pts kept (>={args.min_views} views)")
    sp_u, sp_v, sp_col = project_visible(sp_c, sp_vc, K, c2w, H, W)

    # --- dense aligned-depth ---
    print(f"Loading dense aligned-depth points (n={args.n_dense:,}) …")
    dp = load_dtu_gt_points(args.scene, n_pts=args.n_dense)
    dp_vis = visibility_count(dp, views)
    dp_keep = dp_vis >= args.min_views
    dp_c, dp_vc = dp[dp_keep], dp_vis[dp_keep]
    print(f"  {dp_keep.sum()}/{len(dp)} pts kept (>={args.min_views} views)")
    dp_u, dp_v, dp_col = project_visible(dp_c, dp_vc, K, c2w, H, W)

    print("Loading aligned depth maps …")
    depth_data = load_aligned_depths(args.scene)
    depth_v = depth_data["depths"][args.view].numpy()
    valid_v = depth_data["valid"][args.view].numpy()
    depth_show = np.where(valid_v, depth_v, np.nan)

    panels = [
        (sp_u, sp_v, sp_col, args.s_sparse,
         f"Sparse COLMAP  {len(sp_u)} visible / {len(sp_c)} kept / {len(sp)} raw"),
        (dp_u, dp_v, dp_col, args.s_dense,
         f"Dense GT depth  {len(dp_u)} visible / {len(dp_c)} kept / {args.n_dense} sampled"),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- PNG 1: reference + COLMAP overlay (2 panels) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
    axes2[0].imshow(img)
    axes2[0].set_title(f"View {args.view} (reference)")
    axes2[0].axis("off")
    u, v, col, s, title = panels[0]  # COLMAP only
    axes2[1].imshow(img)
    sc = axes2[1].scatter(u, v, c=col, cmap="YlOrRd", s=s, alpha=args.alpha,
                          linewidths=0, vmin=args.min_views, vmax=V)
    plt.colorbar(sc, ax=axes2[1], fraction=0.03, label="# cameras")
    axes2[1].set_title(title)
    axes2[1].axis("off")
    fig2.tight_layout()
    out2 = args.out.with_stem(args.out.stem + "_ref_colmap")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
    plt.close(fig2)

    # --- PNG 2: all four panels ---
    fig4, axes4 = plt.subplots(1, 4, figsize=(28, 7))
    for ax, (u, v, col, s, title) in zip(axes4, panels):
        ax.imshow(img)
        sc = ax.scatter(u, v, c=col, cmap="YlOrRd", s=s, alpha=args.alpha,
                        linewidths=0, vmin=args.min_views, vmax=V)
        plt.colorbar(sc, ax=ax, fraction=0.03, label="# cameras")
        ax.set_title(title)
        ax.axis("off")
    axes4[2].imshow(img)
    axes4[2].set_title(f"View {args.view} (reference)")
    axes4[2].axis("off")
    im = axes4[3].imshow(depth_show, cmap="plasma")
    plt.colorbar(im, ax=axes4[3], fraction=0.03, label="aligned z-depth")
    axes4[3].set_title(f"Aligned depth map  view {args.view}")
    axes4[3].axis("off")
    fig4.tight_layout()
    out4 = args.out.with_stem(args.out.stem + "_all4")
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Saved → {out4}")
    plt.close(fig4)


if __name__ == "__main__":
    main()
