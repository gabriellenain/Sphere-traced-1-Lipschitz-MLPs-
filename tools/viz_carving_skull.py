"""Visualize MVSFormer++ volumetric carving targets for skull (DTU scan65).

Samples a 3D grid, computes the multi-view inside/outside vote and SDF target
for each point — same logic as mvsdf_carving_loss — and renders 2D slices.

Slices show:
  - outside_perc  : fraction of cameras voting outside  (0=inside, 1=outside)
  - n_valid       : how many cameras have valid depth at that point
  - SDF target    : what the carving loss would push f(x) toward
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

IDR_SCENE = Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65")
DEPTH_DIR = IDR_SCENE / "mvsformer_depth_1536x1152_1536x1152"

from lip_tracer.geomvs import load_mvsformer_depths_idr


def carving_volume(pts: torch.Tensor, depth_maps: torch.Tensor,
                   valid_maps: torch.Tensor, w2c: torch.Tensor,
                   K: torch.Tensor, out_thresh: float,
                   trunc: float, chunk: int = 8192):
    """Compute carving vote for each point. Returns (outside_perc, n_valid, target)."""
    N = pts.shape[0]
    V, H_d, W_d = depth_maps.shape

    total_valid   = torch.zeros(N)
    total_inside  = torch.zeros(N)
    best_inside_d = torch.full((N,),  1e6)
    best_outside_d= torch.full((N,), -1e6)

    for v in range(V):
        R = w2c[v, :3, :3]; t = w2c[v, :3, 3]

        for i in range(0, N, chunk):
            p   = pts[i:i+chunk]
            xc  = p @ R.T + t
            pd  = xc[:, 2]

            xp  = xc @ K[v].T
            uv  = xp[:, :2] / xp[:, 2:3].clamp(min=1e-6)
            u_n = uv[:, 0] / W_d * 2 - 1
            v_n = uv[:, 1] / H_d * 2 - 1
            grid = torch.stack([u_n, v_n], dim=1).view(1, -1, 1, 2)

            in_range = (xc[:, 2] > 0) & (u_n >= -1) & (u_n <= 1) \
                                       & (v_n >= -1) & (v_n <= 1)

            depth_s = F.grid_sample(
                depth_maps[v].unsqueeze(0).unsqueeze(0), grid,
                mode='nearest', padding_mode='zeros', align_corners=False,
            ).view(-1)
            valid_s = F.grid_sample(
                valid_maps[v].float().unsqueeze(0).unsqueeze(0), grid,
                mode='nearest', padding_mode='zeros', align_corners=False,
            ).view(-1) > 0.5

            valid   = (depth_s > 0) & in_range & valid_s
            inside  = (pd > depth_s * 0.99) & valid
            outside = valid & ~inside
            dist    = pd - depth_s

            total_valid[i:i+chunk]    += valid.float()
            total_inside[i:i+chunk]   += inside.float()
            best_inside_d[i:i+chunk]  = torch.where(
                inside & (dist < best_inside_d[i:i+chunk]), dist, best_inside_d[i:i+chunk])
            best_outside_d[i:i+chunk] = torch.where(
                outside & (dist > best_outside_d[i:i+chunk]), dist, best_outside_d[i:i+chunk])

    outside_perc = (total_valid - total_inside) / (total_valid + 1e-9)
    scene_valid   = total_valid > 0
    scene_outside = (outside_perc > out_thresh) & scene_valid
    scene_inside  = scene_valid & ~scene_outside

    safe_in  = best_inside_d.clamp(max=trunc)
    safe_out = best_outside_d.clamp(min=-trunc)
    ave_dist = safe_in * scene_inside.float() + safe_out * scene_outside.float()
    target   = (-ave_dist).clamp(-trunc, trunc)
    target[~scene_valid] = float('nan')

    outside_perc[~scene_valid] = float('nan')
    total_valid[~scene_valid]  = float('nan')

    return outside_perc, total_valid, target


def show_slices(fig, axes_row, vol, coords, axis_label, slice_vals,
                cmap, vmin, vmax, title_prefix, res):
    """Plot 2D slices of a 3D volume along one axis."""
    for col, sv in enumerate(slice_vals):
        idx = int((sv - coords.min()) / (coords.max() - coords.min()) * (res - 1))
        idx = max(0, min(res - 1, idx))
        if axis_label == 'y':
            sl = vol[:, idx, :]
        elif axis_label == 'x':
            sl = vol[idx, :, :]
        else:
            sl = vol[:, :, idx]
        im = axes_row[col].imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax,
                                   origin='lower', aspect='equal')
        axes_row[col].set_title(f"{title_prefix}  {axis_label}={sv:.2f}", fontsize=8)
        axes_row[col].axis('off')
        plt.colorbar(im, ax=axes_row[col], fraction=0.046)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",      type=Path,  default=IDR_SCENE)
    ap.add_argument("--depth-dir",  type=Path,  default=DEPTH_DIR)
    ap.add_argument("--res",        type=int,   default=96)
    ap.add_argument("--bound",      type=float, default=1.5)
    ap.add_argument("--conf-thresh",type=float, default=0.7)
    ap.add_argument("--out-thresh", type=float, default=0.7)
    ap.add_argument("--trunc",      type=float, default=1.25)
    ap.add_argument("--out",        type=Path,
                    default=Path("outputs/skull_carving.png"))
    args = ap.parse_args()

    print("Loading MVSFormer++ depths …")
    data = load_mvsformer_depths_idr(args.scene, args.depth_dir,
                                     conf_thresh=args.conf_thresh)
    depth_maps = torch.stack(data["depths"])   # (V, H, W)
    valid_maps = torch.stack(data["valid"])    # (V, H, W)
    w2c = torch.linalg.inv(torch.from_numpy(data["c2w"]).float())
    K   = torch.from_numpy(data["K"]).float()

    print(f"Building {args.res}³ grid  bound={args.bound} …")
    coords = torch.linspace(-args.bound, args.bound, args.res)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    print(f"Computing carving vote for {len(pts):,} points …")
    out_perc, n_valid, target = carving_volume(
        pts, depth_maps, valid_maps, w2c, K,
        out_thresh=args.out_thresh, trunc=args.trunc)

    out_vol    = out_perc.reshape(args.res, args.res, args.res).numpy()
    valid_vol  = n_valid.reshape(args.res, args.res, args.res).numpy()
    target_vol = target.reshape(args.res, args.res, args.res).numpy()

    slice_vals = [-0.5, 0.0, 0.3, 0.6]
    n_slices = len(slice_vals)
    fig, axes = plt.subplots(3, n_slices, figsize=(5 * n_slices, 14))

    show_slices(fig, axes[0], out_vol,   coords, 'y', slice_vals,
                'RdBu_r', 0, 1,   "outside_perc", args.res)
    show_slices(fig, axes[1], valid_vol, coords, 'y', slice_vals,
                'viridis', 0, None, "n_valid_views", args.res)
    show_slices(fig, axes[2], target_vol, coords, 'y', slice_vals,
                'coolwarm', -args.trunc, args.trunc, "SDF target", args.res)

    axes[0, 0].set_ylabel("outside_perc\n(red=outside)", fontsize=9)
    axes[1, 0].set_ylabel("# valid views", fontsize=9)
    axes[2, 0].set_ylabel("SDF target\n(blue=inside)", fontsize=9)

    fig.suptitle(
        f"Volumetric carving — skull scan65\n"
        f"res={args.res}  conf>{args.conf_thresh}  out_thresh={args.out_thresh}  trunc={args.trunc}",
        fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
