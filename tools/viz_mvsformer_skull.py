"""Visualize MVSFormer++ depth maps for skull (DTU scan65 IDR).

Per chosen view: depth map | confidence | valid mask | surface pts on image
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

IDR_SCENE  = Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65")
DEPTH_DIR  = IDR_SCENE / "mvsformer_depth_1536x1152_1536x1152"

from lip_tracer.geomvs import load_mvsformer_depths_idr, backproject_depth
from lip_tracer.geomvs import _read_pfm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",      type=Path, default=IDR_SCENE)
    ap.add_argument("--depth-dir",  type=Path, default=DEPTH_DIR)
    ap.add_argument("--views",      type=str,  default="0,8,16,24,32,40")
    ap.add_argument("--conf-thresh",type=float, default=0.7)
    ap.add_argument("--out",        type=Path,
                    default=Path("outputs/skull_mvsformer.png"))
    args = ap.parse_args()

    view_ids = [int(v) for v in args.views.split(",")]

    print("Loading MVSFormer++ depths …")
    data = load_mvsformer_depths_idr(args.scene, args.depth_dir,
                                     conf_thresh=args.conf_thresh)
    if data is None:
        raise RuntimeError("No MVSFormer++ depth maps found")

    # also load raw confidence for display
    scan_dirs = [d for d in args.depth_dir.iterdir()
                 if d.is_dir() and (d / "depth_est").exists()]
    scan_root = scan_dirs[0] if scan_dirs else args.depth_dir
    conf_files = sorted((scan_root / "confidence").glob("*.npy"))

    # load IDR images for background
    img_paths = sorted(p for p in (args.scene / "image").iterdir()
                       if p.suffix.lower() in {".png", ".jpg"}
                       and not p.name.startswith("._"))
    import imageio.v2 as imageio

    n_cols = len(view_ids)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 16))
    if n_cols == 1:
        axes = axes[:, None]

    for col, vi in enumerate(view_ids):
        depth = data["depths"][vi].numpy()    # (H, W)
        valid = data["valid"][vi].numpy()     # (H, W) bool
        K     = data["K"][vi]                 # (3, 3)
        c2w   = data["c2w"][vi]               # (4, 4)

        conf_raw = np.load(conf_files[vi])
        if conf_raw.dtype == np.uint8:
            conf_raw = conf_raw.astype(np.float32) / 255.0

        img = imageio.imread(img_paths[vi])[..., :3].astype(np.float32) / 255.0

        # back-project valid surface points and project into this view
        pts = backproject_depth(depth, K, c2w, valid)   # (N, 3)
        pts_t = torch.from_numpy(pts).float()
        w2c = torch.linalg.inv(torch.from_numpy(c2w).float())
        xc  = pts_t @ w2c[:3, :3].T + w2c[:3, 3]
        K_t = torch.from_numpy(K).float()
        uvh = xc @ K_t.T
        uv  = uvh[:, :2] / uvh[:, 2:3].clamp_min(1e-6)
        u, v_ = uv[:, 0].numpy(), uv[:, 1].numpy()
        H, W  = depth.shape
        m = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H) & (xc[:, 2] > 1e-4).numpy()
        depth_proj = xc[:, 2].numpy()[m]

        # row 0: depth map
        d_show = np.where(valid, depth, np.nan)
        im0 = axes[0, col].imshow(d_show, cmap="plasma")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
        axes[0, col].set_title(f"view {vi}  depth")
        axes[0, col].axis("off")

        # row 1: confidence map
        im1 = axes[1, col].imshow(conf_raw, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
        axes[1, col].axhline(0, color="r", lw=0)  # dummy for layout
        axes[1, col].set_title(f"confidence (thr={args.conf_thresh})")
        axes[1, col].axis("off")

        # row 2: valid mask
        axes[2, col].imshow(valid, cmap="gray")
        axes[2, col].set_title(f"valid  ({valid.mean()*100:.1f}%)")
        axes[2, col].axis("off")

        # row 3: surface pts overlaid on image
        d_lo = np.percentile(depth_proj, 2)
        d_hi = np.percentile(depth_proj, 98)
        d_norm = np.clip((depth_proj - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1)
        axes[3, col].imshow(img)
        axes[3, col].scatter(u[m], v_[m], c=d_norm, cmap="plasma",
                             s=0.2, alpha=0.4, linewidths=0)
        axes[3, col].set_title(f"{m.sum()} pts")
        axes[3, col].axis("off")

    for row, label in enumerate(["depth", "confidence", "valid mask", "pts overlay"]):
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=50, va="center", fontsize=9)

    fig.suptitle(f"MVSFormer++ depths — skull scan65  (conf>{args.conf_thresh})", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
