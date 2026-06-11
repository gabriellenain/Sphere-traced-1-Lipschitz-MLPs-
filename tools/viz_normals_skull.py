"""Visualize GT camera-space normal maps for DTU scan65 (skull).

Shows a grid of views: RGB | normal (camera-space color) | normal (world-space color)
World-space normals should be globally coherent across views.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SKULL_SCENE = Path("/scratch/_projets_/willow/1-lip-tracer/data/DTU/scan65")

from lip_tracer.geomvs import load_aligned_depths
from lip_tracer.data import load_views


def decode_normal(n_01: np.ndarray) -> np.ndarray:
    """(H,W,3) in [0,1] → unit vectors in [-1,1]."""
    n = n_01 * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
    return n / norm


def normal_to_rgb(n: np.ndarray) -> np.ndarray:
    """Unit normals [-1,1] → display RGB [0,1]."""
    return (n * 0.5 + 0.5).clip(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SKULL_SCENE)
    ap.add_argument("--views", type=str, default="0,8,16,24,32,40",
                    help="comma-separated view indices to show")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/skull_normals.png"))
    args = ap.parse_args()

    view_ids = [int(v) for v in args.views.split(",")]

    print("Loading views and normals …")
    data   = load_aligned_depths(args.scene)
    views  = load_views(args.scene)

    n_cols = len(view_ids)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, None]

    for col, vi in enumerate(view_ids):
        img = views["images"][vi].numpy()
        c2w = views["c2w"][vi].numpy()   # (4,4)

        n_raw = data["normals"][vi].numpy()   # (H,W,3) in [0,1]
        valid = data["valid"][vi].numpy()

        n_cam = decode_normal(n_raw)           # camera-space unit normals

        # rotate to world space
        R = c2w[:3, :3]                        # (3,3)
        n_world = (n_cam @ R.T)                # (H,W,3)
        norm = np.linalg.norm(n_world, axis=-1, keepdims=True).clip(1e-6)
        n_world = n_world / norm

        n_cam_rgb   = normal_to_rgb(n_cam)
        n_world_rgb = normal_to_rgb(n_world)

        # mask background
        mask3 = valid[..., None]
        n_cam_rgb   = np.where(mask3, n_cam_rgb,   np.zeros_like(n_cam_rgb))
        n_world_rgb = np.where(mask3, n_world_rgb, np.zeros_like(n_world_rgb))

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"view {vi}")
        axes[0, col].axis("off")

        axes[1, col].imshow(n_cam_rgb)
        axes[1, col].set_title("normal (cam)")
        axes[1, col].axis("off")

        axes[2, col].imshow(n_world_rgb)
        axes[2, col].set_title("normal (world)")
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("RGB", rotation=0, labelpad=40, va="center")
    axes[1, 0].set_ylabel("cam-space", rotation=0, labelpad=40, va="center")
    axes[2, 0].set_ylabel("world-space", rotation=0, labelpad=40, va="center")

    fig.suptitle("DTU scan65 (skull) — GT normals coherence check", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
