#!/usr/bin/env python3
"""Visualise sfm_pts (GT surface points) for a blender/lego scene.

Produces:
  - sfm_pts.ply      : point cloud for MeshLab / CloudCompare
  - sfm_pts_views.png: matplotlib scatter from 4 viewpoints

Usage:
    python viz_sfm_pts.py [run_dir_or_scene_dir] [--n-pts 30000] [--out-dir outputs/]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def _save_ply(pts: np.ndarray, path: Path) -> None:
    N = len(pts)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"end_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(pts.astype(np.float32).tobytes())
    print(f"  saved {N:,} pts → {path}")


def _scatter_views(pts: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 4))
    views = [
        ("front  (XY)", (0,  90)),
        ("side   (YZ)", (0,   0)),
        ("top    (XZ)", (90, 90)),
        ("iso",         (30, 45)),
    ]
    step = max(1, len(pts) // 4096)
    p = pts[::step]
    for i, (title, (elev, azim)) in enumerate(views, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.3, c=p[:, 2],
                   cmap="viridis", linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved views → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", type=Path, nargs="?", default=None,
                    help="run_dir (with config.json) or scene dir directly")
    ap.add_argument("--n-pts", type=int, default=30000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    target = args.target
    if target is not None and (target / "config.json").exists():
        from lip_tracer.train import load_config_json
        cfg = load_config_json(target / "config.json")
        scene = cfg.scene
        out_dir = args.out_dir or target
    elif target is not None:
        scene = target
        out_dir = args.out_dir or Path("outputs")
    else:
        # default: most recent lego run
        runs = sorted(Path("outputs").glob("run_*_lego"))
        if not runs:
            ap.error("no run_*_lego dir found; pass a run_dir or scene dir")
        from lip_tracer.train import load_config_json
        cfg = load_config_json(runs[-1] / "config.json")
        scene = cfg.scene
        out_dir = args.out_dir or runs[-1]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  scene : {scene}")
    print(f"  out   : {out_dir}")

    from lip_tracer.data import load_blender_gt_points
    print(f"  loading up to {args.n_pts:,} GT surface points …")
    pts = load_blender_gt_points(scene=scene, n_pts=args.n_pts).numpy()

    print(f"  shape : {pts.shape}")
    print(f"  x     : [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  y     : [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  z     : [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
    print(f"  |r|   : mean={np.linalg.norm(pts, axis=1).mean():.3f}")
    print(f"  std   : x={pts[:,0].std():.4f}  y={pts[:,1].std():.4f}  z={pts[:,2].std():.4f}")

    # nearest-neighbour distance on a random subset — proxy for local noise/density
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(2000, len(pts)), replace=False)
    sub = pts[idx]
    dist = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    print(f"  NN dist (2k sample): mean={nn.mean():.4f}  p90={np.percentile(nn,90):.4f}  max={nn.max():.4f}")
    print(f"  → {'likely noisy / sparse' if nn.mean() > 0.05 else 'dense and clean'}")

    _save_ply(pts, out_dir / "sfm_pts.ply")
    _scatter_views(pts, out_dir / "sfm_pts_views.png")


if __name__ == "__main__":
    main()
