"""Generate sparse_sfm_points.txt + per-frame files + meta_data.json for IDR scenes.

Usage:
    python precompute_sfm_pairs_idr.py --scene <idr_scan_dir> \
        --old-scene <old_dtu_scan_dir>   # source of COLMAP points (same scan)

The old-scene COLMAP points are re-normalized into IDR space via their respective
scale_mat_0 matrices (same raw COLMAP reconstruction, different normalization).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lip_tracer.data import load_views


def transform_sfm_points(old_scene: Path, idr_scene: Path) -> np.ndarray:
    """Load old-normalized SFM points and re-normalize to IDR space."""
    d_old = np.load(old_scene / "cameras.npz")
    d_idr = np.load(idr_scene / "cameras.npz")
    old_scale_mat     = d_old["scale_mat_0"].astype(np.float64)
    idr_scale_mat_inv = np.linalg.inv(d_idr["scale_mat_0"].astype(np.float64))
    pts = np.loadtxt(old_scene / "sparse_sfm_points.txt", dtype=np.float64)
    ones = np.ones((len(pts), 1))
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_idr = (idr_scale_mat_inv @ (old_scale_mat @ pts_h.T)).T[:, :3].astype(np.float32)
    return pts_idr


def visible_points(pts: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                   H: int, W: int) -> np.ndarray:
    """Return subset of pts (N,3) visible from this camera."""
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pc = pts @ R.T + t                        # (N,3) camera-space
    mask = pc[:, 2] > 1e-4
    uv = (pc @ K.T)                           # (N,3)
    uv = uv[:, :2] / uv[:, 2:3].clip(1e-6)
    mask &= (uv[:, 0] >= 0) & (uv[:, 0] < W)
    mask &= (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return pts[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",     type=Path, required=True,
                    help="IDR scene directory (cameras.npz + image/ + mask/)")
    ap.add_argument("--old-scene", type=Path, required=True,
                    help="Old DTU scene with sparse_sfm_points.txt")
    args = ap.parse_args()

    scene: Path = args.scene
    old_scene: Path = args.old_scene

    print("Loading IDR views …")
    views = load_views(scene)
    V   = views["c2w"].shape[0]
    H   = views["H"]
    W   = views["W"]
    c2w = views["c2w"].numpy()   # (V, 4, 4)
    K   = views["K"].numpy()     # (V, 3, 3)

    print("Transforming SFM points to IDR space …")
    pts = transform_sfm_points(old_scene, scene)
    print(f"  {len(pts)} points, range {pts.min(axis=0)} .. {pts.max(axis=0)}")

    out_pts = scene / "sparse_sfm_points.txt"
    np.savetxt(out_pts, pts, fmt="%.8f")
    print(f"  saved {out_pts}")

    print("Computing per-frame visibility …")
    frames = []
    total_pairs = 0
    for v in range(V):
        vis = visible_points(pts, c2w[v], K[v], H, W)
        fname = f"{v:06d}_sfm_points.txt"
        if len(vis) > 0:
            np.savetxt(scene / fname, vis, fmt="%.8f")
        else:
            np.savetxt(scene / fname, np.zeros((1, 3), dtype=np.float32), fmt="%.8f")
        total_pairs += len(vis)
        frames.append({
            "sfm_sparse_points_view": fname,
            "camtoworld": c2w[v].tolist(),
        })

    meta = {"frames": frames}
    (scene / "sfm_pairs.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved sfm_pairs.json  ({V} frames, {total_pairs} total pairs)")


if __name__ == "__main__":
    main()
