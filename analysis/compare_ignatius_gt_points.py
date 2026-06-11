#!/usr/bin/env python3
"""Compare Ignatius carved SDF grids against official TnT GT points.

The sweep SDFs live in the repo's normalized TnT/NSVF frame.  Official TnT GT
points live in the GT-LiDAR frame, so this script applies:

    GT-LiDAR --inv(Ignatius_trans.txt)--> COLMAP-SfM --bbox normalize--> NSVF

Then it samples every p-sweep SDF at the transformed GT points and reports how
many GT points are inside the carved volume.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
import trimesh


def load_ply_points(path: Path) -> np.ndarray:
    obj = trimesh.load(path, process=False)
    if hasattr(obj, "vertices"):
        pts = np.asarray(obj.vertices, dtype=np.float64)
    elif hasattr(obj, "geometry"):
        pts = np.concatenate(
            [np.asarray(g.vertices, dtype=np.float64) for g in obj.geometry.values()],
            axis=0,
        )
    else:
        raise TypeError(f"could not read points from {path}: {type(obj)!r}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"bad point array from {path}: {pts.shape}")
    return pts


def gt_to_normalized(gt_pts: np.ndarray, gt_dir: Path, scene_dir: Path,
                     scene_name: str) -> np.ndarray:
    trans = np.loadtxt(gt_dir / f"{scene_name}_trans.txt", dtype=np.float64)
    if trans.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {trans.shape}")
    homog = np.concatenate([gt_pts, np.ones((len(gt_pts), 1), dtype=np.float64)], axis=1)
    colmap = (np.linalg.inv(trans) @ homog.T).T[:, :3]

    bbox = np.loadtxt(scene_dir / "bbox.txt", dtype=np.float64).reshape(-1)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    return (colmap - center[None]) / scale


def sample_sdf(sdf: np.ndarray, pts: np.ndarray, bound: float) -> tuple[np.ndarray, np.ndarray]:
    res = int(sdf.shape[0])
    grid = (pts + bound) / (2.0 * bound) * (res - 1)
    inb = np.all((grid >= 0.0) & (grid <= (res - 1)), axis=1)
    coords = np.stack([grid[inb, 2], grid[inb, 1], grid[inb, 0]], axis=0)
    vals = np.full(len(pts), np.inf, dtype=np.float32)
    vals[inb] = map_coordinates(
        sdf.astype(np.float32, copy=False),
        coords,
        order=1,
        mode="constant",
        cval=np.inf,
    )
    return vals, inb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, default=Path("data/tnt/Ignatius"))
    ap.add_argument("--gt-dir", type=Path, default=Path("data/tnt_gt/Ignatius"))
    ap.add_argument("--scene-name", default="Ignatius")
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path("_diagnostics/mvsformer_ignatius_gsam_statue_lowthr_p_sweep"))
    ap.add_argument("--sdf-name", default="sdf_vh_mvsformer_carved.npy")
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--out", type=Path,
                    default=Path("_diagnostics/mvsformer_ignatius_gsam_statue_lowthr_p_sweep/gt_point_recall.csv"))
    args = ap.parse_args()

    gt_raw = load_ply_points(args.gt_dir / f"{args.scene_name}.ply")
    gt = gt_to_normalized(gt_raw, args.gt_dir, args.scene_dir, args.scene_name)
    print(f"GT points: {len(gt):,}")
    print(f"GT normalized AABB: min={gt.min(axis=0)} max={gt.max(axis=0)}")

    rows = []
    for sdf_path in sorted(args.sweep_dir.glob(f"p*_res*/{args.sdf_name}")):
        run_dir = sdf_path.parent
        summary = json.loads((run_dir / "summary.json").read_text())
        p = float(summary["settings"]["vh_percentile"])
        sdf = np.load(sdf_path)
        res = int(sdf.shape[0])
        voxel = 2.0 * args.bound / max(res - 1, 1)
        vals, inb = sample_sdf(sdf, gt, args.bound)
        volume_voxels = int(np.sum(sdf <= 0.0))
        finite = np.isfinite(vals)
        if finite.any():
            vals_f = vals[finite]
            mean_pos = float(np.maximum(vals_f, 0.0).mean())
            p95_pos = float(np.quantile(np.maximum(vals_f, 0.0), 0.95))
            max_pos = float(np.maximum(vals_f, 0.0).max())
        else:
            mean_pos = p95_pos = max_pos = float("inf")

        row = {
            "p": p,
            "hull_voxels": int(summary["hull_voxels"]),
            "final_voxels": int(summary["voxels_final"]),
            "volume_voxels": volume_voxels,
            "gt_in_bounds": int(inb.sum()),
            "gt_total": int(len(gt)),
            "recall_exact": float(np.mean(vals <= 0.0)),
            "recall_1vox": float(np.mean(vals <= voxel)),
            "recall_2vox": float(np.mean(vals <= 2.0 * voxel)),
            "mean_positive_sdf": mean_pos,
            "p95_positive_sdf": p95_pos,
            "max_positive_sdf": max_pos,
            "sdf": str(sdf_path),
            "compare_png": summary["outputs"]["compare_png"],
        }
        rows.append(row)

    rows.sort(key=lambda r: r["p"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\np     exact   +1vox   +2vox   volume_vox  p95_out")
    for r in rows:
        print(f"{r['p']:0.2f}  {r['recall_exact']:6.3f}  {r['recall_1vox']:6.3f}  "
              f"{r['recall_2vox']:6.3f}  {r['volume_voxels']:10d}  "
              f"{r['p95_positive_sdf']:.4f}")

    max_r = max(r["recall_1vox"] for r in rows)
    candidates = [r for r in rows if r["recall_1vox"] >= max_r - 0.005]
    best = min(candidates, key=lambda r: r["volume_voxels"])
    print(f"\nBest compact high-recall p: {best['p']:0.2f} "
          f"(recall@1vox={best['recall_1vox']:.3f}, volume_voxels={best['volume_voxels']})")
    print(f"CSV -> {args.out}")


if __name__ == "__main__":
    main()
