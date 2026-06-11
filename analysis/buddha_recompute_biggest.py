#!/usr/bin/env python3
"""Recompute Buddha-sweep Chamfer keeping only each mesh's largest component.

The marching-cubes meshes of the high-frequency PE configs grow spurious
disconnected blobs that inflate Chamfer and Hausdorff unfairly. For every
config this keeps only the largest connected component (by surface area),
writes it as pred_mesh_main.ply, recomputes the bidirectional Chamfer against
the GT mesh (same protocol as fit_gt_sdf.compute_chamfer: 100k surface samples,
seed 0), and writes results_biggest.csv alongside the original results.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SWEEP = Path("outputs/buddha_sweep")
GT_MESH = Path("data/gt_meshes/happy_buddha_norm.ply")


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Connected component with the greatest surface area."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: m.area)


def chamfer(gt_pts: np.ndarray, gt_tree: cKDTree, pred_mesh: trimesh.Trimesh,
            n_points: int, seed: int) -> dict:
    """Bidirectional Chamfer + Hausdorff, mesh units (compute_chamfer protocol)."""
    pred_pts = np.asarray(pred_mesh.sample(n_points), np.float32)
    pred_tree = cKDTree(pred_pts)
    acc_d, _ = pred_tree.query(gt_pts, workers=-1)        # completeness gt->pred
    comp_d, _ = gt_tree.query(pred_pts, workers=-1)       # accuracy     pred->gt
    accuracy, completeness = float(comp_d.mean()), float(acc_d.mean())
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer": 0.5 * (accuracy + completeness),
        "hausdorff": float(max(comp_d.max(), acc_d.max())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-points", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gt = trimesh.load(GT_MESH, process=False)
    gt_pts = np.asarray(gt.sample(args.n_points), np.float32)
    gt_tree = cKDTree(gt_pts)

    rows = list(csv.DictReader(open(SWEEP / "results.csv")))
    out_rows = []
    for r in rows:
        d = Path(r["out_dir"])
        raw = trimesh.load(d / "pred_mesh.ply", process=False)
        main_mesh = largest_component(raw)
        main_mesh.export(d / "pred_mesh_main.ply")
        cd = chamfer(gt_pts, gt_tree, main_mesh, args.n_points, args.seed)
        kept = 100.0 * len(main_mesh.faces) / max(len(raw.faces), 1)
        out = dict(r)
        out.update({k: cd[k] for k in ("chamfer", "accuracy", "completeness", "hausdorff")})
        out["n_verts"], out["n_faces"] = len(main_mesh.vertices), len(main_mesh.faces)
        out_rows.append(out)
        print(f"{r['out_dir'].split('/')[-1]:<18} "
              f"chamfer {float(r['chamfer']):.5f} -> {cd['chamfer']:.5f}   "
              f"hausdorff {float(r['hausdorff']):.4f} -> {cd['hausdorff']:.4f}   "
              f"(kept {kept:.1f}% of faces)")

    out_csv = SWEEP / "results_biggest.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nsaved {out_csv}  ({len(out_rows)} configs)")


if __name__ == "__main__":
    main()
