#!/usr/bin/env python3
"""BlendedMVS Chamfer evaluation against the raw GT meshes.

BlendedMVS has no DTU-style official ObsMask/Plane protocol. The common
ProbeSDF-style protocol is:

  1. Put prediction and GT into the NeuS/IDR normalized frame using
     cameras_sphere.npz scale_mat_0.
  2. Compute bidirectional surface distances.
  3. Ignore distances above 0.025 normalized units when averaging.

For this repo, checkpoint marching-cubes output is already in the normalized
training frame, so only the raw GT mesh is transformed by inv(scale_mat_0).
Use --mesh-space world only when feeding a prediction in the raw BMVS frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


GT_REL = {
    "bmvs_dog": "20-dog/GTMeshRaw.ply",
    "bmvs_bear": "18-bear/GTMeshRaw.ply",
    "bmvs_clock": "3-clock/GTMeshRaw.ply",
    "bmvs_durian": "21-durian/GTMeshRaw.ply",
    "bmvs_man": "19-man/GTMeshRaw.ply",
    "bmvs_sculpture": "14-sculpture/GTMeshRaw.ply",
    "bmvs_stone": "8-stone/GTMeshRaw.ply",
    "bmvs_jade": "22-jade/GTMeshRaw.ply",
}


def _load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    mesh.remove_unreferenced_vertices()
    return mesh


def _transform_mesh(mesh, mat: np.ndarray):
    import trimesh

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts_h = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1)
    verts_t = (mat @ verts_h.T).T[:, :3]
    return trimesh.Trimesh(vertices=verts_t, faces=np.asarray(mesh.faces), process=False)


def _extract_ckpt_mesh(ckpt: Path, bound: float, res: int, device: str):
    import trimesh
    from compare_dtu_chamfer import _extract_mesh_from_model

    verts, faces, _ = _extract_mesh_from_model(ckpt, bound=bound, res=res, device=device)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _sample(mesh, n: int, seed: int) -> np.ndarray:
    import trimesh

    if n <= 0:
        return np.asarray(mesh.vertices, dtype=np.float32)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return pts.astype(np.float32)


def _nn_metrics(pred: np.ndarray, gt: np.ndarray, max_dist: float | None, ignore: bool):
    from scipy.spatial import cKDTree

    acc, _ = cKDTree(gt).query(pred, k=1, workers=-1)
    comp, _ = cKDTree(pred).query(gt, k=1, workers=-1)
    if max_dist is not None and ignore:
        acc_mean = float(acc[acc < max_dist].mean()) if np.any(acc < max_dist) else float("nan")
        comp_mean = float(comp[comp < max_dist].mean()) if np.any(comp < max_dist) else float("nan")
    elif max_dist is not None:
        acc_mean = float(np.minimum(acc, max_dist).mean())
        comp_mean = float(np.minimum(comp, max_dist).mean())
    else:
        acc_mean = float(acc.mean())
        comp_mean = float(comp.mean())
    return {
        "accuracy": acc_mean,
        "completeness": comp_mean,
        "chamfer": 0.5 * (acc_mean + comp_mean),
        "accuracy_p50": float(np.median(acc)),
        "accuracy_p90": float(np.percentile(acc, 90)),
        "completeness_p50": float(np.median(comp)),
        "completeness_p90": float(np.percentile(comp, 90)),
        "n_accuracy_kept": int((acc < max_dist).sum()) if max_dist is not None else int(len(acc)),
        "n_completeness_kept": int((comp < max_dist).sum()) if max_dist is not None else int(len(comp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path)
    src.add_argument("--mesh", type=Path)
    ap.add_argument("--scene", type=Path, required=True,
                    help="data/bmvs/<scene>, containing cameras_sphere.npz")
    ap.add_argument("--gt-mesh", type=Path, default=None,
                    help="raw BlendedMVS GTMeshRaw.ply")
    ap.add_argument("--gt-root", type=Path, default=None,
                    help="directory containing ProbeSDF-style GT_meshes/<relpath> or the relpaths directly")
    ap.add_argument("--mesh-space", choices=["normalized", "world"], default="normalized")
    ap.add_argument("--gt-space", choices=["world", "normalized"], default="world")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--n-points", type=int, default=1_000_000)
    ap.add_argument("--max-dist", type=float, default=0.025,
                    help="distance threshold in normalized units; ProbeSDF ignores larger distances")
    ap.add_argument("--clip", action="store_true",
                    help="clip large distances instead of ignoring them")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-meshes", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cam_path = args.scene / "cameras_sphere.npz"
    if not cam_path.exists():
        raise FileNotFoundError(f"missing {cam_path}")
    scale_mat = np.load(cam_path)["scale_mat_0"].astype(np.float64)
    scale_mat_inv = np.linalg.inv(scale_mat)
    raw_units_per_norm = float(np.linalg.norm(scale_mat[:3, :3], axis=0)[0])

    gt_mesh_path = args.gt_mesh
    if gt_mesh_path is None:
        rel = GT_REL.get(args.scene.name)
        if rel is None or args.gt_root is None:
            ap.error("--gt-mesh is required, or pass --gt-root for a known bmvs_* scene")
        gt_mesh_path = args.gt_root / rel
        if not gt_mesh_path.exists():
            alt = args.gt_root / "GT_meshes" / rel
            gt_mesh_path = alt if alt.exists() else gt_mesh_path
    if not gt_mesh_path.exists():
        raise FileNotFoundError(f"missing GT mesh: {gt_mesh_path}")

    pred_mesh = _extract_ckpt_mesh(args.ckpt, args.bound, args.res, device) if args.ckpt else _load_mesh(args.mesh)
    if args.mesh_space == "world":
        pred_mesh = _transform_mesh(pred_mesh, scale_mat_inv)

    gt_mesh = _load_mesh(gt_mesh_path)
    if args.gt_space == "world":
        gt_mesh = _transform_mesh(gt_mesh, scale_mat_inv)

    pred_pts = _sample(pred_mesh, args.n_points, args.seed)
    gt_pts = _sample(gt_mesh, args.n_points, args.seed + 1)
    metrics = _nn_metrics(pred_pts, gt_pts, args.max_dist, ignore=not args.clip)
    metrics.update({
        "scene": str(args.scene),
        "gt_mesh": str(gt_mesh_path),
        "mesh_space": args.mesh_space,
        "gt_space": args.gt_space,
        "n_pred_points": int(len(pred_pts)),
        "n_gt_points": int(len(gt_pts)),
        "max_dist_normalized": args.max_dist,
        "outlier_mode": "clip" if args.clip else "ignore",
        "raw_units_per_normalized_unit": raw_units_per_norm,
        "accuracy_x1000": metrics["accuracy"] * 1000.0,
        "completeness_x1000": metrics["completeness"] * 1000.0,
        "chamfer_x1000": metrics["chamfer"] * 1000.0,
        "accuracy_raw_units": metrics["accuracy"] * raw_units_per_norm,
        "completeness_raw_units": metrics["completeness"] * raw_units_per_norm,
        "chamfer_raw_units": metrics["chamfer"] * raw_units_per_norm,
    })

    print(json.dumps(metrics, indent=2))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "bmvs_chamfer.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.save_meshes:
            pred_mesh.export(args.out / "pred_normalized.ply")
            gt_mesh.export(args.out / "gt_normalized.ply")


if __name__ == "__main__":
    main()
