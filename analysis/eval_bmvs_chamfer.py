#!/usr/bin/env python3
"""BlendedMVS Chamfer evaluation against the raw GT meshes.

BlendedMVS has no DTU-style official ObsMask/Plane protocol. Two protocols are
supported via --protocol:

ProbeSDF-style (default):
  1. Put prediction and GT into the NeuS/IDR normalized frame using
     cameras_sphere.npz scale_mat_0.
  2. Compute bidirectional vertex-to-surface distances using pysdf, exactly as
     ProbeSDF's BMVS script does.
  3. Ignore distances above 0.025 normalized units when averaging.

Note: these GT meshes (data/bmvs_gt/meshes/BMVS/<scene>/GroundTruth.ply) ship in
the NORMALIZED frame, so pass --gt-space normalized; --mask-crop additionally
removes background geometry from nomask runs via a DTU-style visual-hull test.

VolSDF (paper supplementary B.2): "We used the ground truth meshes supplied by
the authors to evaluate the Chamfer l1 distances from the output surfaces. For
each mesh we evaluated the largest connected surface component above the ground
plane. To measure the Chamfer l1 distance we used 100K random point samples
from each surface." => 100K samples, no outlier threshold, prediction reduced to
its largest connected component above the ground plane (Chamfer l1 = mean of the
two one-sided means). The paper does not define the ground plane, so it is
supplied via --ground-axis/--ground-value (skipped if --ground-value is omitted).

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


def _drop_below_plane(mesh, axis: int, value: float):
    """Keep only faces whose centroid lies on/above `value` along `axis`.

    VolSDF B.2 evaluates the component *above the ground plane*; the paper does
    not define the plane, so it is supplied explicitly (axis + offset)."""
    import trimesh

    tris = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    cent = tris.mean(axis=1)
    keep = cent[:, axis] >= value
    if not np.any(keep):
        return mesh
    out = mesh.copy()
    out.update_faces(keep)
    out.remove_unreferenced_vertices()
    return out


def _largest_component(mesh):
    """Largest connected surface component (by surface area)."""
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh
    return max(comps, key=lambda m: float(m.area))


def _sample(mesh, n: int, seed: int) -> np.ndarray:
    import trimesh

    if n <= 0:
        return np.asarray(mesh.vertices, dtype=np.float32)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return pts.astype(np.float32)


def _mask_hull_keep(pts: np.ndarray, scene_dir: Path, dilate: int,
                    min_views: int, mode: str, min_ratio: float):
    """DTU-style visual-hull crop using the per-view object masks.

    Mirrors the DTU ObsMask crop+dilate: reproject each normalized-frame point
    into every training mask (P = world_mat_i @ scale_mat_i, the IDR/NeuS
    convention for points in the normalized object frame), dilate each mask by
    `dilate` px so points just outside the silhouette are still kept (the analog
    of DTU's voxel margin), then keep points seen as foreground per `mode`.
    Returns (keep_bool, stats)."""
    from PIL import Image

    cam = np.load(scene_dir / "cameras_sphere.npz")
    mask_paths = sorted(p for p in (scene_dir / "mask").glob("*.png")
                        if not p.name.startswith("."))
    if not mask_paths:
        raise FileNotFoundError(f"No mask images found in {scene_dir / 'mask'}")

    try:
        import cv2
        kern = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        _dil = (lambda m: cv2.dilate(m.astype(np.uint8), kern) > 0) if dilate > 0 else (lambda m: m)
    except ImportError:
        from scipy.ndimage import binary_dilation
        _dil = (lambda m: binary_dilation(m, iterations=dilate)) if dilate > 0 else (lambda m: m)

    fg = np.zeros(len(pts), dtype=np.uint16)
    seen = np.zeros(len(pts), dtype=np.uint16)
    pts_h = np.concatenate([pts.astype(np.float64),
                            np.ones((len(pts), 1))], axis=1).T  # 4xN

    for i, mp in enumerate(mask_paths):
        wkey, skey = f"world_mat_{i}", f"scale_mat_{i}"
        if wkey not in cam:
            continue
        P = cam[wkey].astype(np.float64)
        if skey in cam:
            P = P @ cam[skey].astype(np.float64)
        m = np.array(Image.open(mp))
        m = (m[..., 0] if m.ndim == 3 else m) > 0
        m = _dil(m)
        H, W = m.shape[:2]

        proj = P @ pts_h
        w = proj[2]
        pos = w > 0
        ui = np.round(np.where(pos, proj[0] / np.where(pos, w, 1.0), -1.0)).astype(np.int64)
        vi = np.round(np.where(pos, proj[1] / np.where(pos, w, 1.0), -1.0)).astype(np.int64)
        inb = pos & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        idx = np.where(inb)[0]
        if not len(idx):
            continue
        seen[idx] += 1
        fg[idx[m[vi[idx], ui[idx]]]] += 1

    min_views = max(1, int(min_views))
    seen_ok = seen >= min_views
    if mode == "any":
        keep = fg >= min_views
    elif mode == "all":
        keep = seen_ok & (fg == seen)
    elif mode == "ratio":
        ratio = np.divide(fg, np.maximum(seen, 1), out=np.zeros(len(pts), np.float32),
                          where=seen > 0)
        keep = seen_ok & (ratio >= float(min_ratio))
    else:
        raise ValueError(f"unknown mask mode: {mode}")
    stats = {"mode": mode, "dilate": int(dilate), "min_views": min_views,
             "min_ratio": float(min_ratio), "n_kept": int(keep.sum()),
             "n_total": int(len(pts)), "frac_kept": float(keep.mean())}
    return keep, stats


def _dists_to_mesh_pysdf(query: np.ndarray, mesh) -> np.ndarray:
    """ProbeSDF's distance primitive: pysdf distance to a mesh surface.

    ProbeSDF evaluates this at the opposite mesh's *vertices* for BMVS.

    pysdf is not installed in the training venv, so fall back to the open3d
    point-to-mesh surface distance (`_dists_to_mesh`) — the same primitive the
    pre-06-29 BMVS runs used — when pysdf is unavailable.
    """
    try:
        from pysdf import SDF
    except ImportError:
        return _dists_to_mesh(query, mesh)

    sdf = SDF(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return np.abs(sdf(query.astype(np.float64))).astype(np.float64)


def _dists_to_mesh(query: np.ndarray, mesh) -> np.ndarray:
    """Exact unsigned distance from each query point to the mesh *surface*
    (open3d RaycastingScene BVH). This diagnostic metric avoids the positive
    bias of point-to-point NN on finite surface samples."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32),
    )
    q = o3d.core.Tensor(query.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    return scene.compute_distance(q).numpy().astype(np.float64)


def _dists_to_points(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Point-to-point nearest-neighbour distance (legacy metric, biased high)."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(ref).query(query, k=1, workers=-1)
    return d


def _nn_metrics(acc: np.ndarray, comp: np.ndarray, max_dist: float | None, ignore: bool):
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
    ap.add_argument("--protocol", choices=["probesdf", "volsdf"], default="probesdf",
                    help="probesdf: ProbeSDF vertex-to-mesh pysdf eval, ignore dist>0.025. "
                         "volsdf (paper B.2): 100K samples, no outlier threshold, "
                         "largest connected component above the ground plane.")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--n-points", type=int, default=None,
                    help="samples per surface (default: 1M for probesdf, 100K for volsdf)")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="outlier threshold in normalized units "
                         "(default: 0.025 for probesdf, disabled for volsdf)")
    ap.add_argument("--ground-axis", type=int, default=2,
                    help="volsdf: axis (0=x,1=y,2=z) normal to the ground plane")
    ap.add_argument("--ground-value", type=float, default=None,
                    help="volsdf: drop geometry below this offset along --ground-axis "
                         "(in the comparison frame); omit to skip ground-plane removal")
    ap.add_argument("--mask-crop", action="store_true",
                    help="DTU-style crop: keep only sampled points inside the "
                         "per-view object masks (visual hull), applied to BOTH "
                         "prediction and GT (like the DTU ObsMask). Removes "
                         "background slabs from nomask reconstructions.")
    ap.add_argument("--mask-dilate", type=int, default=12,
                    help="px dilation of each mask before the hull test "
                         "(analog of the DTU ObsMask margin; default 12)")
    ap.add_argument("--mask-min-views", type=int, default=1)
    ap.add_argument("--mask-mode", choices=["any", "all", "ratio"], default="all",
                    help="visual-hull test: 'all' keeps points foreground in every "
                         "view that sees them (proper hull); 'any' (>=1 view) does "
                         "NOT carve background and is rarely what you want")
    ap.add_argument("--mask-min-ratio", type=float, default=0.95)
    ap.add_argument("--metric", choices=["auto", "probesdf", "point-to-mesh", "point-to-point"],
                    default="auto",
                    help="auto: probesdf protocol uses ProbeSDF's pysdf vertex-to-mesh "
                         "eval; volsdf keeps sampled point-to-mesh. probesdf: exact "
                         "ProbeSDF BMVS distance mode. point-to-mesh: Open3D surface "
                         "distance on random samples. point-to-point: legacy KDTree NN "
                         "on samples (biased high, shrinks with more samples)")
    ap.add_argument("--clip", action="store_true",
                    help="clip large distances instead of ignoring them")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-meshes", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    n_points = args.n_points if args.n_points is not None else (100_000 if args.protocol == "volsdf" else 1_000_000)
    if args.max_dist is not None:
        max_dist = args.max_dist
    else:
        max_dist = None if args.protocol == "volsdf" else 0.025

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

    metric = args.metric
    if metric == "auto":
        metric = "probesdf" if args.protocol == "probesdf" else "point-to-mesh"

    if args.protocol == "volsdf":
        # B.2: "the largest connected surface component above the ground plane".
        if args.ground_value is not None:
            pred_mesh = _drop_below_plane(pred_mesh, args.ground_axis, args.ground_value)
        else:
            print("[volsdf] --ground-value not set: skipping ground-plane removal "
                  "(paper does not define the plane; largest component still applied)",
                  file=sys.stderr)
        pred_mesh = _largest_component(pred_mesh)

    if metric == "probesdf":
        pred_pts = np.asarray(pred_mesh.vertices, dtype=np.float32)
        gt_pts = np.asarray(gt_mesh.vertices, dtype=np.float32)
    else:
        pred_pts = _sample(pred_mesh, n_points, args.seed)
        gt_pts = _sample(gt_mesh, n_points, args.seed + 1)
    mask_stats = None
    if args.mask_crop:
        keep_p, sp = _mask_hull_keep(pred_pts, args.scene, args.mask_dilate,
                                     args.mask_min_views, args.mask_mode, args.mask_min_ratio)
        keep_g, sg = _mask_hull_keep(gt_pts, args.scene, args.mask_dilate,
                                     args.mask_min_views, args.mask_mode, args.mask_min_ratio)
        pred_pts, gt_pts = pred_pts[keep_p], gt_pts[keep_g]
        mask_stats = {"pred": sp, "gt": sg}
        print(f"[mask-crop] pred kept {sp['n_kept']}/{sp['n_total']} "
              f"({sp['frac_kept']:.1%}), gt kept {sg['n_kept']}/{sg['n_total']} "
              f"({sg['frac_kept']:.1%})", file=sys.stderr)
    if metric == "probesdf":
        acc = _dists_to_mesh_pysdf(pred_pts, gt_mesh)    # pred vertices -> GT surface
        comp = _dists_to_mesh_pysdf(gt_pts, pred_mesh)   # GT vertices -> pred surface
    elif metric == "point-to-mesh":
        acc = _dists_to_mesh(pred_pts, gt_mesh)    # pred surface -> GT surface
        comp = _dists_to_mesh(gt_pts, pred_mesh)   # GT surface -> pred surface
    else:
        acc = _dists_to_points(pred_pts, gt_pts)
        comp = _dists_to_points(gt_pts, pred_pts)
    metrics = _nn_metrics(acc, comp, max_dist, ignore=not args.clip)
    metrics.update({
        "scene": str(args.scene),
        "gt_mesh": str(gt_mesh_path),
        "protocol": args.protocol,
        "metric": metric,
        "mesh_space": args.mesh_space,
        "gt_space": args.gt_space,
        "n_pred_points": int(len(pred_pts)),
        "n_gt_points": int(len(gt_pts)),
        "max_dist_normalized": max_dist,
        "outlier_mode": ("none" if max_dist is None else ("clip" if args.clip else "ignore")),
        "ground_value": args.ground_value if args.protocol == "volsdf" else None,
        "mask_crop": mask_stats,
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
