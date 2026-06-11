#!/usr/bin/env python3
"""Make a watertight mesh from the official DTU filtered GT point cloud.

The input point cloud is exactly the one used by analysis/eval_dtu_official.py for
completeness: GT STL vertices after ObsMask and Plane filtering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_dtu_official import _load_gt_filtered


def _subsample_points(
    pts: np.ndarray,
    normals: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(pts) <= max_points:
        return pts, normals
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), max_points, replace=False)
    return pts[idx], normals[idx]


def _largest_component(mesh):
    import open3d as o3d

    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) <= 1:
        return mesh
    keep_label = int(np.argmax(counts))
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(labels != keep_label)
    out.remove_unreferenced_vertices()
    return out


def _clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _render_view(tm, basis: np.ndarray, res: int) -> np.ndarray:
    import trimesh

    verts = np.asarray(tm.vertices, dtype=np.float64) @ basis
    view_mesh = trimesh.Trimesh(vertices=verts, faces=tm.faces, process=False)
    view_mesh.remove_unreferenced_vertices()
    v = np.asarray(view_mesh.vertices, dtype=np.float32)
    center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    v = v - center
    view_mesh.vertices = v
    lim = max(float(np.abs(v).max()), 1e-3)

    xs = np.linspace(-lim, lim, res)
    ys = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(xs, ys[::-1])
    origins = np.stack(
        [xx.ravel(), yy.ravel(), np.full(res * res, -lim * 5.0)], axis=1
    ).astype(np.float32)
    dirs = np.zeros_like(origins)
    dirs[:, 2] = 1.0

    _, idx_ray, idx_tri = view_mesh.ray.intersects_location(
        origins, dirs, multiple_hits=False
    )
    img = np.ones((res * res, 3), dtype=np.float32)
    if len(idx_ray) > 0:
        fn = view_mesh.face_normals[idx_tri].copy()
        fn[fn[:, 2] > 0] *= -1.0
        key = np.array([-0.35, 0.45, -1.0], dtype=np.float64)
        fill = np.array([0.45, -0.2, -0.6], dtype=np.float64)
        key /= np.linalg.norm(key)
        fill /= np.linalg.norm(fill)
        shade = (
            0.78 * (fn @ key).clip(0, 1)
            + 0.20 * (fn @ fill).clip(0, 1)
            + 0.12
        ).clip(0, 1)
        img[idx_ray] = shade[:, None] * np.array([0.72, 0.84, 0.92], dtype=np.float32)
    return img.reshape(res, res, 3).clip(0, 1)


def _view_basis(camera_dir: tuple[float, float, float]) -> np.ndarray:
    z = np.asarray(camera_dir, dtype=np.float64)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(up @ z)) > 0.95:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _save_preview(mesh_path: Path, out_png: Path, res: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    views = [
        ("front", (0.0, 0.0, 1.0)),
        ("right", (1.0, 0.0, 0.0)),
        ("back", (0.0, 0.0, -1.0)),
        ("top", (0.0, 1.0, 0.15)),
    ]
    imgs = [_render_view(tm, _view_basis(d), res) for _, d in views]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="white")
    for ax, img, (label, _) in zip(axes, imgs, views):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"{mesh_path.name}  |  watertight={tm.is_watertight}  "
        f"verts={len(tm.vertices):,} faces={len(tm.faces):,}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _force_watertight_with_voxels(tm, pitch: float | None, max_steps: int = 6):
    """Return a filled voxel marching-cubes wrapper if it becomes watertight."""
    longest_axis = float((tm.bounds[1] - tm.bounds[0]).max())
    cur_pitch = pitch if pitch and pitch > 0 else longest_axis / 90.0
    best = None
    best_pitch = cur_pitch
    for _ in range(max_steps):
        vox = tm.voxelized(cur_pitch).fill()
        wrapped = vox.marching_cubes
        wrapped.apply_transform(vox.transform)
        wrapped.remove_unreferenced_vertices()
        best = wrapped
        best_pitch = cur_pitch
        if wrapped.is_watertight:
            return wrapped, best_pitch, True
        cur_pitch *= 1.25
    return best, best_pitch, bool(best is not None and best.is_watertight)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-id", type=int, required=True)
    ap.add_argument("--dtu-eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.08)
    ap.add_argument("--linear-fit", action="store_true")
    ap.add_argument("--max-points", type=int, default=800_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--density-quantile",
        type=float,
        default=0.0,
        help="Optional low-density vertex trim. Keep 0 to preserve watertightness.",
    )
    ap.add_argument(
        "--crop-to-input-bbox",
        action="store_true",
        help="Optional bbox crop. This usually makes the mesh non-watertight.",
    )
    ap.add_argument("--bbox-margin", type=float, default=0.03)
    ap.add_argument("--largest-component", action="store_true")
    ap.add_argument(
        "--no-force-watertight",
        action="store_true",
        help="Keep the raw Poisson mesh even if it is open.",
    )
    ap.add_argument(
        "--watertight-pitch",
        type=float,
        default=0.0,
        help="Voxel pitch in DTU units for watertight fallback. 0 chooses an automatic pitch.",
    )
    ap.add_argument("--preview-res", type=int, default=512)
    ap.add_argument("--show", action="store_true", help="Open an interactive Open3D window.")
    args = ap.parse_args()

    import open3d as o3d
    import trimesh

    out_mesh = args.out or Path("outputs") / f"dtu_scan{args.scan_id:03d}_gt_watertight.ply"
    out_mesh.parent.mkdir(parents=True, exist_ok=True)

    print("[gt] loading official filtered point cloud...", flush=True)
    gt_obs, gt_above, _, gt_normals, _, _ = _load_gt_filtered(args.dtu_eval_dir, args.scan_id)
    pts, normals = _subsample_points(gt_above, gt_normals, args.max_points, args.seed)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    print(
        f"[gt] filtered points={len(gt_above):,}  used={len(pts):,}  "
        f"obs-mask-only={len(gt_obs):,}",
        flush=True,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    pc_path = out_mesh.with_name(out_mesh.stem + "_filtered_points.ply")
    o3d.io.write_point_cloud(str(pc_path), pcd, write_ascii=False)
    print(f"[pcd] saved -> {pc_path}", flush=True)

    print(
        f"[poisson] depth={args.depth} scale={args.scale} linear_fit={args.linear_fit}",
        flush=True,
    )
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.depth,
        scale=args.scale,
        linear_fit=args.linear_fit,
    )
    mesh = _clean_mesh(mesh)
    print(
        f"[poisson] raw verts={len(mesh.vertices):,} faces={len(mesh.triangles):,}",
        flush=True,
    )

    if args.density_quantile > 0:
        density = np.asarray(density)
        keep = density > np.quantile(density, args.density_quantile)
        mesh.remove_vertices_by_mask(~keep)
        mesh = _clean_mesh(mesh)
        print(
            f"[trim] q={args.density_quantile:g} verts={len(mesh.vertices):,} "
            f"faces={len(mesh.triangles):,}",
            flush=True,
        )

    if args.crop_to_input_bbox:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            bbox.min_bound - args.bbox_margin * extent,
            bbox.max_bound + args.bbox_margin * extent,
        )
        mesh = mesh.crop(bbox)
        mesh = _clean_mesh(mesh)
        print(
            f"[crop] verts={len(mesh.vertices):,} faces={len(mesh.triangles):,}",
            flush=True,
        )

    if args.largest_component:
        mesh = _largest_component(mesh)
        mesh = _clean_mesh(mesh)
        print(
            f"[component] verts={len(mesh.vertices):,} faces={len(mesh.triangles):,}",
            flush=True,
        )

    print("[mesh] writing...", flush=True)
    o3d.io.write_triangle_mesh(str(out_mesh), mesh, write_ascii=False)
    tm = trimesh.load(str(out_mesh), force="mesh", process=False)
    watertight_method = "poisson"
    watertight_pitch = None
    if not tm.is_watertight and not args.no_force_watertight:
        print("[mesh] Poisson mesh is open; building filled voxel watertight wrapper...", flush=True)
        wrapped, used_pitch, ok = _force_watertight_with_voxels(
            tm, args.watertight_pitch if args.watertight_pitch > 0 else None
        )
        if wrapped is not None:
            wrapped.export(str(out_mesh))
            tm = trimesh.load(str(out_mesh), force="mesh", process=False)
            watertight_method = "poisson+filled_voxels"
            watertight_pitch = float(used_pitch)
            print(
                f"[mesh] voxel wrapper pitch={used_pitch:.4g} watertight={ok}",
                flush=True,
            )
    print(
        f"[mesh] saved -> {out_mesh}\n"
        f"[mesh] verts={len(tm.vertices):,} faces={len(tm.faces):,} "
        f"watertight={tm.is_watertight} euler={tm.euler_number}\n"
        f"[mesh] bounds={tm.bounds.tolist()}",
        flush=True,
    )

    meta = {
        "scan_id": args.scan_id,
        "dtu_eval_dir": str(args.dtu_eval_dir),
        "mesh": str(out_mesh),
        "filtered_point_cloud": str(pc_path),
        "n_gt_obsmask": int(len(gt_obs)),
        "n_gt_filtered": int(len(gt_above)),
        "n_points_used": int(len(pts)),
        "poisson_depth": args.depth,
        "poisson_scale": args.scale,
        "linear_fit": bool(args.linear_fit),
        "density_quantile": args.density_quantile,
        "crop_to_input_bbox": bool(args.crop_to_input_bbox),
        "largest_component": bool(args.largest_component),
        "watertight_method": watertight_method,
        "watertight_pitch": watertight_pitch,
        "watertight": bool(tm.is_watertight),
        "euler_number": int(tm.euler_number),
        "bounds": tm.bounds.tolist(),
        "protocol": "DTU GT STL vertices filtered by ObsMask + Plane, as in analysis/eval_dtu_official.py",
    }
    out_json = out_mesh.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"[meta] saved -> {out_json}", flush=True)

    out_png = out_mesh.with_suffix(".png")
    print("[viz] rendering preview...", flush=True)
    _save_preview(out_mesh, out_png, args.preview_res)
    print(f"[viz] saved -> {out_png}", flush=True)

    if args.show:
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=f"DTU scan{args.scan_id} GT Poisson")


if __name__ == "__main__":
    main()
