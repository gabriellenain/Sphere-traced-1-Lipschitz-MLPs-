#!/usr/bin/env python3
"""Make a normalized watertight-ish Stanford Dragon mesh for SDF regression."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Poisson-reconstruct and normalize Stanford Dragon.")
    ap.add_argument("--input", type=Path, default=Path("data/gt_meshes/dragon_recon/dragon_vrip_res2.ply"))
    ap.add_argument("--out", type=Path, default=Path("data/gt_meshes/dragon_watertight_norm.ply"))
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.05)
    ap.add_argument("--density-quantile", type=float, default=0.01)
    ap.add_argument("--bbox-margin", type=float, default=0.03)
    ap.add_argument("--target-size", type=float, default=1.2)
    ap.add_argument("--no-crop", action="store_true", help="skip bbox crop (preserves watertightness)")
    args = ap.parse_args()

    import open3d as o3d
    import trimesh

    print(f"input: {args.input}", flush=True)
    print(f"out:   {args.out}", flush=True)
    print(f"poisson depth={args.depth} scale={args.scale}", flush=True)

    print("loading mesh ...", flush=True)
    mesh = o3d.io.read_triangle_mesh(str(args.input))
    print(f"loaded: verts={len(mesh.vertices):,} faces={len(mesh.triangles):,}", flush=True)

    print("cleaning + normals ...", flush=True)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    print(f"cleaned: verts={len(mesh.vertices):,} faces={len(mesh.triangles):,}", flush=True)

    print("building point cloud ...", flush=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    print("running Poisson reconstruction ...", flush=True)
    rec, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.depth,
        scale=args.scale,
        linear_fit=False,
    )
    print(f"poisson raw: verts={len(rec.vertices):,} faces={len(rec.triangles):,}", flush=True)

    if args.density_quantile > 0:
        print("density trimming ...", flush=True)
        density = np.asarray(density)
        keep = density > np.quantile(density, args.density_quantile)
        rec.remove_vertices_by_mask(~keep)
        print(f"after density trim: verts={len(rec.vertices):,} faces={len(rec.triangles):,}", flush=True)
    else:
        print("skipping density trim (quantile=0)", flush=True)

    if not args.no_crop:
        print("bbox crop ...", flush=True)
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            bbox.min_bound - args.bbox_margin * extent,
            bbox.max_bound + args.bbox_margin * extent,
        )
        rec = rec.crop(bbox)
        print(f"after crop: verts={len(rec.vertices):,} faces={len(rec.triangles):,}", flush=True)

    print("normalizing coordinates ...", flush=True)
    verts = np.asarray(rec.vertices, dtype=np.float32)
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    norm_scale = args.target_size / max((verts.max(axis=0) - verts.min(axis=0)).max(), 1e-8)
    rec.vertices = o3d.utility.Vector3dVector(((verts - center) * norm_scale).astype(np.float64))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("writing mesh ...", flush=True)
    o3d.io.write_triangle_mesh(str(args.out), rec, write_ascii=False)
    tm = trimesh.load(str(args.out), force="mesh")
    print(f"saved: {args.out}", flush=True)
    print(f"final: verts={len(tm.vertices):,} faces={len(tm.faces):,} watertight={tm.is_watertight}", flush=True)
    print(f"bounds: {tm.bounds.tolist()}", flush=True)

    print("rendering preview PNG ...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = 512
    v = np.array(tm.vertices, dtype=np.float32)
    lim = max(np.abs(v).max(), 1e-3)
    xs = np.linspace(-lim, lim, res)
    ys = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(xs, ys[::-1])
    origins = np.stack([xx.ravel(), yy.ravel(), np.full(res * res, -lim * 5)], 1).astype(np.float32)
    dirs = np.zeros_like(origins); dirs[:, 2] = 1.0
    locs, idx_ray, idx_tri = tm.ray.intersects_location(origins, dirs, multiple_hits=False)
    img = np.ones((res * res, 3))
    if len(locs) > 0:
        fn = tm.face_normals[idx_tri]
        fn[fn[:, 2] > 0] *= -1
        key = np.array([-0.3, 0.5, -1.0]); key /= np.linalg.norm(key)
        fill = np.array([0.5, 0.2, -0.5]); fill /= np.linalg.norm(fill)
        shade = (0.75 * (fn @ key).clip(0, 1) + 0.25 * (fn @ fill).clip(0, 1) + 0.1).clip(0, 1)
        img[idx_ray] = shade[:, None] * np.array([0.95, 0.90, 0.82])
    png_path = args.out.with_suffix(".png")
    plt.imsave(str(png_path), img.reshape(res, res, 3).clip(0, 1))
    print(f"preview PNG -> {png_path}", flush=True)


if __name__ == "__main__":
    main()
