#!/usr/bin/env python3
"""Diagnose MVSFormer++ depth geometry on DTU scan24 before neural training.

Default mode is intentionally gated: it back-projects MVSFormer++ depths, saves a
merged point cloud and visual-hull comparison screenshots, then stops.  Run with
--fuse only after visually confirming the point cloud recovers the scan24 cavity
better than the visual hull.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lip_tracer.data import load_pair_file
from lip_tracer.geomvs import _read_pfm, load_mvsformer_depths_idr
from lip_tracer.visual_hull import carve


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("data/dtu_idr/scan24"))
    ap.add_argument("--depth-dir", type=Path,
                    default=Path("_diagnostics/mvsformer_scan24_idr/depths_1152x864"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("_diagnostics/mvsformer_scan24_idr/diagnostic"))
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--hull-res", type=int, default=256)
    ap.add_argument("--max-points-per-view", type=int, default=80_000)
    ap.add_argument("--screenshot-points", type=int, default=300_000)
    ap.add_argument("--fuse", action="store_true",
                    help="run consistency filtering + Open3D TSDF + 256^3 SDF")
    ap.add_argument("--consistency-src", type=int, default=4)
    ap.add_argument("--min-consistent", type=int, default=2)
    ap.add_argument("--rel-depth-tol", type=float, default=0.01)
    ap.add_argument("--abs-depth-tol", type=float, default=0.015)
    ap.add_argument("--grid-res", type=int, default=256)
    return ap.parse_args()


def find_scan_root(depth_dir: Path) -> Path:
    scan_dirs = [d for d in depth_dir.iterdir()
                 if d.is_dir() and (d / "depth_est").exists()]
    return scan_dirs[0] if scan_dirs else depth_dir


def save_depth_npy(scan_root: Path, out_dir: Path) -> None:
    out = out_dir / "depth_maps_npy"
    out.mkdir(parents=True, exist_ok=True)
    for pfm in sorted((scan_root / "depth_est").glob("*.pfm")):
        np.save(out / f"{pfm.stem}.npy", _read_pfm(pfm).astype(np.float32))


def backproject_depth(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray,
                      valid: np.ndarray) -> np.ndarray:
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float64)
    pts_cam = np.stack([
        (xs.astype(np.float64) - K[0, 2]) / K[0, 0] * z,
        (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * z,
        z,
    ], axis=-1)
    pts = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return pts.astype(np.float32)


def write_point_cloud_ply(points: np.ndarray, colors: np.ndarray | None,
                          path: Path) -> None:
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if colors is not None and len(colors) == len(points):
        pc.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pc)


def occ_to_mesh_world(occ: np.ndarray, bound: float):
    from skimage.measure import marching_cubes
    res = occ.shape[0]
    voxel = 2 * bound / max(res - 1, 1)
    verts, faces, *_ = marching_cubes(occ.astype(np.float32), level=0.5,
                                      spacing=(voxel,) * 3)
    verts_world = verts[:, [2, 1, 0]] - bound
    return verts_world.astype(np.float32), faces.astype(np.int64)


def save_mesh_ply(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    import trimesh
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def sample_mesh_points(mesh_path: Path, n: int, seed: int = 0) -> np.ndarray:
    import trimesh
    mesh = trimesh.load_mesh(mesh_path, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return pts.astype(np.float32)


def render_points_grid(point_sets: list[tuple[str, np.ndarray]], out: Path,
                       bound: float, max_points: int, seed: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    views = [(-65, 18), (25, 18), (115, 25), (0, 82)]
    fig, axes = plt.subplots(len(point_sets), len(views),
                             figsize=(3.2 * len(views), 3.2 * len(point_sets)),
                             squeeze=False)
    for r, (title, pts_full) in enumerate(point_sets):
        pts = pts_full
        if len(pts) > max_points:
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
        pts = np.asarray(pts, dtype=np.float32)
        for c, (az, el) in enumerate(views):
            ax = axes[r, c]
            ax.set_facecolor("white")
            ax.set_aspect("equal")
            ax.axis("off")
            if len(pts) == 0:
                ax.set_title(title if c == 0 else "")
                continue
            azr, elr = np.deg2rad(az), np.deg2rad(el)
            view = np.array([np.cos(elr) * np.cos(azr),
                             np.cos(elr) * np.sin(azr),
                             np.sin(elr)], dtype=np.float32)
            up0 = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(up0, view)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1, 0, 0], dtype=np.float32)
            right /= np.linalg.norm(right)
            up = np.cross(view, right)
            x = pts @ right
            y = pts @ up
            z = pts @ view
            order = np.argsort(z)
            shade = (z[order] - z.min()) / max(float(z.max() - z.min()), 1e-6)
            ax.scatter(x[order], y[order], c=shade, cmap="viridis",
                       s=0.15, linewidths=0, rasterized=True)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            if c == 0:
                ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.05)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_point_cloud(mvs: dict, views: dict, out_dir: Path,
                     max_points_per_view: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    pts_all, col_all = [], []
    images = views["images"].numpy()
    for i, (depth_t, valid_t) in enumerate(zip(mvs["depths"], mvs["valid"])):
        depth = depth_t.numpy()
        valid = valid_t.numpy().astype(bool)
        ys, xs = np.where(valid)
        if len(xs) > max_points_per_view:
            pick = rng.choice(len(xs), max_points_per_view, replace=False)
            sub = np.zeros_like(valid)
            sub[ys[pick], xs[pick]] = True
            valid = sub
            ys, xs = ys[pick], xs[pick]
        pts = backproject_depth(depth, mvs["K"][i], mvs["c2w"][i], valid)
        if len(pts) == 0:
            continue
        img = images[i]
        H, W = depth.shape
        if img.shape[:2] != (H, W):
            pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            img = np.asarray(pil.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
        col_all.append(img[ys, xs].astype(np.float32))
        pts_all.append(pts)

    points = np.concatenate(pts_all, axis=0) if pts_all else np.empty((0, 3), np.float32)
    colors = np.concatenate(col_all, axis=0) if col_all else np.empty((0, 3), np.float32)
    write_point_cloud_ply(points, colors, out_dir / "mvsformer_point_cloud.ply")
    return points, colors


def consistency_filter(mvs: dict, scene: Path, n_src: int, min_consistent: int,
                       rel_tol: float, abs_tol: float) -> list[np.ndarray]:
    pairs = load_pair_file(scene / "pair.txt")
    depths = [d.numpy().astype(np.float32) for d in mvs["depths"]]
    valid = [v.numpy().astype(bool).copy() for v in mvs["valid"]]
    H, W = depths[0].shape
    filtered: list[np.ndarray] = []
    for ref, depth in enumerate(depths):
        ys, xs = np.where(valid[ref])
        keep = np.zeros(len(xs), dtype=bool)
        if len(xs) == 0:
            filtered.append(np.zeros_like(valid[ref]))
            continue
        z = depth[ys, xs].astype(np.float64)
        pts_cam = np.stack([
            (xs.astype(np.float64) - mvs["K"][ref, 0, 2]) / mvs["K"][ref, 0, 0] * z,
            (ys.astype(np.float64) - mvs["K"][ref, 1, 2]) / mvs["K"][ref, 1, 1] * z,
            z,
        ], axis=-1)
        pts = pts_cam @ mvs["c2w"][ref, :3, :3].T + mvs["c2w"][ref, :3, 3]
        votes = np.zeros(len(pts), dtype=np.int16)
        srcs = [s for s in pairs.get(ref, []) if s != ref][:n_src]
        for src in srcs:
            w2c = np.linalg.inv(mvs["c2w"][src])
            cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
            front = cam[:, 2] > 1e-5
            u = cam[:, 0] / np.maximum(cam[:, 2], 1e-6) * mvs["K"][src, 0, 0] + mvs["K"][src, 0, 2]
            v = cam[:, 1] / np.maximum(cam[:, 2], 1e-6) * mvs["K"][src, 1, 1] + mvs["K"][src, 1, 2]
            ui = np.rint(u).astype(np.int32)
            vi = np.rint(v).astype(np.int32)
            inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            ok = np.zeros(len(pts), dtype=bool)
            idx = np.where(inb)[0]
            if len(idx):
                ds = depths[src][vi[idx], ui[idx]]
                vs = valid[src][vi[idx], ui[idx]]
                tol = np.maximum(abs_tol, rel_tol * np.maximum(cam[idx, 2], 1e-6))
                ok[idx] = vs & (np.abs(ds - cam[idx, 2]) <= tol)
            votes += ok.astype(np.int16)
        keep = votes >= min_consistent
        filt = np.zeros_like(valid[ref])
        filt[ys[keep], xs[keep]] = True
        filtered.append(filt)
        print(f"  [filter] view {ref:02d}: {int(filt.sum())} / {len(xs)} kept")
    return filtered


def fuse_tsdf(mvs: dict, filtered: list[np.ndarray], out_dir: Path,
              voxel_size: float, sdf_trunc: float) -> Path:
    import open3d as o3d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    for i, (depth_t, valid) in enumerate(zip(mvs["depths"], filtered)):
        depth = depth_t.numpy().astype(np.float32).copy()
        depth[~valid] = 0.0
        color = np.zeros((*depth.shape, 3), dtype=np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=6.0,
            convert_rgb_to_intensity=False,
        )
        K = mvs["K"][i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth.shape[1], depth.shape[0],
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        volume.integrate(rgbd, intrinsic, np.linalg.inv(mvs["c2w"][i]))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    import trimesh
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                         faces=np.asarray(mesh.triangles), process=False)
    parts = tm.split(only_watertight=False)
    if parts:
        tm = max(parts, key=lambda m: m.area)
    trimesh.repair.fill_holes(tm)
    raw_path = out_dir / "mesh_raw_tsdf.ply"
    tm.export(raw_path)

    mesh_path = out_dir / "mesh.ply"
    if tm.is_watertight:
        tm.export(mesh_path)
        return mesh_path

    wrapped = None
    pitch = voxel_size
    for _ in range(6):
        vox = tm.voxelized(pitch).fill()
        cand = vox.marching_cubes
        cand.apply_transform(vox.transform)
        cand.remove_unreferenced_vertices()
        wrapped = cand
        if cand.is_watertight:
            break
        pitch *= 1.25
    if wrapped is None:
        raise RuntimeError("failed to build watertight TSDF wrapper")
    wrapped.export(mesh_path)
    print(f"  [repair] raw TSDF watertight={tm.is_watertight}; "
          f"saved watertight voxel wrapper pitch={pitch:.6f} as {mesh_path.name}")
    return mesh_path


def signed_distance_grid(mesh_path: Path, out_path: Path, bound: float, res: int) -> None:
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    sdf = np.empty((res, res, res), dtype=np.float32)  # axes z,y,x
    print(f"  [sdf] full-space signed distance to watertight mesh: {mesh_path}")
    for iz, z in enumerate(lin):
        yy, xx = np.meshgrid(lin, lin, indexing="ij")
        pts = np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3)
        d = scene.compute_signed_distance(o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32))
        sdf[iz] = d.numpy().reshape(res, res)
        if iz % 32 == 0:
            print(f"  [sdf] slice {iz}/{res}")
    np.save(out_path, sdf)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scan_root = find_scan_root(args.depth_dir)
    pfms = sorted((scan_root / "depth_est").glob("*.pfm"))
    if not pfms:
        raise FileNotFoundError(
            f"No MVSFormer++ depth maps found under {scan_root / 'depth_est'}.\n"
            "Run precompute_mvsformer_depths_scan24.slurm first.")

    mvs = load_mvsformer_depths_idr(args.scene, args.depth_dir,
                                    conf_thresh=args.conf_thr,
                                    use_idr_mask=True)
    if mvs is None:
        raise RuntimeError("MVSFormer++ depth loading failed")
    save_depth_npy(scan_root, args.out_dir)

    from lip_tracer.data import load_views
    views = load_views(args.scene, down=1)
    points, colors = make_point_cloud(mvs, views, args.out_dir, args.max_points_per_view)
    print(f"  [pc] merged {len(points)} points")

    print(f"  [hull] carving visual hull res={args.hull_res}, bound={args.bound}")
    occ = carve(scene=args.scene, res=args.hull_res, bound=args.bound, border_aware=True)
    hv, hf = occ_to_mesh_world(occ, args.bound)
    hull_mesh = args.out_dir / "visual_hull_mesh.ply"
    save_mesh_ply(hv, hf, hull_mesh)

    hull_pts = sample_mesh_points(hull_mesh, args.screenshot_points)
    render_points_grid([("MVSFormer++ point cloud", points)],
                       args.out_dir / "mvsformer_point_cloud.png",
                       args.bound, args.screenshot_points)
    render_points_grid([("visual-hull mesh", hull_pts)],
                       args.out_dir / "visual_hull_mesh.png",
                       args.bound, args.screenshot_points)
    render_points_grid([("MVSFormer++ point cloud", points),
                        ("visual-hull mesh", hull_pts)],
                       args.out_dir / "mvsformer_vs_visual_hull.png",
                       args.bound, args.screenshot_points)

    summary = {
        "scene": str(args.scene),
        "depth_root": str(scan_root),
        "conf_thr": args.conf_thr,
        "n_views": len(mvs["depths"]),
        "point_count": int(len(points)),
        "visual_hull_mesh": str(hull_mesh),
        "gate": "Inspect mvsformer_vs_visual_hull.png for scan24 cavity recovery; rerun with --fuse only if better.",
    }

    if args.fuse:
        voxel_size = 3.0 / 256.0
        sdf_trunc = 5.0 * voxel_size
        filtered = consistency_filter(mvs, args.scene, args.consistency_src,
                                      args.min_consistent, args.rel_depth_tol,
                                      args.abs_depth_tol)
        mesh_path = fuse_tsdf(mvs, filtered, args.out_dir, voxel_size, sdf_trunc)
        tsdf_pts = sample_mesh_points(mesh_path, args.screenshot_points)
        render_points_grid([("TSDF mesh", tsdf_pts)],
                           args.out_dir / "tsdf_mesh.png",
                           args.bound, args.screenshot_points)
        signed_distance_grid(mesh_path, args.out_dir / "sdf_grid.npy",
                             args.bound, args.grid_res)
        summary.update({
            "tsdf_mesh": str(mesh_path),
            "sdf_grid": str(args.out_dir / "sdf_grid.npy"),
            "voxel_size": voxel_size,
            "sdf_trunc": sdf_trunc,
            "consistency_src": args.consistency_src,
            "min_consistent": args.min_consistent,
        })
    else:
        print("\n[gated] Stopping before TSDF. Inspect the point cloud vs visual hull.")
        print("        If the scan24 cavity/concavity is recovered, rerun with --fuse.")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
