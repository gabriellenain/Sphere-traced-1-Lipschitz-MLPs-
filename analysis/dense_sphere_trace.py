"""Dense sphere-trace from N Fibonacci-distributed viewpoints, fuse, voxel-dedup.

Paper-grade 3D mesh from sphere tracing only — no marching cubes, no volume
rendering. For each synthetic camera on a sphere around the object we sphere-
trace at H×W resolution, stitch a screen-space mesh from the hits + analytic
∇f normals (same logic as analysis/sphere_traced_screen_mesh.py), then concatenate all
views into one PLY. Optional voxel-dedup merges coincident vertices from
overlapping views without averaging positions (snap-to-first, not average).

Output PLY is in DTU world frame, matching pred_world_mesh.ply convention.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sphere_traced_screen_mesh import (
    build_screen_mesh, grad_at, load_run, trace, write_ply,
)


def fibonacci_sphere(n: int, elev_min_deg: float, elev_max_deg: float) -> np.ndarray:
    """N points on a sphere via Fibonacci spiral, clipped to an elevation band.

    Elevation 90° is the +z pole, -90° is -z. The spiral is generated on the
    full sphere then rejection-trimmed to the band; we resample slightly above
    n to land at least n points after trimming.
    """
    z_max = np.sin(np.deg2rad(elev_max_deg))
    z_min = np.sin(np.deg2rad(elev_min_deg))
    n_gen = int(np.ceil(n * 2.2 / max(1e-3, z_max - z_min)))  # over-generate
    i = np.arange(n_gen) + 0.5
    z = 1.0 - 2.0 * i / n_gen
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i      # golden-angle spiral
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    pts = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    keep = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[keep]
    return pts[:n] if len(pts) >= n else pts


def lookat_c2w(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """OpenCV-convention c2w: camera +x = right, +y = down, +z = forward."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward).clip(min=1e-9)
    if abs(np.dot(forward, up_hint)) > 0.99:
        up_hint = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.5 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right).clip(min=1e-9)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=1)        # columns = basis
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w


def make_intrinsics(H: int, W: int, fov_deg: float) -> np.ndarray:
    f = (H / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    K = np.array([[f, 0.0, W / 2.0],
                  [0.0, f, H / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def rays_from_KC(K: np.ndarray, c2w: np.ndarray, H: int, W: int, device: str):
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d_cam = np.stack([
        (xs + 0.5 - K[0, 2]) / K[0, 0],
        (ys + 0.5 - K[1, 2]) / K[1, 1],
        np.ones_like(xs, dtype=np.float64),
    ], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).copy()
    return (
        torch.from_numpy(origins.astype(np.float32)).to(device),
        torch.from_numpy(dirs.reshape(-1, 3).astype(np.float32)).to(device),
    )


def voxel_dedup(verts: np.ndarray, normals: np.ndarray, faces: np.ndarray,
                voxel_size: float):
    """Snap coincident vertices (within `voxel_size`) to a single representative.

    Keeps the first vertex per voxel — no position averaging — so the geometry
    stays raw-sphere-traced. Re-indexes faces and drops degenerate triangles
    that collapse to an edge or point after merging.
    """
    qi = np.floor(verts / voxel_size).astype(np.int64)
    # 64-bit hash of (x,y,z) voxel index; primes chosen to span 64 bits.
    key = (qi[:, 0].astype(np.int64) * np.int64(73856093)) \
        ^ (qi[:, 1].astype(np.int64) * np.int64(19349663)) \
        ^ (qi[:, 2].astype(np.int64) * np.int64(83492791))
    _, first_idx, inverse = np.unique(key, return_index=True, return_inverse=True)
    new_verts = verts[first_idx]
    new_norms = normals[first_idx]
    new_faces = inverse[faces]
    non_deg = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[non_deg]
    # dedup faces by sorted vertex triple: different views triangulate the
    # same voxel patch slightly differently, leaving stacks of near-coincident
    # faces that wreck BVH performance. After voxel-merge, identical triples
    # collapse to a single face.
    sorted_faces = np.sort(new_faces, axis=1)
    _, uniq_face_idx = np.unique(sorted_faces, axis=0, return_index=True)
    return new_verts, new_norms, new_faces[uniq_face_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--n-views", type=int, default=200,
                    help="number of Fibonacci-sphere cameras")
    ap.add_argument("--cam-radius", type=float, default=2.7,
                    help="camera distance in normalized frame (DTU uses ~2.7)")
    ap.add_argument("--fov-deg", type=float, default=50.0)
    ap.add_argument("--res", type=int, default=512, help="square render resolution")
    ap.add_argument("--elev-min-deg", type=float, default=-30.0)
    ap.add_argument("--elev-max-deg", type=float, default=85.0)
    ap.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help="world up hint (normalized frame)")
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--rel-depth-gap", type=float, default=0.02)
    ap.add_argument("--grazing-cos", type=float, default=0.1,
                    help="per-quad rejection: drop quads with all corners more grazing")
    ap.add_argument("--voxel-size", type=float, default=0.0,
                    help="if >0, voxel-dedup merged verts at this size (normalized frame units)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    f, cfg, scene = load_run(args.run_dir, args.ckpt, args.device)
    cfg_trace = replace(cfg, iters=args.max_iters)

    cam_path = scene / "cameras_sphere.npz"
    if not cam_path.exists():
        cam_path = scene / "cameras.npz"
    S = np.load(cam_path)["scale_mat_0"].astype(np.float64)

    H = W = args.res
    K = make_intrinsics(H, W, args.fov_deg)
    target = np.zeros(3)
    up_hint = np.asarray(args.up, dtype=np.float64)
    cam_dirs = fibonacci_sphere(args.n_views, args.elev_min_deg, args.elev_max_deg)
    print(f"generating {len(cam_dirs)} cameras at radius {args.cam_radius} "
          f"({args.elev_min_deg:.0f}°…{args.elev_max_deg:.0f}° elevation)", flush=True)

    all_v, all_n, all_f = [], [], []
    voff = 0
    for vi, d in enumerate(cam_dirs):
        eye = d * args.cam_radius
        c2w = lookat_c2w(eye, target, up_hint)
        origins, dirs = rays_from_KC(K, c2w, H, W, args.device)
        print(f"\n=== cam {vi+1}/{len(cam_dirs)} ===  eye={eye.round(2).tolist()}", flush=True)
        t, hit = trace(f, origins, dirs, cfg_trace, args.chunk)
        n_hits = int(hit.sum())
        print(f"  hits: {n_hits:,}/{len(hit):,} ({100*n_hits/len(hit):.1f}%)", flush=True)
        if n_hits == 0:
            continue
        pts = origins + torch.from_numpy(t).to(origins.device).unsqueeze(-1) * dirs
        normals_raw = grad_at(f, pts, args.chunk)
        n_norm = np.linalg.norm(normals_raw, axis=1, keepdims=True).clip(min=1e-9)
        normals = (normals_raw / n_norm).astype(np.float32)

        verts_n, vnorm, faces = build_screen_mesh(
            hit, pts.detach().cpu().numpy().astype(np.float32),
            t.astype(np.float32), normals, H, W,
            args.rel_depth_gap, args.grazing_cos,
            dirs.detach().cpu().numpy().astype(np.float32),
        )
        if len(faces) == 0:
            continue
        all_v.append(verts_n)
        all_n.append(vnorm)
        all_f.append(faces + voff)
        voff += len(verts_n)

    if not all_f:
        raise SystemExit("no faces produced — check cam-radius / model")
    verts = np.concatenate(all_v).astype(np.float32)
    normals = np.concatenate(all_n).astype(np.float32)
    faces = np.concatenate(all_f).astype(np.int64)
    print(f"\nfused (pre-dedup): {len(verts):,} verts  {len(faces):,} faces", flush=True)

    if args.voxel_size > 0:
        verts, normals, faces = voxel_dedup(verts, normals, faces, args.voxel_size)
        print(f"voxel-deduped:    {len(verts):,} verts  {len(faces):,} faces "
              f"(voxel={args.voxel_size:g})", flush=True)

    # normalized → DTU world frame
    v_h = np.concatenate([verts.astype(np.float64), np.ones((len(verts), 1))], axis=1)
    verts_world = (v_h @ S.T)[:, :3].astype(np.float32)
    norms_world = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-9)

    write_ply(args.out, verts_world, norms_world.astype(np.float32), faces.astype(np.int64))


if __name__ == "__main__":
    main()
