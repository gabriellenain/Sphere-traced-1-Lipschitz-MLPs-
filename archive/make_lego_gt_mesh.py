#!/usr/bin/env python3
"""Build a lego reference surface mesh from NeRF-synthetic depth maps.

The lego download used by this repo does not ship a triangle mesh. It does,
however, include foreground-masked Blender Z-pass depth PNGs for the test
split. This script unprojects those depth maps and triangulates each valid
depth image in world coordinates, producing a surface mesh that can be passed
to archive/compare_geometry.py as --a-mesh/--b-mesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lip_tracer.config import BLENDER_SCENE


def _depth_path(scene: Path, frame_path: str) -> Path | None:
    for suffix in ("_depth_0001.png", "_depth_0029.png"):
        path = scene / f"{frame_path}{suffix}"
        if path.exists():
            return path
    return None


def _frame_mesh(scene: Path, frame: dict, fov_x: float, stride: int,
                far: float, max_edge: float) -> tuple[np.ndarray, np.ndarray] | None:
    import imageio.v2 as imageio

    path = _depth_path(scene, frame["file_path"])
    if path is None:
        return None

    rgba = imageio.imread(str(path)).astype(np.float32) / 255.0
    h, w = rgba.shape[:2]
    depth = rgba[..., 0] * far
    valid = (rgba[..., 3] > 0.5) & (depth > 0.0)

    ys, xs = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing="ij")
    depth_s = depth[::stride, ::stride]
    valid_s = valid[::stride, ::stride]
    hs, ws = depth_s.shape

    fx = 0.5 * w / np.tan(0.5 * fov_x)
    cx, cy = w / 2.0, h / 2.0
    pts_cam = np.stack([
        (xs - cx) / fx * depth_s,
        (ys - cy) / fx * depth_s,
        depth_s,
    ], axis=-1).astype(np.float32)

    c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
    c2w = c2w @ np.diag([1, -1, -1, 1]).astype(np.float32)
    verts_grid = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]

    index = -np.ones((hs, ws), dtype=np.int64)
    index[valid_s] = np.arange(int(valid_s.sum()))
    verts = verts_grid[valid_s].astype(np.float32)

    if len(verts) == 0:
        return None

    faces: list[tuple[int, int, int]] = []
    for y in range(hs - 1):
        for x in range(ws - 1):
            ids = (index[y, x], index[y, x + 1], index[y + 1, x], index[y + 1, x + 1])
            if min(ids) < 0:
                continue
            p00 = verts_grid[y, x]
            p10 = verts_grid[y, x + 1]
            p01 = verts_grid[y + 1, x]
            p11 = verts_grid[y + 1, x + 1]
            edge_ok = (
                np.linalg.norm(p00 - p10) <= max_edge
                and np.linalg.norm(p00 - p01) <= max_edge
                and np.linalg.norm(p11 - p10) <= max_edge
                and np.linalg.norm(p11 - p01) <= max_edge
            )
            if not edge_ok:
                continue
            faces.append((int(ids[0]), int(ids[2]), int(ids[1])))
            faces.append((int(ids[1]), int(ids[2]), int(ids[3])))

    if not faces:
        return None
    return verts, np.asarray(faces, dtype=np.int64)


def build_mesh(scene: Path, split: str, stride: int, far: float,
               max_edge: float, max_views: int | None):
    import trimesh

    meta = json.loads((scene / f"transforms_{split}.json").read_text())
    frames = meta["frames"][:max_views]
    meshes = []
    used = 0
    for frame in frames:
        out = _frame_mesh(scene, frame, meta["camera_angle_x"], stride, far, max_edge)
        if out is None:
            continue
        verts, faces = out
        meshes.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
        used += 1
        print(f"  {frame['file_path']}: {len(verts):,} verts  {len(faces):,} faces", flush=True)

    if not meshes:
        raise RuntimeError(f"No usable depth maps found under {scene} for split={split!r}")

    mesh = trimesh.util.concatenate(meshes)
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    mesh.fix_normals()
    print(f"used {used} depth views", flush=True)
    return mesh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=BLENDER_SCENE)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out", type=Path, default=Path("data/gt_meshes/lego_depth_gt_mesh.ply"))
    ap.add_argument("--stride", type=int, default=4,
                    help="depth-pixel stride; lower is denser but heavier")
    ap.add_argument("--far", type=float, default=6.0,
                    help="Blender Z-pass far clip used to decode 8-bit depth")
    ap.add_argument("--max-edge", type=float, default=0.08,
                    help="discard image-grid triangles with long world-space edges")
    ap.add_argument("--max-views", type=int, default=None)
    args = ap.parse_args()

    mesh = build_mesh(args.scene, args.split, args.stride, args.far,
                      args.max_edge, args.max_views)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)
    print(f"saved lego GT mesh -> {args.out}", flush=True)
    print(f"  verts={len(mesh.vertices):,} faces={len(mesh.faces):,}", flush=True)
    print(f"  bounds={mesh.bounds.tolist()}", flush=True)


if __name__ == "__main__":
    main()
