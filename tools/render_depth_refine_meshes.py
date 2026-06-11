"""Render per-view meshes produced by analysis/refine_depths.py.

Each viewXX_downY_before/after.ply is a single-view depth surface. This script
projects those meshes back into their source camera and writes quick PNG contact
sheets so the refinement can be inspected without opening PLYs by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from lip_tracer.data import load_views
from lip_tracer.train import load_config_json


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"not a PLY file: {path}")
        n_verts = n_faces = None
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break
        if n_verts is None or n_faces is None:
            raise ValueError(f"missing vertex/face count in {path}")
        verts = np.array([[float(x) for x in f.readline().split()] for _ in range(n_verts)],
                         dtype=np.float32)
        faces = []
        for _ in range(n_faces):
            parts = f.readline().split()
            if parts:
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return verts, np.asarray(faces, dtype=np.int32)


def project_vertices(verts: np.ndarray, K: np.ndarray, w2c: np.ndarray, down: int):
    vh = np.concatenate([verts.astype(np.float64), np.ones((len(verts), 1))], axis=1)
    cam = (w2c @ vh.T).T[:, :3]
    uvh = (K @ cam.T).T
    uv = uvh[:, :2] / np.clip(uvh[:, 2:3], 1e-8, None)
    uv = uv / float(down)
    return uv, cam[:, 2], cam


def rasterize_mesh(verts: np.ndarray, faces: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                   c2w: np.ndarray, H: int, W: int, down: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("render_depth_refine_meshes.py needs opencv-python/cv2") from exc

    H_d, W_d = H // down, W // down
    img = np.ones((H_d, W_d, 3), dtype=np.float32)
    depth = np.full((H_d, W_d), np.inf, dtype=np.float32)

    uv, z, cam = project_vertices(verts, K, w2c, down)
    face_verts = verts[faces]
    normals = np.cross(face_verts[:, 1] - face_verts[:, 0],
                       face_verts[:, 2] - face_verts[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    view_dir = c2w[:3, 2].astype(np.float32)
    shade = 0.35 + 0.65 * np.abs(normals @ view_dir).clip(0, 1)

    face_depth = z[faces].mean(axis=1)
    order = np.argsort(face_depth)[::-1]  # far to near; z-buffer still resolves overlaps
    for fi in order:
        tri = faces[fi]
        if (z[tri] <= 0).any():
            continue
        pts = uv[tri]
        if ((pts[:, 0] < -2) | (pts[:, 0] >= W_d + 2) |
                (pts[:, 1] < -2) | (pts[:, 1] >= H_d + 2)).all():
            continue
        poly = np.round(pts).astype(np.int32)
        x0 = max(int(poly[:, 0].min()), 0)
        x1 = min(int(poly[:, 0].max()) + 1, W_d - 1)
        y0 = max(int(poly[:, 1].min()), 0)
        y1 = min(int(poly[:, 1].max()) + 1, H_d - 1)
        if x1 < x0 or y1 < y0:
            continue

        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly - np.array([x0, y0], dtype=np.int32), 1)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        px = xs + x0
        py = ys + y0

        p0, p1, p2 = pts.astype(np.float64)
        denom = ((p1[1] - p2[1]) * (p0[0] - p2[0]) +
                 (p2[0] - p1[0]) * (p0[1] - p2[1]))
        if abs(denom) < 1e-8:
            continue
        qx = px + 0.5
        qy = py + 0.5
        w0 = ((p1[1] - p2[1]) * (qx - p2[0]) +
              (p2[0] - p1[0]) * (qy - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (qx - p2[0]) +
              (p0[0] - p2[0]) * (qy - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        z_pix = (w0 * z[tri[0]] + w1 * z[tri[1]] + w2 * z[tri[2]]).astype(np.float32)
        keep = z_pix < depth[py, px]
        if not keep.any():
            continue
        depth[py[keep], px[keep]] = z_pix[keep]
        img[py[keep], px[keep]] = shade[fi]

    depth_vis = np.zeros((H_d, W_d, 3), dtype=np.float32)
    hit = np.isfinite(depth)
    if hit.any():
        lo, hi = np.percentile(depth[hit], [1, 99])
        d = (1.0 - (depth - lo) / max(hi - lo, 1e-6)).clip(0, 1)
        depth_vis[hit] = d[hit, None]
        depth_vis[~hit] = 1.0
    return img, depth_vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("outputs/run_20260510_094943_scan65"))
    ap.add_argument("--depth-dir", type=Path, required=True)
    ap.add_argument("--views", type=int, nargs="+", default=None)
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or args.depth_dir / "mesh_renders"
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config_json(args.run / "config.json")
    views = load_views(cfg.scene, down=1)
    H, W = views["H"], views["W"]
    K_all = views["K"].numpy()
    c2w_all = views["c2w"].numpy()
    w2c_all = np.linalg.inv(c2w_all)
    images = views.get("images")

    if args.views is None:
        view_ids = sorted({
            int(p.name[4:6]) for p in args.depth_dir.glob("view*_down*_before.ply")
        })
    else:
        view_ids = args.views

    rows = []
    for vi in view_ids:
        tag = f"view{vi:02d}_down{args.down}"
        before_p = args.depth_dir / f"{tag}_before.ply"
        after_p = args.depth_dir / f"{tag}_after.ply"
        if not before_p.exists() or not after_p.exists():
            print(f"[skip] missing before/after PLY for view {vi:02d}")
            continue
        print(f"[render] view {vi:02d}", flush=True)
        vb, fb = read_ascii_ply(before_p)
        va, fa = read_ascii_ply(after_p)
        before, before_depth = rasterize_mesh(vb, fb, K_all[vi], w2c_all[vi], c2w_all[vi],
                                              H, W, args.down)
        after, after_depth = rasterize_mesh(va, fa, K_all[vi], w2c_all[vi], c2w_all[vi],
                                            H, W, args.down)
        if images is not None:
            gt = images[vi].numpy()[::args.down, ::args.down][:before.shape[0], :before.shape[1]]
        else:
            gt = np.ones_like(before)
        row = np.concatenate([gt, before, after, before_depth, after_depth], axis=1)
        imageio.imwrite(out / f"{tag}_mesh_compare.png",
                        np.clip(row * 255 + 0.5, 0, 255).astype(np.uint8))
        rows.append(row)

    if rows:
        grid = np.concatenate(rows, axis=0)
        imageio.imwrite(out / "grid_mesh_compare.png",
                        np.clip(grid * 255 + 0.5, 0, 255).astype(np.uint8))
        print(f"wrote {out / 'grid_mesh_compare.png'}")


if __name__ == "__main__":
    main()
