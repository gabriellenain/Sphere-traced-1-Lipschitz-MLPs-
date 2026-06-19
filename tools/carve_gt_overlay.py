#!/usr/bin/env python3
"""Minimal carve-vs-GT overlay: is anything missing from a carved init?

Ray-casts a carved mesh (or SDF occupancy grid) from a handful of training views
and overlays the result on the ground-truth images, so missing geometry is
directly visible. Three rows per view-set:
  1. Phong of the carved mesh (white background)
  2. GT × hit   (GT shown only where the carve is hit; white = carve miss)
  3. hit-map    (green = hit on GT foreground, red = GT foreground the carve
                 misses, white = background)

Scene-agnostic; reuses lip_tracer.data.load_views and the init-viz Phong
constants. Foreground for the hit-map comes from the scene masks (override the
mask directory with --mask-dir, e.g. a gsam object mask).

Usage:
  python tools/carve_gt_overlay.py \
    --scene data/tnt/Caterpillar \
    --mesh  _diagnostics/.../sphere_mvsformer_carved.ply \
    --out   _diagnostics/.../carve_gt_overlay.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from lip_tracer.data import load_views


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mesh", type=Path, help="carved mesh .ply")
    g.add_argument("--sdf", type=Path, help="carved SDF occupancy .npy")
    ap.add_argument("--bound", type=float, default=1.5,
                    help="grid bound (only used with --sdf)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--views", type=int, nargs="*", default=None,
                    help="explicit view indices (overrides the spread of --n-views)")
    ap.add_argument("--res", type=int, default=500, help="render height (px)")
    ap.add_argument("--mask-dir", type=Path, default=None,
                    help="override foreground-mask dir (PNGs named <pose-stem>.png)")
    return ap.parse_args()


def load_mesh(args) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    if args.mesh is not None:
        m = trimesh.load(str(args.mesh), process=False)
        return np.asarray(m.vertices), np.asarray(m.faces)
    from skimage.measure import marching_cubes
    occ = np.load(args.sdf)
    verts, faces, _, _ = marching_cubes(occ, level=0.0)
    res = occ.shape[0]
    verts = verts / (res - 1) * (2 * args.bound) - args.bound
    return verts, faces


def load_mask_override(mask_dir: Path, scene: Path, n: int, H: int, W: int) -> np.ndarray:
    """Load masks aligned to the sorted pose order (TnT 0_*.txt convention)."""
    from PIL import Image
    pose_paths = sorted(p for p in (scene / "pose").glob("0_*.txt"))
    masks = []
    for pp in pose_paths:
        mp = mask_dir / (pp.stem + ".png")
        if mp.exists():
            m = np.array(Image.open(mp).convert("L").resize((W, H), Image.NEAREST)) > 127
        else:
            m = np.ones((H, W), dtype=bool)
        masks.append(m)
    return np.stack(masks)


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    views = load_views(args.scene, down=1)
    V = views["c2w"].shape[0]
    H_full, W_full = views["H"], views["W"]
    down = max(1, H_full // args.res)
    H, W = H_full // down, W_full // down

    if args.views:
        ids = [i % V for i in args.views]
    else:
        ids = [int(round(i * (V - 1) / max(args.n_views - 1, 1)))
               for i in range(args.n_views)]

    images = views["images"].numpy()
    if args.mask_dir is not None:
        masks_full = load_mask_override(args.mask_dir, args.scene, V, H_full, W_full)
    else:
        masks_full = views["masks"].numpy().astype(bool)

    verts, faces = load_mesh(args)
    print(f"mesh: {len(verts)} verts / {len(faces)} faces  | {len(ids)} views @ {H}x{W}")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    fnorm = mesh.face_normals
    light = np.array([0.577, 0.577, 0.577])
    base = np.array([0.72, 0.72, 0.85])

    phong, gtxhit, hitmap = [], [], []
    miss_fracs = []
    for vi in ids:
        K = views["K"][vi].numpy(); c2w = views["c2w"][vi].numpy()
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xs_f = (xs + 0.5) * down - 0.5; ys_f = (ys + 0.5) * down - 0.5
        d_cam = np.stack([(xs_f - K[0, 2]) / K[0, 0],
                          (ys_f - K[1, 2]) / K[1, 1],
                          np.ones_like(xs_f)], axis=-1)
        d_w = d_cam @ c2w[:3, :3].T
        d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
        d_w = d_w.reshape(-1, 3)
        o_w = np.broadcast_to(c2w[:3, 3], d_w.shape)
        tri = inter.intersects_first(ray_origins=o_w, ray_directions=d_w)
        hit = (tri >= 0)
        n = np.zeros_like(d_w)
        n[hit] = fnorm[tri[hit]]
        flip = (n * d_w).sum(-1) > 0
        n[flip] *= -1.0
        diffuse = np.clip((n * light).sum(-1, keepdims=True), 0, 1)
        shaded = (0.35 + 0.65 * diffuse) * base
        hit2d = hit.reshape(H, W)
        phong.append(np.where(hit2d[..., None], shaded.reshape(H, W, 3), 1.0))

        from PIL import Image as _PIL
        gt = images[vi]
        if down > 1:
            gt = np.array(_PIL.fromarray((gt * 255).astype(np.uint8)).resize(
                (W, H), _PIL.BILINEAR)).astype(np.float32) / 255.0
        gtxhit.append(np.where(hit2d[..., None], gt, 1.0))

        fg = masks_full[vi]
        if fg.shape != (H, W):
            fg = np.array(_PIL.fromarray(fg.astype(np.uint8) * 255).resize(
                (W, H), _PIL.NEAREST)) > 127
        hm = np.ones((H, W, 3), dtype=np.float32)
        hm[fg & hit2d] = [0.4, 0.85, 0.4]
        hm[fg & ~hit2d] = [0.9, 0.2, 0.2]
        hitmap.append(hm)
        miss = int((fg & ~hit2d).sum()); tot = int(fg.sum())
        miss_fracs.append(miss / max(tot, 1))

    nv = len(ids)
    fig, axes = plt.subplots(3, nv, figsize=(5 * nv, 15), squeeze=False)
    for c, vi in enumerate(ids):
        axes[0][c].imshow(phong[c].clip(0, 1)); axes[0][c].axis("off")
        axes[0][c].set_title(f"view {vi}  miss={miss_fracs[c]:.1%}", fontsize=10)
        axes[1][c].imshow(gtxhit[c].clip(0, 1)); axes[1][c].axis("off")
        axes[2][c].imshow(hitmap[c].clip(0, 1)); axes[2][c].axis("off")
    axes[0][0].set_ylabel("Phong", fontsize=11)
    axes[1][0].set_ylabel("GT × hit", fontsize=11)
    axes[2][0].set_ylabel("hit map (green=hit red=miss)", fontsize=11)
    fig.suptitle(f"carve vs GT — {args.scene.name}  "
                 f"(mean fg miss {np.mean(miss_fracs):.1%})", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}  | mean fg miss {np.mean(miss_fracs):.1%}")


if __name__ == "__main__":
    main()
