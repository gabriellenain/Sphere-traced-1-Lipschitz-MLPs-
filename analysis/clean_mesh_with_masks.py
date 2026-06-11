#!/usr/bin/env python3
"""Clean a normalized DTU mesh by projecting it into foreground masks.

The mesh and scene must be in the same normalized DTU coordinates used by the
repo checkpoints. This is intended for cleaning NeuS/MC exports before
distilling them into the CPL+PE model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _load_masks(scene: Path, dilate: int) -> tuple[np.lib.npyio.NpzFile, list[np.ndarray]]:
    from PIL import Image

    cam_dict = np.load(scene / "cameras.npz")
    mask_paths = sorted(p for p in (scene / "mask").glob("*.png") if not p.name.startswith("."))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found under {scene / 'mask'}")

    masks = []
    for path in mask_paths:
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask > 0
        if dilate > 0:
            try:
                from scipy.ndimage import binary_dilation
                mask = binary_dilation(mask, iterations=dilate)
            except Exception:
                # Conservative fallback if scipy is unavailable.
                from PIL import ImageFilter
                img = Image.fromarray(mask.astype(np.uint8) * 255)
                for _ in range(dilate):
                    img = img.filter(ImageFilter.MaxFilter(3))
                mask = np.array(img) > 0
        masks.append(mask)
    return cam_dict, masks


def _mask_visibility_counts(
    pts: np.ndarray,
    scene: Path,
    chunk: int,
    dilate: int,
    coords: str,
) -> np.ndarray:
    cam_dict, masks = _load_masks(scene, dilate=dilate)
    counts = np.zeros(len(pts), dtype=np.uint16)

    for start in range(0, len(pts), chunk):
        end = min(start + chunk, len(pts))
        p = pts[start:end].astype(np.float64)
        p_h = np.concatenate([p, np.ones((len(p), 1), dtype=np.float64)], axis=1).T
        c = np.zeros(len(p), dtype=np.uint16)

        for i, mask in enumerate(masks):
            key = f"world_mat_{i}"
            if key not in cam_dict:
                continue
            P = cam_dict[key].astype(np.float64)
            H, W = mask.shape[:2]
            if coords == "normalized":
                scale_key = f"scale_mat_{i}"
                if scale_key not in cam_dict:
                    raise KeyError(
                        f"{scale_key} missing from {scene / 'cameras.npz'}; "
                        "use --coords world if the mesh is already in DTU world coordinates"
                    )
                proj = P @ (cam_dict[scale_key].astype(np.float64) @ p_h)
            else:
                proj = P @ p_h
            z = proj[2]
            front = z > 0
            u = np.where(front, proj[0] / np.where(front, z, 1.0), -1.0)
            v = np.where(front, proj[1] / np.where(front, z, 1.0), -1.0)
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)
            in_bounds = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            idx = np.where(in_bounds)[0]
            if len(idx):
                c[idx[mask[vi[idx], ui[idx]]]] += 1

        counts[start:end] = c
        print(
            f"[mask] vertices {end:,}/{len(pts):,} "
            f"keep@1={(counts[:end] >= 1).mean():.1%}",
            flush=True,
        )
    return counts


def _face_keep_from_vertices(faces: np.ndarray, keep_v: np.ndarray, mode: str) -> np.ndarray:
    vals = keep_v[faces]
    if mode == "all":
        return vals.all(axis=1)
    if mode == "majority":
        return vals.sum(axis=1) >= 2
    return vals.any(axis=1)


def _save_sample_preview(mesh: trimesh.Trimesh, out: Path, n: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts) == 0:
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(verts), size=min(n, len(verts)), replace=False)
    pts = verts[idx]
    zspan = max(float(np.ptp(pts[:, 2])), 1e-6)
    colors = np.clip((pts[:, 2] - pts[:, 2].min()) / zspan, 0, 1)
    ctr = verts.mean(axis=0)
    half = max(0.55 * float(np.ptp(verts, axis=0).max()), 0.1)

    fig = plt.figure(figsize=(12, 5))
    for i, (elev, azim) in enumerate([(20, 35), (20, 125)], start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=colors, cmap="viridis", s=0.15, alpha=0.65, linewidths=0,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"mask-cleaned mesh view {i}", fontsize=10)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-views", type=int, default=1,
                    help="minimum number of foreground masks a vertex must project into")
    ap.add_argument("--dilate", type=int, default=3,
                    help="foreground mask dilation iterations before projection")
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--coords", choices=["normalized", "world"], default="normalized",
                    help="coordinate system of --mesh; repo checkpoints/MC meshes are normalized")
    ap.add_argument("--face-mode", choices=["any", "majority", "all"], default="majority")
    ap.add_argument("--keep-largest", action="store_true")
    ap.add_argument("--preview", type=Path, default=None)
    ap.add_argument("--preview-points", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms)
    print(
        f"[mesh] input verts={len(mesh.vertices):,} faces={len(mesh.faces):,} "
        f"bounds={mesh.bounds.tolist()}",
        flush=True,
    )

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    counts = _mask_visibility_counts(
        verts,
        args.scene,
        chunk=args.chunk,
        dilate=args.dilate,
        coords=args.coords,
    )
    keep_v = counts >= args.min_views
    keep_f = _face_keep_from_vertices(faces, keep_v, args.face_mode)
    print(
        f"[mask] kept vertices={keep_v.sum():,}/{len(keep_v):,} "
        f"faces={keep_f.sum():,}/{len(keep_f):,} "
        f"(min_views={args.min_views}, face_mode={args.face_mode})",
        flush=True,
    )

    # Rebuild directly from kept faces instead of relying on trimesh.submesh,
    # whose return type differs across trimesh versions.
    cleaned = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=faces[keep_f],
        process=False,
    )
    cleaned.remove_unreferenced_vertices()

    if args.keep_largest and len(cleaned.faces):
        comps = cleaned.split(only_watertight=False)
        cleaned = max(comps, key=lambda m: len(m.faces))
        print(
            f"[mesh] kept largest component: verts={len(cleaned.vertices):,} "
            f"faces={len(cleaned.faces):,}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(args.out)
    meta = {
        "mesh": str(args.mesh),
        "scene": str(args.scene),
        "out": str(args.out),
        "min_views": args.min_views,
        "dilate": args.dilate,
        "face_mode": args.face_mode,
        "coords": args.coords,
        "keep_largest": args.keep_largest,
        "input_vertices": int(len(mesh.vertices)),
        "input_faces": int(len(mesh.faces)),
        "output_vertices": int(len(cleaned.vertices)),
        "output_faces": int(len(cleaned.faces)),
        "output_bounds": cleaned.bounds.tolist() if len(cleaned.vertices) else None,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(
        f"[mesh] wrote {args.out} "
        f"verts={len(cleaned.vertices):,} faces={len(cleaned.faces):,}",
        flush=True,
    )

    if args.preview is not None:
        _save_sample_preview(cleaned, args.preview, args.preview_points, args.seed)
        print(f"[mesh] wrote preview {args.preview}", flush=True)


if __name__ == "__main__":
    main()
