"""Diagnostic: depth-map space-carving of the scan24 visual hull.

This is geometry-only: it rebuilds the visual hull, uses COLMAP geometric depth
maps to carve voxels strictly in front of measured surfaces, optionally applies a
conservative sparse-SfM AABB clip, and renders hull vs depth-carved meshes with
the same Phong settings used by the hull/photo-carve diagnostics.

Expected depth maps:
    <scene>/stereo/depth_maps/*.geometric.bin

Example:
    python _hull_depth_carve_scan24.py --margin-voxels 3 --votes-req 2 --clip
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from lip_tracer.data import load_views
from lip_tracer.visual_hull import carve

from _hull_photo_carve_scan24 import (
    default_ref_views,
    occ_to_mesh_world,
    render_mesh,
    save_ply,
    world_to_idx,
)


RUN = Path(
    "/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/"
    "outputs/run_20260603_093037_scan24_4962197"
)
OUT_DIR = Path(
    "/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/"
    "outputs/hull_depth_carve_scan24"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Depth-map space-carve the scan24 visual hull; no SDF/MLP."
    )
    ap.add_argument("--run", type=Path, default=RUN,
                    help="run dir whose config.json gives the scene/bound")
    ap.add_argument("--scene", type=Path, default=None,
                    help="scene dir; defaults to --run config.json scene")
    ap.add_argument("--depth-dir", type=Path, default=None,
                    help="COLMAP stereo/depth_maps dir; defaults to <scene>/stereo/depth_maps")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=None,
                    help="world bound; defaults to config eval.bound_dtu")
    ap.add_argument("--down", type=int, default=1,
                    help="load_views downscale before resizing K/masks to depth-map resolution")
    ap.add_argument("--margin-voxels", type=float, default=3.0,
                    help="empty-space margin in voxel units before measured depth")
    ap.add_argument("--votes-req", type=int, default=2,
                    help="number of views that must vote a voxel empty")
    ap.add_argument("--sfm-roi", dest="sfm_roi", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="start from the same padded sparse-SfM ROI used by hull_sfm_roi")
    ap.add_argument("--clip", dest="clip", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="apply sparse-SfM AABB clip after depth carving")
    ap.add_argument("--clip-margin-voxels", type=float, default=6.0,
                    help="AABB clip expansion in voxel units")
    ap.add_argument("--alignment-view", type=int, default=0,
                    help="view used for depth/camera/hull alignment check")
    ap.add_argument("--alignment-thresh", type=float, default=0.8,
                    help="bail if valid foreground back-projected depths inside hull fall below this")
    ap.add_argument("--align-samples", type=int, default=200_000,
                    help="max valid depth pixels sampled for the alignment check")
    ap.add_argument("--chunk-size", type=int, default=500_000,
                    help="occupied voxels projected per chunk per view")
    return ap.parse_args()


def read_colmap_depth_bin(path: Path) -> np.ndarray:
    """Read COLMAP dense stereo .bin array.

    COLMAP dense arrays have an ASCII header "width&height&channels&" followed by
    float32 payload in Fortran order. Depth maps are returned as (H, W).
    """
    with path.open("rb") as f:
        header = b""
        ampersands = 0
        while ampersands < 3:
            c = f.read(1)
            if not c:
                raise ValueError(f"truncated COLMAP depth header: {path}")
            header += c
            if c == b"&":
                ampersands += 1
        width, height, channels = map(int, header[:-1].decode("ascii").split("&"))
        arr = np.fromfile(f, dtype=np.float32)
    expected = width * height * channels
    if arr.size != expected:
        raise ValueError(f"{path} payload has {arr.size} floats, expected {expected}")
    arr = arr.reshape((width, height, channels), order="F")
    arr = np.transpose(arr, (1, 0, 2))
    if channels == 1:
        arr = arr[..., 0]
    return arr.astype(np.float32, copy=False)


def load_depth_maps(scene: Path, n_views: int, depth_dir: Path | None = None) -> list[np.ndarray]:
    depth_dir = depth_dir or (scene / "stereo" / "depth_maps")
    if not depth_dir.is_dir():
        raise FileNotFoundError(
            f"missing COLMAP geometric depth directory: {depth_dir}\n"
            "Expected one *.geometric.bin per view."
        )

    all_depths = sorted(depth_dir.glob("*.geometric.bin"))
    if len(all_depths) != n_views:
        raise FileNotFoundError(
            f"expected {n_views} geometric depth maps in {depth_dir}, found {len(all_depths)}"
        )

    image_paths = sorted(
        p for p in (scene / "image").glob("*")
        if p.is_file() and not p.name.startswith("._")
    )
    if len(image_paths) != n_views:
        raise FileNotFoundError(
            f"expected {n_views} images under {scene / 'image'}, found {len(image_paths)}"
        )

    matched: list[Path] = []
    for img in image_paths:
        candidates = [
            depth_dir / f"{img.name}.geometric.bin",
            depth_dir / f"{img.stem}.geometric.bin",
        ]
        hit = next((p for p in candidates if p.exists()), None)
        if hit is None:
            matched = []
            break
        matched.append(hit)

    depth_paths = matched if matched else all_depths
    if not matched:
        print("  [depth] image-name matching failed; using sorted depth-map order")

    depths = [read_colmap_depth_bin(p) for p in depth_paths]
    shapes = sorted({d.shape for d in depths})
    print(f"  [depth] loaded {len(depths)} geometric maps; shapes={shapes}")
    return depths


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    im = PILImage.fromarray(mask.astype(np.uint8) * 255)
    return np.array(im.resize((w, h), PILImage.NEAREST)) > 127


def camera_at_depth_resolution(
    views: dict, view_id: int, depth_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return K and foreground mask resized from loaded-view resolution to depth resolution."""
    hd, wd = depth_shape
    hv, wv = int(views["H"]), int(views["W"])
    K = views["K"][view_id].numpy().astype(np.float64).copy()
    K[0, :] *= wd / float(wv)
    K[1, :] *= hd / float(hv)
    mask = resize_mask(views["masks"][view_id].numpy().astype(bool), (hd, wd))
    return K, mask


def voxel_world_from_occ_indices(
    occ_idx_zyx: np.ndarray, bound: float, res: int
) -> np.ndarray:
    """Occupied voxel indices (z,y,x) -> world xyz, matching visual_hull.carve."""
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    iz, iy, ix = occ_idx_zyx[:, 0], occ_idx_zyx[:, 1], occ_idx_zyx[:, 2]
    return np.stack([lin[ix], lin[iy], lin[iz]], axis=-1).astype(np.float32)


def backproject_depth_samples(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    selector: np.ndarray,
    max_samples: int,
    seed: int = 0,
) -> np.ndarray:
    ys, xs = np.where(selector)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(xs) > max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(xs), max_samples, replace=False)
        xs, ys = xs[sel], ys[sel]
    d = depth[ys, xs].astype(np.float64)
    xcam = np.stack([
        (xs.astype(np.float64) - K[0, 2]) / K[0, 0] * d,
        (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * d,
        d,
    ], axis=-1)
    R = c2w[:3, :3].astype(np.float64)
    c = c2w[:3, 3].astype(np.float64)
    world = xcam @ R.T + c[None]
    return world.astype(np.float32)


def fraction_points_inside_hull(
    pts: np.ndarray,
    occ: np.ndarray,
    bound: float,
) -> float:
    iz, iy, ix, ok = world_to_idx(pts, bound, occ.shape[0])
    inside = np.zeros(len(pts), dtype=bool)
    if ok.any():
        inside[ok] = occ[iz[ok], iy[ok], ix[ok]]
    return float(inside.mean()) if len(inside) else 0.0


def alignment_check(
    view_id: int,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    mask: np.ndarray,
    occ: np.ndarray,
    bound: float,
    max_samples: int,
) -> dict:
    valid = np.isfinite(depth) & (depth > 0)
    valid_fg = valid & mask
    pts_all = backproject_depth_samples(depth, K, c2w, valid, max_samples, seed=0)
    pts_fg = backproject_depth_samples(depth, K, c2w, valid_fg, max_samples, seed=1)
    frac_all = fraction_points_inside_hull(pts_all, occ, bound)
    frac_fg = fraction_points_inside_hull(pts_fg, occ, bound)
    valid_px = int(valid.sum())
    valid_fg_px = int(valid_fg.sum())
    return {
        "view": int(view_id),
        "valid_depth_px": valid_px,
        "valid_fg_depth_px": valid_fg_px,
        "sampled_depth_px": int(len(pts_all)),
        "sampled_fg_depth_px": int(len(pts_fg)),
        "frac_valid_depth_inside_hull": frac_all,
        "frac_valid_fg_depth_inside_hull": frac_fg,
    }


def depth_vote_view(
    view_id: int,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    mask: np.ndarray,
    occ_idx: np.ndarray,
    occ_pts: np.ndarray,
    votes: np.ndarray,
    margin: float,
    chunk_size: int,
) -> dict:
    R = c2w[:3, :3].astype(np.float64)
    c = c2w[:3, 3].astype(np.float64)
    h, w = depth.shape
    valid_depth_px = int((np.isfinite(depth) & (depth > 0)).sum())
    fg_px = int(mask.sum())
    n_voted = 0

    for start in range(0, len(occ_pts), chunk_size):
        end = min(start + chunk_size, len(occ_pts))
        pts = occ_pts[start:end].astype(np.float64, copy=False)
        cam = (pts - c[None]) @ R
        z = cam[:, 2]
        front = z > 0
        zsafe = np.where(front, z, 1.0)
        u = cam[:, 0] / zsafe * K[0, 0] + K[0, 2]
        v = cam[:, 1] / zsafe * K[1, 1] + K[1, 2]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not inb.any():
            continue

        d = np.zeros(end - start, dtype=np.float64)
        fg = np.zeros(end - start, dtype=bool)
        d[inb] = depth[vi[inb], ui[inb]]
        fg[inb] = mask[vi[inb], ui[inb]]
        valid = inb & fg & np.isfinite(d) & (d > 0)
        empty = valid & (z < (d - margin))
        if empty.any():
            idx = occ_idx[start:end][empty]
            votes[idx[:, 0], idx[:, 1], idx[:, 2]] += 1
            n_voted += int(empty.sum())

    return {
        "view": int(view_id),
        "valid_depth_px": valid_depth_px,
        "fg_px": fg_px,
        "voxels_voted_empty": n_voted,
    }


def sfm_aabb_clip_mask(
    scene: Path,
    occ: np.ndarray,
    bound: float,
    margin: float,
    clip_top: bool = True,
) -> tuple[np.ndarray, dict]:
    sfm_path = scene / "sparse_sfm_points.txt"
    if not sfm_path.exists():
        raise FileNotFoundError(f"--clip requested, but missing sparse points: {sfm_path}")
    pts = np.loadtxt(sfm_path, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[np.all(np.abs(pts) <= bound, axis=1)]
    if len(pts) == 0:
        raise ValueError(f"--clip requested, but no finite in-bound SfM points in {sfm_path}")

    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    if not clip_top:
        # don't cap the +y face: textureless caps (e.g. the scan65 skull dome)
        # have no sparse SfM points, so the AABB top slices the dome flat. Let the
        # sphere/depth define the top; keep the tight side/bottom clip.
        hi[1] = float(bound)

    res = occ.shape[0]
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    z_ok = (lin >= lo[2]) & (lin <= hi[2])
    y_ok = (lin >= lo[1]) & (lin <= hi[1])
    x_ok = (lin >= lo[0]) & (lin <= hi[0])
    keep = z_ok[:, None, None] & y_ok[None, :, None] & x_ok[None, None, :]
    info = {
        "mode": "sfm_aabb",
        "points": int(len(pts)),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "margin": float(margin),
        "clip_top": bool(clip_top),
    }
    return keep, info


def sfm_roi_bounds(scene: Path, bound: float) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """Padded sparse-SfM AABB matching lip_tracer.train hull_sfm_roi."""
    sfm_path = scene / "sparse_sfm_points.txt"
    if not sfm_path.exists():
        raise FileNotFoundError(f"--sfm-roi requested, but missing sparse points: {sfm_path}")
    pts = np.loadtxt(sfm_path, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        raise ValueError(f"--sfm-roi requested, but no finite SfM points in {sfm_path}")
    sfm_lo = pts.min(axis=0)
    sfm_hi = pts.max(axis=0)
    pad = np.maximum(0.15, 0.15 * (sfm_hi - sfm_lo))
    lo = np.maximum(sfm_lo - pad, -bound).astype(np.float32)
    hi = np.minimum(sfm_hi + pad, bound).astype(np.float32)
    info = {
        "points": int(len(pts)),
        "sfm_lo": sfm_lo.tolist(),
        "sfm_hi": sfm_hi.tolist(),
        "pad": pad.tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
    }
    return (lo, hi), info


def gray_photo(views: dict, vi: int, h: int, w: int) -> np.ndarray:
    im = views["images"][vi]
    if hasattr(im, "numpy"):
        im = im.numpy()
    g = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
    g = np.array(
        PILImage.fromarray((g * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
    ) / 255.0
    return np.stack([g, g, g], axis=-1)


def save_compare_render(
    out_path: Path,
    verts_hull: np.ndarray,
    faces_hull: np.ndarray,
    verts_carved: np.ndarray,
    faces_carved: np.ndarray,
    views_hi: dict,
    ref_views: list[int],
    pct_removed_final: float,
    pct_removed_depth: float,
    pct_removed_clip: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    render_views = {
        "H": views_hi["H"],
        "W": views_hi["W"],
        "K": views_hi["K"].numpy() if hasattr(views_hi["K"], "numpy") else views_hi["K"],
        "c2w": views_hi["c2w"].numpy() if hasattr(views_hi["c2w"], "numpy") else views_hi["c2w"],
        "images": views_hi["images"],
    }
    imgs_h = render_mesh(verts_hull, faces_hull, render_views, ref_views)
    imgs_c = render_mesh(verts_carved, faces_carved, render_views, ref_views)
    h, w = imgs_h[0].shape[:2]

    fig, axes = plt.subplots(len(ref_views), 3, figsize=(12, 4 * len(ref_views)), squeeze=False)
    for row, vi in enumerate(ref_views):
        panels = [
            (gray_photo(views_hi, vi, h, w), f"photo v{vi}"),
            (imgs_h[row], "vh_original"),
            (imgs_c[row], "vh_depth_carved"),
        ]
        for col, (img, label) in enumerate(panels):
            axes[row][col].imshow(np.clip(img, 0, 1))
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(label, fontsize=11)
    fig.suptitle(
        "scan24 visual hull vs depth-carved "
        f"(removed {pct_removed_final:.1f}%: depth {pct_removed_depth:.1f}%"
        f" + clip {pct_removed_clip:.1f}%)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((args.run / "config.json").read_text())
    scene = args.scene or Path(cfg["scene"])
    bound = float(args.bound if args.bound is not None else cfg["eval"]["bound_dtu"])
    res = int(args.res)
    voxel = 2 * bound / max(res - 1, 1)
    margin = args.margin_voxels * voxel
    clip_margin = args.clip_margin_voxels * voxel

    print(f"scene={scene}")
    print(f"res={res}  bound={bound}  voxel={voxel:.6f}")
    print(f"depth margin={margin:.6f} ({args.margin_voxels:g} voxels)")
    print(f"votes_req={args.votes_req}  sfm_roi={args.sfm_roi}  clip={args.clip}")

    roi_bounds = None
    roi_info = None
    if args.sfm_roi:
        roi_bounds, roi_info = sfm_roi_bounds(scene, bound)
        lo, hi = roi_bounds
        print(
            "SFM ROI "
            f"x=[{lo[0]:+.3f},{hi[0]:+.3f}] "
            f"y=[{lo[1]:+.3f},{hi[1]:+.3f}] "
            f"z=[{lo[2]:+.3f},{hi[2]:+.3f}] "
            f"points={roi_info['points']}"
        )

    print(f"\nloading views (down={args.down}) ...")
    views = load_views(scene, down=args.down)
    views_hi = load_views(scene, down=1)
    n_views = int(views["c2w"].shape[0])

    print("\nloading COLMAP geometric depth maps ...")
    depths = load_depth_maps(scene, n_views, args.depth_dir)
    cameras = []
    for vi, depth in enumerate(depths):
        Kd, md = camera_at_depth_resolution(views, vi, depth.shape)
        cameras.append((Kd, md))

    print(f"\ncarving visual hull (res={res}, border_aware=True) ...")
    occ = carve(scene=scene, res=res, bound=bound, roi_bounds=roi_bounds,
                border_aware=True)
    n_occ0 = int(occ.sum())
    print(f"  occupied voxels: {n_occ0} / {occ.size}")
    verts_h, faces_h = occ_to_mesh_world(occ, bound)
    save_ply(verts_h, faces_h, args.out_dir / "vh_original.ply")

    print("\nchecking depth-map/camera/hull alignment ...")
    av = int(args.alignment_view)
    if av < 0 or av >= n_views:
        raise ValueError(f"--alignment-view {av} outside [0, {n_views - 1}]")
    K_align, mask_align = cameras[av]
    align = alignment_check(
        av, depths[av], K_align, views["c2w"][av].numpy(),
        mask_align, occ, bound, args.align_samples,
    )
    print(
        f"  view {av}: valid_depth_px={align['valid_depth_px']}  "
        f"valid_fg_depth_px={align['valid_fg_depth_px']}  "
        f"sampled={align['sampled_depth_px']}/{align['sampled_fg_depth_px']}  "
        f"inside_hull_all={align['frac_valid_depth_inside_hull']:.3f}  "
        f"inside_hull_fg={align['frac_valid_fg_depth_inside_hull']:.3f}"
    )
    if align["frac_valid_fg_depth_inside_hull"] < args.alignment_thresh:
        raise RuntimeError(
            "depth/camera alignment check failed: "
            f"{align['frac_valid_fg_depth_inside_hull']:.3f} < {args.alignment_thresh:.3f}. "
            "Depth scale, pose frame, or K resolution is likely wrong; bailing before carving."
        )

    print("\ndepth-space carving ...")
    occ_idx = np.argwhere(occ)  # (M,3), axes are (z,y,x)
    occ_pts = voxel_world_from_occ_indices(occ_idx, bound, res)
    votes = np.zeros_like(occ, dtype=np.uint16)
    per_view: list[dict] = []
    for vi, depth in enumerate(depths):
        Kd, md = cameras[vi]
        stats = depth_vote_view(
            vi, depth, Kd, views["c2w"][vi].numpy(), md,
            occ_idx, occ_pts, votes, margin, args.chunk_size,
        )
        per_view.append(stats)
        print(
            f"  view {vi:02d}: valid_depth_px={stats['valid_depth_px']:8d}  "
            f"fg_px={stats['fg_px']:8d}  "
            f"voxels_voted_empty={stats['voxels_voted_empty']:8d}"
        )

    depth_remove = (votes >= args.votes_req) & occ
    depth_carved = occ & ~depth_remove
    n_depth_removed = int(depth_remove.sum())
    print(
        f"\nremoved by depth votes: {n_depth_removed} / {n_occ0} "
        f"({100 * n_depth_removed / max(n_occ0, 1):.2f}%)"
    )

    clip_info = None
    n_clip_removed = 0
    final_occ = depth_carved
    if args.clip:
        print("\napplying sparse-SfM AABB clip ...")
        keep, clip_info = sfm_aabb_clip_mask(scene, depth_carved, bound, clip_margin)
        final_occ = depth_carved & keep
        n_clip_removed = int(depth_carved.sum() - final_occ.sum())
        print(
            f"  SfM points={clip_info['points']}  "
            f"clip_margin={clip_margin:.6f} ({args.clip_margin_voxels:g} voxels)"
        )
        print(
            f"  removed by clip after depth-carve: {n_clip_removed} "
            f"({100 * n_clip_removed / max(n_occ0, 1):.2f}% of original hull)"
        )

    n_final = int(final_occ.sum())
    n_final_removed = n_occ0 - n_final
    pct_depth = 100 * n_depth_removed / max(n_occ0, 1)
    pct_clip = 100 * n_clip_removed / max(n_occ0, 1)
    pct_final = 100 * n_final_removed / max(n_occ0, 1)
    print(
        f"\ntotal removed: {n_final_removed} / {n_occ0} ({pct_final:.2f}%) "
        f"[depth={pct_depth:.2f}%, clip={pct_clip:.2f}%]"
    )

    verts_c, faces_c = occ_to_mesh_world(final_occ, bound)
    save_ply(verts_c, faces_c, args.out_dir / "vh_depth_carved.ply")

    print("\nrendering comparison (init-viz Phong settings) ...")
    ref_views = default_ref_views(views_hi["c2w"].numpy())
    save_compare_render(
        args.out_dir / "vh_depth_compare.png",
        verts_h, faces_h, verts_c, faces_c,
        views_hi, ref_views,
        pct_final, pct_depth, pct_clip,
    )

    summary = {
        "scene": str(scene),
        "settings": {
            "res": res,
            "bound": bound,
            "down": args.down,
            "margin_voxels": args.margin_voxels,
            "margin": margin,
            "votes_req": args.votes_req,
            "sfm_roi": args.sfm_roi,
            "clip": args.clip,
            "clip_margin_voxels": args.clip_margin_voxels,
            "clip_margin": clip_margin,
            "chunk_size": args.chunk_size,
        },
        "alignment": align,
        "sfm_roi_info": roi_info,
        "per_view": per_view,
        "hull_voxels": n_occ0,
        "voxels_removed_depth": n_depth_removed,
        "pct_removed_depth": pct_depth,
        "voxels_removed_clip": n_clip_removed,
        "pct_removed_clip": pct_clip,
        "voxels_removed_total": n_final_removed,
        "pct_removed_total": pct_final,
        "clip_info": clip_info,
        "runtime_sec": time.perf_counter() - t0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY ================")
    print(f"  alignment inside hull : {align['frac_valid_fg_depth_inside_hull']:.3f}")
    print(f"  hull voxels           : {n_occ0}")
    print(f"  removed by depth      : {n_depth_removed} ({pct_depth:.2f}%)")
    print(f"  removed by clip       : {n_clip_removed} ({pct_clip:.2f}%)")
    print(f"  removed total         : {n_final_removed} ({pct_final:.2f}%)")
    print(f"  outputs               : {args.out_dir}")
    print("=========================================")


if __name__ == "__main__":
    main()
