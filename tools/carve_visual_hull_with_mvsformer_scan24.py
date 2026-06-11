#!/usr/bin/env python3
"""Carve scan24's edge-aware visual hull with MVSFormer++ depths.

This keeps visual-hull topology as the starting solid, applies the same sparse
SfM ROI used by the hull diagnostics, then removes voxels that are confidently in
front of MVSFormer++ depth surfaces in multiple views.  It is a diagnostic only:
no neural field training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the package importable when run as `python tools/<this>.py` from the repo
# root (Python only puts this script's own dir on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from lip_tracer.data import load_views
from lip_tracer.geomvs import _read_pfm, load_mvsformer_depths_idr
from lip_tracer.visual_hull import carve, keep_largest_component

from _hull_depth_carve_scan24 import (
    default_ref_views,
    occ_to_mesh_world,
    render_mesh,
    save_ply,
    sfm_aabb_clip_mask,
    sfm_roi_bounds,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=Path("data/dtu_idr/scan24"))
    ap.add_argument("--depth-dir", type=Path,
                    default=Path("_diagnostics/mvsformer_scan24_idr/depths_1152x864"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("_diagnostics/mvsformer_scan24_idr/hull_carve"))
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--votes-req", type=int, default=3)
    ap.add_argument("--vh-percentile", type=float, default=0.99,
                    help="TnT visibility-aware visual-hull foreground ratio p")
    ap.add_argument("--vh-min-views", type=int, default=8,
                    help="TnT minimum in-frame views for percentile visual hull")
    ap.add_argument("--margin-voxels", type=float, default=3.0,
                    help="carve only voxels at least this many voxels in front of depth")
    ap.add_argument("--clip-margin-voxels", type=float, default=6.0)
    ap.add_argument("--sfm-clip", action="store_true",
                    help="force the sparse-SfM ROI clip + largest-component cleanup "
                         "even for TnT scenes (needs sparse_sfm_points.txt). Off by "
                         "default for TnT open scenes; on for object-centric TnT "
                         "(e.g. Truck) where the COLMAP cloud bounds the object.")
    ap.add_argument("--chunk-size", type=int, default=500_000)
    ap.add_argument("--good-views", type=Path, default=None,
                    help="file of view indices to keep; drops noisy-mask views "
                         "from BOTH the percentile visual hull and the depth carve")
    return ap.parse_args()


def voxel_world_from_occ_indices(occ_idx_zyx: np.ndarray, bound: float, res: int) -> np.ndarray:
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    iz, iy, ix = occ_idx_zyx[:, 0], occ_idx_zyx[:, 1], occ_idx_zyx[:, 2]
    return np.stack([lin[ix], lin[iy], lin[iz]], axis=-1).astype(np.float32)


def depth_vote_view(
    view_id: int,
    depth: np.ndarray,
    valid_depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    occ_idx: np.ndarray,
    occ_pts: np.ndarray,
    votes: np.ndarray,
    margin: float,
    chunk_size: int,
) -> dict:
    R = c2w[:3, :3].astype(np.float64)
    c = c2w[:3, 3].astype(np.float64)
    h, w = depth.shape
    n_voted = 0
    valid_depth_px = int(valid_depth.sum())

    for start in range(0, len(occ_pts), chunk_size):
        end = min(start + chunk_size, len(occ_pts))
        pts = occ_pts[start:end].astype(np.float64, copy=False)
        cam = (pts - c[None]) @ R
        z = cam[:, 2]
        front = z > 1e-5
        zsafe = np.where(front, z, 1.0)
        u = cam[:, 0] / zsafe * K[0, 0] + K[0, 2]
        v = cam[:, 1] / zsafe * K[1, 1] + K[1, 2]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not inb.any():
            continue

        d = np.zeros(end - start, dtype=np.float64)
        good = np.zeros(end - start, dtype=bool)
        d[inb] = depth[vi[inb], ui[inb]]
        good[inb] = valid_depth[vi[inb], ui[inb]]
        empty = inb & good & (z < (d - margin))
        if empty.any():
            idx = occ_idx[start:end][empty]
            votes[idx[:, 0], idx[:, 1], idx[:, 2]] += 1
            n_voted += int(empty.sum())

    return {
        "view": int(view_id),
        "valid_depth_px": valid_depth_px,
        "voxels_voted_empty": n_voted,
    }


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(im.resize((w, h), Image.NEAREST)) > 127


def load_mvsformer_depths_tnt(scene: Path, depth_dir: Path,
                              conf_thresh: float) -> dict | None:
    """Load MVSFormer++ depths for NSVF/Tanks-and-Temples scenes.

    Depths are produced by tools/precompute_mvsformer_depths.py from the same
    normalised TnT camera frame used by lip_tracer.data.load_views.
    """
    scan_dirs = [d for d in depth_dir.iterdir()
                 if d.is_dir() and (d / "depth_est").exists()]
    scan_root = scan_dirs[0] if scan_dirs else depth_dir
    pfm_files = sorted((scan_root / "depth_est").glob("*.pfm"))
    if not pfm_files:
        print(f"  [geomvs] no .pfm files under {scan_root}")
        return None

    views = load_views(scene, down=1)
    c2w = views["c2w"].numpy().astype(np.float32)
    K0 = views["K"].numpy().astype(np.float32)
    masks = views["masks"].numpy().astype(bool)
    H0, W0 = int(views["H"]), int(views["W"])
    if len(pfm_files) > len(c2w):
        raise ValueError(f"more depth maps ({len(pfm_files)}) than views ({len(c2w)})")

    depths, valid, Ks = [], [], []
    for i, pfm in enumerate(pfm_files):
        depth = np.asarray(_read_pfm(pfm), dtype=np.float32)
        conf = np.load(scan_root / "confidence" / f"{pfm.stem}.npy")
        if conf.dtype == np.uint8:
            conf = conf.astype(np.float32) / 255.0
        H, W = depth.shape
        K = K0[i].copy()
        K[0, :] *= W / float(W0)
        K[1, :] *= H / float(H0)
        m = resize_mask(masks[i], (H, W))
        good = (conf > conf_thresh) & (depth > 1e-3) & m
        depths.append(depth)
        valid.append(good)
        Ks.append(K)
        if i % 25 == 0:
            pct = 100 * good.mean()
            zr = depth[good] if good.any() else np.array([0.0], dtype=np.float32)
            print(f"  [geomvs] tnt mvsf view {i:03d}: valid={pct:.1f}% "
                  f"z=[{float(zr.min()):.3f},{float(zr.max()):.3f}]")

    print(f"  [geomvs] loaded {len(depths)} MVSFormer++ depth maps from {scan_root}")
    return {
        "depths": depths,
        "valid": valid,
        "K": np.stack(Ks),
        "c2w": c2w[:len(depths)],
        "H": depths[0].shape[0],
        "W": depths[0].shape[1],
    }


def load_mvsformer_depths_any(scene: Path, depth_dir: Path,
                              conf_thresh: float) -> dict | None:
    is_tnt = (scene / "intrinsics.txt").exists() and (scene / "pose").is_dir()
    if is_tnt:
        return load_mvsformer_depths_tnt(scene, depth_dir, conf_thresh)
    return load_mvsformer_depths_idr(scene, depth_dir,
                                     conf_thresh=conf_thresh,
                                     use_idr_mask=True)


def save_compare_render(out_path: Path, verts_h: np.ndarray, faces_h: np.ndarray,
                        verts_c: np.ndarray, faces_c: np.ndarray, views_hi: dict,
                        pct_removed: float, scene_label: str,
                        hull_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rv = {
        "H": views_hi["H"],
        "W": views_hi["W"],
        "K": views_hi["K"].numpy(),
        "c2w": views_hi["c2w"].numpy(),
    }
    ref_views = default_ref_views(rv["c2w"])
    imgs_h = render_mesh(verts_h, faces_h, rv, ref_views)
    imgs_c = render_mesh(verts_c, faces_c, rv, ref_views)

    fig, axes = plt.subplots(len(ref_views), 2, figsize=(8, 4 * len(ref_views)), squeeze=False)
    for row, vi in enumerate(ref_views):
        for col, (img, label) in enumerate([
            (imgs_h[row], hull_label),
            (imgs_c[row], "VH carved by MVSFormer++"),
        ]):
            axes[row, col].imshow(np.clip(img, 0, 1))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label, fontsize=11)
        axes[row, 0].set_ylabel(f"view {vi}", fontsize=10)
    fig.suptitle(f"{scene_label} visual hull carved by MVSFormer++ depths "
                 f"({pct_removed:.1f}% voxels removed)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def save_sdf_grid(occ: np.ndarray, bound: float, out_path: Path) -> None:
    from scipy.ndimage import distance_transform_edt
    voxel = 2 * bound / max(occ.shape[0] - 1, 1)
    sdf = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * voxel
    np.save(out_path, sdf.astype(np.float32))
    print(f"  saved {out_path}  range=[{sdf.min():+.4f},{sdf.max():+.4f}]")


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    voxel = 2 * args.bound / max(args.res - 1, 1)
    margin = args.margin_voxels * voxel
    clip_margin = args.clip_margin_voxels * voxel

    print(f"scene={args.scene}")
    print(f"res={args.res} bound={args.bound} voxel={voxel:.6f}")
    print(f"MVSFormer conf_thr={args.conf_thr} margin={margin:.6f} votes_req={args.votes_req}")

    keep = None
    if args.good_views is not None:
        keep = [int(t) for t in args.good_views.read_text().split()]
        print(f"good-views: keeping {len(keep)} views from {args.good_views}")

    is_tnt = (args.scene / "intrinsics.txt").exists() and (args.scene / "pose").is_dir()
    # bmvs (NeuS BlendedMVS) ships no sparse_sfm_points.txt → no SFM ROI / AABB clip.
    # It is still a closed object, so keep the DTU border-aware carve + largest
    # component (unlike the TnT open-scene percentile hull).
    has_sfm = (args.scene / "sparse_sfm_points.txt").exists()
    use_sfm = has_sfm and (not is_tnt or args.sfm_clip)
    roi_bounds = None
    roi_info = None
    if use_sfm:
        hull_label = "visual hull + SFM ROI" if is_tnt else "edge-aware VH + SFM ROI"
        roi_bounds, roi_info = sfm_roi_bounds(args.scene, args.bound)
        print(f"SFM ROI lo={np.round(roi_bounds[0], 4)} hi={np.round(roi_bounds[1], 4)}")
    elif is_tnt:
        hull_label = "visual hull"
        print("TnT scene: using visibility-aware percentile visual hull; no SFM ROI clip")
    else:
        hull_label = "edge-aware VH (no SFM)"
        print("no sparse_sfm_points.txt: edge-aware VH, no SFM ROI / clip")

    print("\nloading MVSFormer++ depths ...")
    mvs = load_mvsformer_depths_any(args.scene, args.depth_dir, args.conf_thr)
    if mvs is None:
        raise RuntimeError("failed to load MVSFormer++ depths")
    if keep is not None:
        nd = len(mvs["depths"])
        sel = [i for i in keep if 0 <= i < nd]
        if len(sel) != len(keep):
            print(f"  WARN: {len(keep) - len(sel)} keep indices out of range "
                  f"for n_depths={nd}; ignored")
        mvs["depths"] = [mvs["depths"][i] for i in sel]
        mvs["valid"] = [mvs["valid"][i] for i in sel]
        mvs["K"] = mvs["K"][sel]
        mvs["c2w"] = mvs["c2w"][sel]
        print(f"  depth carve uses {len(sel)} good views")

    print("\ncarving visual hull ...")
    occ = carve(scene=args.scene, res=args.res, bound=args.bound,
                roi_bounds=roi_bounds, border_aware=not is_tnt,
                vh_percentile=args.vh_percentile,
                vh_min_views=args.vh_min_views, view_keep=keep)
    # carve() applies roi_bounds only on the DTU/border-aware path; the TnT
    # percentile path ignores it. When --sfm-clip forces SFM on for a TnT scene,
    # apply the ROI to the hull here so the hull (and everything downstream) is
    # bounded to the object cloud, matching the standalone sweep.
    if use_sfm and is_tnt and roi_bounds is not None:
        lo, hi = roi_bounds
        lin = np.linspace(-args.bound, args.bound, args.res, dtype=np.float32)
        roi = ((lin[:, None, None] >= lo[2]) & (lin[:, None, None] <= hi[2]) &
               (lin[None, :, None] >= lo[1]) & (lin[None, :, None] <= hi[1]) &
               (lin[None, None, :] >= lo[0]) & (lin[None, None, :] <= hi[0]))
        occ &= roi
    if (not is_tnt) or use_sfm:
        occ = keep_largest_component(occ)
    n_occ0 = int(occ.sum())
    print(f"  hull voxels: {n_occ0}")
    verts_h, faces_h = occ_to_mesh_world(occ, args.bound)
    hull_mesh_name = "vh_tnt.ply" if is_tnt else "vh_edge_roi.ply"
    hull_sdf_name = "sdf_vh_tnt.npy" if is_tnt else "sdf_vh_edge_roi.npy"
    save_ply(verts_h, faces_h, args.out_dir / hull_mesh_name)
    save_sdf_grid(occ, args.bound, args.out_dir / hull_sdf_name)

    print("\nMVSFormer depth-space free carving ...")
    occ_idx = np.argwhere(occ)
    occ_pts = voxel_world_from_occ_indices(occ_idx, args.bound, args.res)
    votes = np.zeros_like(occ, dtype=np.uint16)
    per_view = []
    for vi, (depth_t, valid_t) in enumerate(zip(mvs["depths"], mvs["valid"])):
        stats = depth_vote_view(
            vi,
            np.asarray(depth_t, dtype=np.float32),
            np.asarray(valid_t, dtype=bool),
            mvs["K"][vi],
            mvs["c2w"][vi],
            occ_idx,
            occ_pts,
            votes,
            margin,
            args.chunk_size,
        )
        per_view.append(stats)
        print(f"  view {vi:02d}: valid={stats['valid_depth_px']:8d} "
              f"voted_empty={stats['voxels_voted_empty']:8d}")

    remove = (votes >= args.votes_req) & occ
    carved = occ & ~remove
    n_depth_removed = int(remove.sum())
    print(f"\nremoved by MVSFormer depth votes: {n_depth_removed} / {n_occ0} "
          f"({100*n_depth_removed/max(n_occ0,1):.2f}%)")

    clip_info = None
    if not use_sfm:
        if is_tnt:
            print("\nskipping sparse-SfM AABB clip for TnT open scene")
            carved_clip = carved
        else:
            print("\nno SFM points: skipping AABB clip, keeping largest component")
            carved_clip = keep_largest_component(carved)
        n_clip_removed = int(carved.sum() - carved_clip.sum())
    else:
        print("\napplying sparse-SfM AABB clip after carving ...")
        keep, clip_info = sfm_aabb_clip_mask(args.scene, carved, args.bound, clip_margin)
        carved_clip = keep_largest_component(carved & keep)
        n_clip_removed = int(carved.sum() - carved_clip.sum())
    n_final = int(carved_clip.sum())
    pct_total = 100 * (n_occ0 - n_final) / max(n_occ0, 1)
    print(f"removed by post clip/component: {n_clip_removed} ({100*n_clip_removed/max(n_occ0,1):.2f}%)")
    print(f"total removed: {n_occ0 - n_final} / {n_occ0} ({pct_total:.2f}%)")

    verts_c, faces_c = occ_to_mesh_world(carved_clip, args.bound)
    save_ply(verts_c, faces_c, args.out_dir / "vh_mvsformer_carved.ply")
    save_sdf_grid(carved_clip, args.bound, args.out_dir / "sdf_vh_mvsformer_carved.npy")

    print("\nrendering Phong comparison ...")
    views_hi = load_views(args.scene, down=1)
    save_compare_render(args.out_dir / "vh_mvsformer_carve_compare.png",
                        verts_h, faces_h, verts_c, faces_c, views_hi, pct_total,
                        args.scene.name, hull_label)

    summary = {
        "scene": str(args.scene),
        "depth_dir": str(args.depth_dir),
        "settings": {
            "res": args.res,
            "bound": args.bound,
            "conf_thr": args.conf_thr,
            "vh_percentile": args.vh_percentile,
            "vh_min_views": args.vh_min_views,
            "margin_voxels": args.margin_voxels,
            "margin": margin,
            "votes_req": args.votes_req,
            "clip_margin_voxels": args.clip_margin_voxels,
            "clip_margin": clip_margin,
        },
        "sfm_roi_info": roi_info,
        "clip_info": clip_info,
        "per_view": per_view,
        "hull_voxels": n_occ0,
        "voxels_removed_depth": n_depth_removed,
        "voxels_removed_clip_component": n_clip_removed,
        "voxels_final": n_final,
        "pct_removed_total": pct_total,
        "outputs": {
            "hull_mesh": str(args.out_dir / hull_mesh_name),
            "carved_mesh": str(args.out_dir / "vh_mvsformer_carved.ply"),
            "carved_sdf": str(args.out_dir / "sdf_vh_mvsformer_carved.npy"),
            "compare_png": str(args.out_dir / "vh_mvsformer_carve_compare.png"),
        },
        "runtime_sec": time.perf_counter() - t0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
