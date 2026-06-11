#!/usr/bin/env python3
"""Carve an enclosing sphere with cached MVSFormer++ depths.

This is a geometry-only Truck/TnT diagnostic:
  1. initialize occupancy as a sphere in the normalized scene frame,
  2. remove voxels voted empty by MVSFormer++ depths,
  3. save mesh, SDF grid, and a rendered PNG comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from lip_tracer.data import load_blender_views, load_views
from lip_tracer.visual_hull import keep_largest_component

from carve_visual_hull_with_mvsformer_scan24 import (
    _read_pfm,
    depth_vote_view,
    save_sdf_grid,
    voxel_world_from_occ_indices,
)
from _hull_depth_carve_scan24 import (
    default_ref_views,
    occ_to_mesh_world,
    render_mesh,
    save_ply,
    sfm_aabb_clip_mask,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=Path("data/tnt/Truck"))
    ap.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("_diagnostics/mvsformer_truck/depths_1152x640"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("_diagnostics/mvsformer_truck_sphere/sphere_carve_r145_res192"),
    )
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--sphere-radius", default="auto",
                    help="float radius, or 'auto' from sparse-SfM ROI")
    ap.add_argument("--sphere-radius-percentile", type=float, default=99.9)
    ap.add_argument("--sphere-radius-pad", type=float, default=0.02)
    ap.add_argument("--sphere-center", type=float, nargs=3, default=None)
    ap.add_argument(
        "--center-from-sfm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use sparse_sfm_points.txt AABB center when --sphere-center is absent",
    )
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument(
        "--use-depth-mask",
        action="store_true",
        help="debug option: restrict MVSFormer valid pixels with scene masks",
    )
    ap.add_argument("--votes-req", type=int, default=3)
    ap.add_argument("--margin-voxels", type=float, default=3.0)
    ap.add_argument(
        "--depth-cap", type=float, default=None,
        help="drop depth pixels beyond this value before carving (absolute units). "
             "Use 'auto' via --depth-cap-pad to derive max_cam_dist+bound+pad.")
    ap.add_argument(
        "--depth-cap-auto", action="store_true",
        help="auto depth cap = max camera-center distance + bound + depth-cap-pad")
    ap.add_argument("--depth-cap-pad", type=float, default=0.25,
                    help="pad added to the auto depth cap (absolute units)")
    ap.add_argument("--sfm-clip", action="store_true")
    ap.add_argument("--clip-margin-voxels", type=float, default=6.0)
    ap.add_argument("--open-iters", type=int, default=0,
                    help="3D binary opening iterations after depth/ROI carving")
    ap.add_argument("--close-iters", type=int, default=0,
                    help="3D binary closing iterations after depth/ROI carving")
    ap.add_argument("--fill-holes", action="store_true",
                    help="fill enclosed holes in the voxel occupancy")
    ap.add_argument("--chunk-size", type=int, default=500_000)
    ap.add_argument(
        "--largest-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep largest connected component after depth carving",
    )
    ap.add_argument("--blender", action="store_true",
                    help="load NeRF Blender cameras from transforms_train.json")
    return ap.parse_args()


def sfm_points_in_bound(scene: Path, bound: float) -> np.ndarray:
    sfm = scene / "sparse_sfm_points.txt"
    if not sfm.exists():
        return np.empty((0, 3), dtype=np.float32)
    pts = np.loadtxt(sfm, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[np.all(np.abs(pts) <= bound, axis=1)]
    return pts.astype(np.float32)


def sfm_aabb_center(scene: Path, bound: float) -> np.ndarray:
    pts = sfm_points_in_bound(scene, bound)
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32)
    return ((pts.min(axis=0) + pts.max(axis=0)) * 0.5).astype(np.float32)


def sfm_auto_radius(scene: Path, bound: float, center: np.ndarray,
                    percentile: float, pad: float) -> tuple[float, dict]:
    pts = sfm_points_in_bound(scene, bound)
    if len(pts) == 0:
        radius = 0.95 * bound
        return radius, {
            "mode": "fallback_bound",
            "radius": radius,
            "points": 0,
        }
    d = np.linalg.norm(pts.astype(np.float64) - center.astype(np.float64), axis=1)
    core_radius = float(np.percentile(d, percentile))
    radius = min(float(bound), core_radius + float(pad))
    return radius, {
        "mode": "sfm_distance_percentile",
        "points": int(len(pts)),
        "percentile": float(percentile),
        "core_radius": core_radius,
        "pad": float(pad),
        "radius": radius,
    }


def sphere_occ(res: int, bound: float, center: np.ndarray, radius: float) -> np.ndarray:
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    z = lin[:, None, None] - center[2]
    y = lin[None, :, None] - center[1]
    x = lin[None, None, :] - center[0]
    return (x * x + y * y + z * z) <= float(radius * radius)


def load_mvsformer_depths_tnt_nomask(
    scene: Path,
    depth_dir: Path,
    conf_thresh: float,
    use_depth_mask: bool = False,
    blender: bool = False,
    depth_cap: float | None = None,
    depth_cap_auto: bool = False,
    bound: float = 1.5,
    depth_cap_pad: float = 0.25,
) -> dict | None:
    """Load cached TnT MVSFormer++ depths.

    For sphere carving the default deliberately does not use object masks:
    background depths are needed to carve the sphere outside the object.

    depth_cap: if set, pixels whose depth exceeds it are dropped from the carve
    mask. MVSFormer emits runaway-large depths on textureless walls/sky; such a
    pixel votes its whole ray through the bound empty and carves real surface.
    Capping at ~(max camera distance + bound) keeps those rays from eating walls.
    """
    scan_dirs = [d for d in depth_dir.iterdir()
                 if d.is_dir() and (d / "depth_est").exists()]
    scan_root = scan_dirs[0] if scan_dirs else depth_dir
    pfm_files = sorted((scan_root / "depth_est").glob("*.pfm"))
    if not pfm_files:
        print(f"  [geomvs] no .pfm files under {scan_root}")
        return None

    views = load_blender_views(scene, split="train", down=1) if blender else load_views(scene, down=1)
    c2w = views["c2w"].numpy().astype(np.float32)
    K0 = views["K"].numpy().astype(np.float32)
    H0, W0 = int(views["H"]), int(views["W"])
    masks = views["masks"].numpy().astype(bool) if use_depth_mask else None
    if len(pfm_files) > len(c2w):
        raise ValueError(f"more depth maps ({len(pfm_files)}) than views ({len(c2w)})")
    if depth_cap_auto and depth_cap is None:
        max_cam_dist = float(np.linalg.norm(c2w[:, :3, 3], axis=1).max())
        depth_cap = max_cam_dist + bound + depth_cap_pad
        print(f"  [geomvs] auto depth cap = {depth_cap:.3f} "
              f"(max_cam_dist {max_cam_dist:.3f} + bound {bound:.3f} + pad {depth_cap_pad:.3f})")
    if depth_cap is not None:
        print(f"  [geomvs] depth cap = {depth_cap:.3f} (dropping runaway/background pixels)")

    if use_depth_mask:
        from carve_visual_hull_with_mvsformer_scan24 import resize_mask

    depths, valid, Ks = [], [], []
    n_capped = 0
    for i, pfm in enumerate(pfm_files):
        depth = np.asarray(_read_pfm(pfm), dtype=np.float32)
        conf = np.load(scan_root / "confidence" / f"{pfm.stem}.npy")
        if conf.dtype == np.uint8:
            conf = conf.astype(np.float32) / 255.0
        H, W = depth.shape
        K = K0[i].copy()
        K[0, :] *= W / float(W0)
        K[1, :] *= H / float(H0)
        good = (conf > conf_thresh) & (depth > 1e-3)
        if depth_cap is not None:
            far = good & (depth > depth_cap)
            n_capped += int(far.sum())
            good &= depth <= depth_cap
        if masks is not None:
            good &= resize_mask(masks[i], (H, W))
        depths.append(depth)
        valid.append(good)
        Ks.append(K)
        if i % 25 == 0:
            pct = 100 * good.mean()
            zr = depth[good] if good.any() else np.array([0.0], dtype=np.float32)
            mode = "masked" if use_depth_mask else "unmasked"
            print(f"  [geomvs] tnt mvsf {mode} view {i:03d}: valid={pct:.1f}% "
                  f"z=[{float(zr.min()):.3f},{float(zr.max()):.3f}]")

    print(f"  [geomvs] loaded {len(depths)} unmasked MVSFormer++ depth maps from {scan_root}")
    if depth_cap is not None:
        print(f"  [geomvs] depth cap dropped {n_capped} runaway pixels")
    return {
        "depths": depths,
        "valid": valid,
        "K": np.stack(Ks),
        "c2w": c2w[:len(depths)],
        "H": depths[0].shape[0],
        "W": depths[0].shape[1],
    }


def clean_occupancy(
    occ: np.ndarray,
    open_iters: int = 0,
    close_iters: int = 0,
    fill_holes: bool = False,
) -> tuple[np.ndarray, dict]:
    """Optional light morphology for diagnostic init grids."""
    info = {
        "open_iters": int(open_iters),
        "close_iters": int(close_iters),
        "fill_holes": bool(fill_holes),
        "before": int(occ.sum()),
    }
    out = occ
    if open_iters > 0 or close_iters > 0 or fill_holes:
        from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
        structure = np.ones((3, 3, 3), dtype=bool)
        if open_iters > 0:
            out = binary_opening(out, structure=structure, iterations=open_iters)
        if close_iters > 0:
            out = binary_closing(out, structure=structure, iterations=close_iters)
        if fill_holes:
            out = binary_fill_holes(out)
        out = np.asarray(out, dtype=bool)
    info["after"] = int(out.sum())
    info["delta"] = int(out.sum()) - info["before"]
    return out, info


def save_sphere_compare_render(
    out_path: Path,
    verts_s: np.ndarray,
    faces_s: np.ndarray,
    verts_c: np.ndarray,
    faces_c: np.ndarray,
    views_hi: dict,
    pct_removed: float,
    carved_label: str = "sphere carved by MVSFormer++",
    title_suffix: str = "",
) -> None:
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
    imgs_s = render_mesh(verts_s, faces_s, rv, ref_views)
    imgs_c = render_mesh(verts_c, faces_c, rv, ref_views)

    fig, axes = plt.subplots(len(ref_views), 2, figsize=(8, 4 * len(ref_views)), squeeze=False)
    for row, vi in enumerate(ref_views):
        for col, (img, label) in enumerate([
            (imgs_s[row], "enclosing sphere init"),
            (imgs_c[row], carved_label),
        ]):
            axes[row, col].imshow(np.clip(img, 0, 1))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label, fontsize=11)
        axes[row, 0].set_ylabel(f"view {vi}", fontsize=10)
    fig.suptitle(
        f"Truck sphere init carved by MVSFormer++ depths "
        f"{title_suffix}({pct_removed:.1f}% voxels removed)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    center = (
        np.array(args.sphere_center, dtype=np.float32)
        if args.sphere_center is not None
        else sfm_aabb_center(args.scene, args.bound) if args.center_from_sfm
        else np.zeros(3, dtype=np.float32)
    )
    if str(args.sphere_radius).lower() == "auto":
        sphere_radius, radius_info = sfm_auto_radius(
            args.scene,
            args.bound,
            center,
            args.sphere_radius_percentile,
            args.sphere_radius_pad,
        )
    else:
        sphere_radius = float(args.sphere_radius)
        radius_info = {"mode": "manual", "radius": sphere_radius}
    voxel = 2 * args.bound / max(args.res - 1, 1)
    margin = args.margin_voxels * voxel
    clip_margin = args.clip_margin_voxels * voxel

    print(f"scene={args.scene}")
    print(f"res={args.res} bound={args.bound} voxel={voxel:.6f}")
    print(f"sphere center={np.round(center, 6).tolist()} radius={sphere_radius:.6f} "
          f"({radius_info['mode']})")
    print(f"MVSFormer conf_thr={args.conf_thr} margin={margin:.6f} votes_req={args.votes_req}")
    if args.sfm_clip:
        print(f"SfM ROI clip margin={clip_margin:.6f} ({args.clip_margin_voxels:g} voxels)")

    print("\nloading MVSFormer++ depths ...")
    mvs = load_mvsformer_depths_tnt_nomask(
        args.scene,
        args.depth_dir,
        args.conf_thr,
        use_depth_mask=args.use_depth_mask,
        blender=args.blender,
        depth_cap=args.depth_cap,
        depth_cap_auto=args.depth_cap_auto,
        bound=args.bound,
        depth_cap_pad=args.depth_cap_pad,
    )
    if mvs is None:
        raise RuntimeError("failed to load MVSFormer++ depths")

    print("\ninitializing enclosing sphere ...")
    occ = sphere_occ(args.res, args.bound, center, sphere_radius)
    n_occ0 = int(occ.sum())
    print(f"  sphere voxels: {n_occ0} / {occ.size}")
    verts_s, faces_s = occ_to_mesh_world(occ, args.bound)
    save_ply(verts_s, faces_s, args.out_dir / "sphere_init.ply")
    save_sdf_grid(occ, args.bound, args.out_dir / "sdf_sphere_init.npy")

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
        print(f"  view {vi:03d}: valid={stats['valid_depth_px']:8d} "
              f"voted_empty={stats['voxels_voted_empty']:8d}")

    remove = (votes >= args.votes_req) & occ
    carved = occ & ~remove
    n_depth_removed = int(remove.sum())
    print(f"\nremoved by MVSFormer depth votes: {n_depth_removed} / {n_occ0} "
          f"({100 * n_depth_removed / max(n_occ0, 1):.2f}%)")

    n_clip_removed = 0
    clip_info = None
    if args.sfm_clip:
        print("\napplying sparse-SfM AABB clip after MVSFormer carving ...")
        keep, clip_info = sfm_aabb_clip_mask(args.scene, carved, args.bound, clip_margin)
        before = int(carved.sum())
        carved = carved & keep
        n_clip_removed = before - int(carved.sum())
        print(f"removed by SFM ROI clip: {n_clip_removed}")

    print("\napplying voxel cleanup ...")
    carved, cleanup_info = clean_occupancy(
        carved,
        open_iters=args.open_iters,
        close_iters=args.close_iters,
        fill_holes=args.fill_holes,
    )
    print(f"  cleanup delta: {cleanup_info['delta']} voxels "
          f"({cleanup_info['before']} -> {cleanup_info['after']})")

    n_component_removed = 0
    if args.largest_component:
        before = int(carved.sum())
        carved = keep_largest_component(carved)
        n_component_removed = before - int(carved.sum())
        print(f"removed by largest-component cleanup: {n_component_removed}")

    n_final = int(carved.sum())
    pct_total = 100 * (n_occ0 - n_final) / max(n_occ0, 1)
    print(f"total removed: {n_occ0 - n_final} / {n_occ0} ({pct_total:.2f}%)")

    verts_c, faces_c = occ_to_mesh_world(carved, args.bound)
    save_ply(verts_c, faces_c, args.out_dir / "sphere_mvsformer_carved.ply")
    save_sdf_grid(carved, args.bound, args.out_dir / "sdf_sphere_mvsformer_carved.npy")

    print("\nrendering Phong comparison ...")
    views_hi = (
        load_blender_views(args.scene, split="train", down=1)
        if args.blender else load_views(args.scene, down=1)
    )
    save_sphere_compare_render(
        args.out_dir / "sphere_mvsformer_carve_compare.png",
        verts_s,
        faces_s,
        verts_c,
        faces_c,
        views_hi,
        pct_total,
        "sphere carved by MVSFormer++ + SFM ROI" if args.sfm_clip else "sphere carved by MVSFormer++",
        "+ SFM ROI " if args.sfm_clip else "",
    )

    summary = {
        "scene": str(args.scene),
        "depth_dir": str(args.depth_dir),
        "settings": {
            "res": args.res,
            "bound": args.bound,
            "sphere_center": center.tolist(),
            "sphere_radius": sphere_radius,
            "sphere_radius_info": radius_info,
            "conf_thr": args.conf_thr,
            "use_depth_mask": args.use_depth_mask,
            "margin_voxels": args.margin_voxels,
            "margin": margin,
            "votes_req": args.votes_req,
            "sfm_clip": args.sfm_clip,
            "clip_margin_voxels": args.clip_margin_voxels,
            "clip_margin": clip_margin,
            "open_iters": args.open_iters,
            "close_iters": args.close_iters,
            "fill_holes": args.fill_holes,
            "largest_component": args.largest_component,
        },
        "clip_info": clip_info,
        "cleanup_info": cleanup_info,
        "per_view": per_view,
        "sphere_voxels": n_occ0,
        "voxels_removed_depth": n_depth_removed,
        "voxels_removed_sfm_clip": n_clip_removed,
        "voxels_removed_component": n_component_removed,
        "voxels_final": n_final,
        "pct_removed_total": pct_total,
        "outputs": {
            "sphere_mesh": str(args.out_dir / "sphere_init.ply"),
            "sphere_sdf": str(args.out_dir / "sdf_sphere_init.npy"),
            "carved_mesh": str(args.out_dir / "sphere_mvsformer_carved.ply"),
            "carved_sdf": str(args.out_dir / "sdf_sphere_mvsformer_carved.npy"),
            "compare_png": str(args.out_dir / "sphere_mvsformer_carve_compare.png"),
        },
        "runtime_sec": time.perf_counter() - t0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
