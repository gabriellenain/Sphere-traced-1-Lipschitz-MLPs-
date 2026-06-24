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
    depth_vote_all_torch,
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
    ap.add_argument(
        "--method", choices=["carve", "tsdf"], default="carve",
        help="occupancy estimator. 'carve' = hard depth-vote space carving from a "
             "solid sphere (default). 'tsdf' = confidence-weighted, front-truncated "
             "signed fusion: per-voxel weighted mean of clamped (d_meas - z)/trunc "
             "over all confident views, occupied where the mean is negative. Soft and "
             "noise-robust (wrong low-confidence depths are out-weighted, not "
             "hole-punching); no enclosing sphere needed.")
    ap.add_argument("--tsdf-trunc-voxels", type=float, default=4.0,
                    help="TSDF truncation distance in voxels (front ramp width)")
    ap.add_argument("--tsdf-shell", action="store_true",
                    help="keep only a thin SURFACE shell (voxels within +-trunc of the "
                         "measured depth) instead of filling everything behind the "
                         "surface solid. REQUIRED for open/interior scenes (ETH3D): "
                         "cameras cluster centrally, so 'fill behind' degenerates into a "
                         "solid frustum/bowtie. The shell is the actual observed surface.")
    ap.add_argument("--tsdf-shell-lo", type=float, default=-0.95,
                    help="shell mode: drop voxels with mean signed value <= this "
                         "(deep-interior, far behind the surface). -0.95 ~= one trunc "
                         "band behind the surface.")
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
    ap.add_argument(
        "--geo-consistency", action="store_true",
        help="cross-view geometric consistency filter on MVSFormer depths before "
             "carving: keep a pixel only if its depth reprojects consistently through "
             "enough neighbour views. Drops the too-near textureless-panel depths "
             "that otherwise carve real surface.")
    ap.add_argument("--geo-n-views", type=int, default=8,
                    help="number of nearest-camera neighbour views to check")
    ap.add_argument("--geo-n-consistent", type=int, default=2,
                    help="min consistent neighbours required to keep a pixel")
    ap.add_argument("--geo-tau-pix", type=float, default=1.0,
                    help="max reprojection error (pixels)")
    ap.add_argument("--geo-tau-depth", type=float, default=0.01,
                    help="max relative depth error |dz|/d")
    ap.add_argument(
        "--drop-seethrough", action="store_true",
        help="drop per-view depth pixels that are local FAR outliers (depth > factor x "
             "large-window local median). Glass (MVS sees through it, consistently, so "
             "geo-consistency keeps it) and sky gaps inside thin structures are far-"
             "outlier blobs against the surrounding correct surface; their votes are "
             "what carve init holes. Mask-free.")
    ap.add_argument("--seethrough-win", type=int, default=31,
                    help="median window (on the downsampled map)")
    ap.add_argument("--seethrough-factor", type=float, default=1.3,
                    help="drop pixel if depth > factor x local median")
    ap.add_argument("--seethrough-down", type=int, default=8,
                    help="downsample for the local median (win x down = full-res context)")
    ap.add_argument(
        "--protect-sfm-radius", type=float, default=None,
        help="voxels within this distance (world units) of a sparse-SfM point cannot "
             "be carved by depth votes: SfM points are photometrically verified "
             "surface, so depth votes against them are wrong. Mask-free; keep "
             "<= --sfm-roi-dist.")
    ap.add_argument("--sfm-clip", action="store_true")
    ap.add_argument("--clip-margin-voxels", type=float, default=6.0)
    ap.add_argument("--depth-label", type=str, default="MVSFormer++",
                    help="name of the depth source, used in the compare-plot title, "
                         "carved-panel label, and console messages (e.g. 'ACMMP'). "
                         "Output filenames are left unchanged for downstream tooling.")
    ap.add_argument(
        "--sfm-roi-dist", type=float, default=None,
        help="after the AABB clip, drop voxels farther than this (world units) from "
             "the nearest sparse-SfM point. The AABB cannot remove a blob floating in "
             "air inside the scene's bounding box; this distance ROI does, since empty "
             "air has no reconstructed points nearby. Keep generous so textureless "
             "panels (sparse points) survive.")
    ap.add_argument("--open-iters", type=int, default=0,
                    help="3D binary opening iterations after depth/ROI carving")
    ap.add_argument("--close-iters", type=int, default=0,
                    help="3D binary closing iterations after depth/ROI carving")
    ap.add_argument("--fill-holes", action="store_true",
                    help="fill enclosed holes in the voxel occupancy")
    ap.add_argument("--chunk-size", type=int, default=500_000)
    ap.add_argument("--device", type=str, default="cpu",
                    help="'cpu' (NumPy depth-vote loop) or 'cuda' (torch). The "
                         "per-view projection+scatter is the carve bottleneck.")
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
    # ETH3D: ACMMP depths are metric cam-z; load_views normalized the cameras by
    # `scale` (eth3d_norm.json). Divide depths by the same scale so back-projected
    # points land in the normalized [-bound,bound] carve grid.
    norm_path = scene / "eth3d_norm.json"
    depth_norm_scale = (json.loads(norm_path.read_text())["scale"]
                        if norm_path.exists() else 1.0)
    if depth_norm_scale != 1.0:
        print(f"  [geomvs] ETH3D normalized: dividing metric depths by scale={depth_norm_scale:.4f}")
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

    depths, valid, Ks, confs = [], [], [], []
    n_capped = 0
    for i, pfm in enumerate(pfm_files):
        depth = np.asarray(_read_pfm(pfm), dtype=np.float32)
        if depth_norm_scale != 1.0:
            depth = depth / np.float32(depth_norm_scale)
        conf = np.load(scan_root / "confidence" / f"{pfm.stem}.npy")
        if conf.dtype == np.uint8:
            conf = conf.astype(np.float32) / 255.0
        conf = conf.astype(np.float32)
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
        confs.append(conf)
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
        "conf": confs,
        "K": np.stack(Ks),
        "c2w": c2w[:len(depths)],
        "H": depths[0].shape[0],
        "W": depths[0].shape[1],
    }


def drop_seethrough_pixels(
    depths: list[np.ndarray],
    valid: list[np.ndarray],
    win: int = 31,
    factor: float = 1.3,
    down: int = 8,
) -> list[np.ndarray]:
    """Drop local FAR-outlier depth pixels (see-through glass, sky gaps in thin
    structures) before carving.

    The local median is computed on a ``down``-sampled map with a ``win`` window,
    i.e. a (win*down)-pixel full-resolution context — large enough that a glass /
    between-bars blob is the minority inside its window, so the median locks onto
    the surrounding correct surface and the blob exceeds ``factor`` x median.
    Smooth far surfaces (ground, walls) match their own local median and survive.
    Conservative bias: invalid pixels are filled with the global median, which can
    only raise the local median and suppress fewer pixels.
    """
    from scipy.ndimage import median_filter

    out = []
    n_drop_total = n_valid_total = 0
    for i, (depth, v) in enumerate(zip(depths, valid)):
        if not v.any():
            out.append(v)
            continue
        d = depth.astype(np.float32, copy=True)
        d[~v] = float(np.median(depth[v]))
        ds = d[::down, ::down]
        med = median_filter(ds, size=win, mode="nearest")
        med_full = np.repeat(np.repeat(med, down, 0), down, 1)[: d.shape[0], : d.shape[1]]
        far = v & (depth > factor * med_full)
        out.append(v & ~far)
        n_drop_total += int(far.sum())
        n_valid_total += int(v.sum())
        if i % 25 == 0:
            print(f"  [seethrough] view {i:03d}: dropped {int(far.sum()):7d} "
                  f"/ {int(v.sum()):8d} far-outlier px")
    print(f"  [seethrough] total dropped {n_drop_total} / {n_valid_total} "
          f"({100 * n_drop_total / max(n_valid_total, 1):.1f}%) "
          f"win={win} factor={factor} down={down}")
    return out


def geometric_consistency_filter(
    depths: list[np.ndarray],
    valid: list[np.ndarray],
    K: np.ndarray,
    c2w: np.ndarray,
    n_views: int = 8,
    n_consistent: int = 2,
    tau_pix: float = 1.0,
    tau_depth: float = 0.01,
) -> list[np.ndarray]:
    """Cross-view geometric consistency check (MVSNet/COLMAP fusion style).

    For each reference pixel with depth d_r: back-project to world, project into a
    neighbour view, sample that view's depth, back-project the neighbour pixel to
    world, reproject into the reference view, and require small reprojection error
    (pixels) AND small relative depth error. A pixel is kept only if at least
    ``n_consistent`` of its ``n_views`` nearest-camera neighbours agree. Wrong
    "too-near" depths on textureless panels fail this test and are dropped, so they
    no longer free-carve real surface; texture-consistent background depths survive.
    """
    V = len(depths)
    centers = c2w[:, :3, 3].astype(np.float64)
    out: list[np.ndarray] = []
    total_before = total_after = 0
    for r in range(V):
        Hr, Wr = depths[r].shape
        Kr = K[r].astype(np.float64)
        Rr = c2w[r, :3, :3].astype(np.float64)
        tr = c2w[r, :3, 3].astype(np.float64)
        dr = depths[r].astype(np.float64)
        vr = valid[r]

        ys, xs = np.meshgrid(np.arange(Hr), np.arange(Wr), indexing="ij")
        xs = xs.astype(np.float64); ys = ys.astype(np.float64)
        # ref pixel + depth -> world
        xc = (xs - Kr[0, 2]) / Kr[0, 0] * dr
        yc = (ys - Kr[1, 2]) / Kr[1, 1] * dr
        Xw = np.stack([xc, yc, dr], axis=-1) @ Rr.T + tr  # (Hr,Wr,3)

        d2 = np.sum((centers - centers[r]) ** 2, axis=1)
        d2[r] = np.inf
        nbrs = np.argsort(d2)[:n_views]

        cons = np.zeros((Hr, Wr), dtype=np.int32)
        for n in nbrs:
            Hn, Wn = depths[n].shape
            Kn = K[n].astype(np.float64)
            Rn = c2w[n, :3, :3].astype(np.float64)
            tn = c2w[n, :3, 3].astype(np.float64)
            dn = depths[n].astype(np.float64)
            vn = valid[n]

            # world -> neighbour cam  (R_n^T (Xw - t_n) == (Xw - t_n) @ R_n)
            Xnc = (Xw - tn) @ Rn
            zn = Xnc[..., 2]
            front = zn > 1e-6
            zsafe = np.where(front, zn, 1.0)
            un = Xnc[..., 0] / zsafe * Kn[0, 0] + Kn[0, 2]
            vn_ = Xnc[..., 1] / zsafe * Kn[1, 1] + Kn[1, 2]
            ui = np.rint(un).astype(np.int64)
            vi = np.rint(vn_).astype(np.int64)
            inb = front & (ui >= 0) & (ui < Wn) & (vi >= 0) & (vi < Hn)
            ii = np.clip(vi, 0, Hn - 1); jj = np.clip(ui, 0, Wn - 1)
            d_n = dn[ii, jj]
            good_n = inb & vn[ii, jj] & (d_n > 1e-6)

            # neighbour pixel + sampled depth -> world -> back to ref
            xb = (jj.astype(np.float64) - Kn[0, 2]) / Kn[0, 0] * d_n
            yb = (ii.astype(np.float64) - Kn[1, 2]) / Kn[1, 1] * d_n
            Xw2 = np.stack([xb, yb, d_n], axis=-1) @ Rn.T + tn
            Xrc = (Xw2 - tr) @ Rr
            zr2 = Xrc[..., 2]
            frontr = zr2 > 1e-6
            zr2s = np.where(frontr, zr2, 1.0)
            ur2 = Xrc[..., 0] / zr2s * Kr[0, 0] + Kr[0, 2]
            vr2 = Xrc[..., 1] / zr2s * Kr[1, 1] + Kr[1, 2]
            reproj = np.sqrt((ur2 - xs) ** 2 + (vr2 - ys) ** 2)
            ddepth = np.abs(zr2 - dr) / np.clip(dr, 1e-6, None)
            cons += (good_n & frontr & (reproj < tau_pix) & (ddepth < tau_depth))

        keep = vr & (cons >= n_consistent)
        out.append(keep)
        total_before += int(vr.sum())
        total_after += int(keep.sum())
        if r % 25 == 0:
            print(f"  [geo] view {r:03d}: valid {int(vr.sum()):8d} -> "
                  f"{int(keep.sum()):8d} ({100*keep.sum()/max(vr.sum(),1):.1f}%)")
    print(f"  [geo] consistency filter: {total_before} -> {total_after} valid px "
          f"({100*total_after/max(total_before,1):.1f}% kept), "
          f"n_views={n_views} n_consistent={n_consistent} "
          f"tau_pix={tau_pix} tau_depth={tau_depth}")
    return out


def geometric_consistency_filter_torch(
    depths: list[np.ndarray],
    valid: list[np.ndarray],
    K: np.ndarray,
    c2w: np.ndarray,
    n_views: int = 8,
    n_consistent: int = 2,
    tau_pix: float = 1.0,
    tau_depth: float = 0.01,
    device: str = "cuda",
) -> list[np.ndarray]:
    """GPU port of geometric_consistency_filter (identical semantics)."""
    import torch

    dev = torch.device(device)
    V = len(depths)
    centers = c2w[:, :3, 3].astype(np.float64)
    dt = [torch.as_tensor(np.asarray(d, np.float32), device=dev) for d in depths]
    vt = [torch.as_tensor(np.asarray(v, np.bool_), device=dev) for v in valid]
    Kt = [torch.as_tensor(K[i].astype(np.float32), device=dev) for i in range(V)]
    Rt = [torch.as_tensor(c2w[i, :3, :3].astype(np.float32), device=dev) for i in range(V)]
    tt = [torch.as_tensor(c2w[i, :3, 3].astype(np.float32), device=dev) for i in range(V)]

    out, total_b, total_a = [], 0, 0
    for r in range(V):
        Hr, Wr = dt[r].shape
        Kr, Rr, tr, dr, vr = Kt[r], Rt[r], tt[r], dt[r], vt[r]
        ys, xs = torch.meshgrid(torch.arange(Hr, device=dev, dtype=torch.float32),
                                torch.arange(Wr, device=dev, dtype=torch.float32),
                                indexing="ij")
        xc = (xs - Kr[0, 2]) / Kr[0, 0] * dr
        yc = (ys - Kr[1, 2]) / Kr[1, 1] * dr
        Xw = torch.stack([xc, yc, dr], -1) @ Rr.T + tr           # (Hr,Wr,3)

        d2 = ((centers - centers[r]) ** 2).sum(1); d2[r] = np.inf
        nbrs = np.argsort(d2)[:n_views]
        cons = torch.zeros((Hr, Wr), dtype=torch.int32, device=dev)
        for n in nbrs:
            Hn, Wn = dt[n].shape
            Kn, Rn, tn, dn, vn = Kt[n], Rt[n], tt[n], dt[n], vt[n]
            Xnc = (Xw - tn) @ Rn
            zn = Xnc[..., 2]; front = zn > 1e-6
            zsafe = torch.where(front, zn, torch.ones_like(zn))
            un = Xnc[..., 0] / zsafe * Kn[0, 0] + Kn[0, 2]
            vv = Xnc[..., 1] / zsafe * Kn[1, 1] + Kn[1, 2]
            ui = torch.round(un).long(); vi = torch.round(vv).long()
            inb = front & (ui >= 0) & (ui < Wn) & (vi >= 0) & (vi < Hn)
            ii = vi.clamp(0, Hn - 1); jj = ui.clamp(0, Wn - 1)
            d_n = dn[ii, jj]
            good_n = inb & vn[ii, jj] & (d_n > 1e-6)
            xb = (jj.float() - Kn[0, 2]) / Kn[0, 0] * d_n
            yb = (ii.float() - Kn[1, 2]) / Kn[1, 1] * d_n
            Xw2 = torch.stack([xb, yb, d_n], -1) @ Rn.T + tn
            Xrc = (Xw2 - tr) @ Rr
            zr2 = Xrc[..., 2]; frontr = zr2 > 1e-6
            zr2s = torch.where(frontr, zr2, torch.ones_like(zr2))
            ur2 = Xrc[..., 0] / zr2s * Kr[0, 0] + Kr[0, 2]
            vr2 = Xrc[..., 1] / zr2s * Kr[1, 1] + Kr[1, 2]
            reproj = torch.sqrt((ur2 - xs) ** 2 + (vr2 - ys) ** 2)
            ddepth = torch.abs(zr2 - dr) / dr.clamp(min=1e-6)
            cons += (good_n & frontr & (reproj < tau_pix) & (ddepth < tau_depth)).int()
        keep = vr & (cons >= n_consistent)
        out.append(keep.cpu().numpy())
        total_b += int(vr.sum()); total_a += int(keep.sum())
    print(f"  [geo-gpu] consistency filter: {total_b} -> {total_a} valid px "
          f"({100*total_a/max(total_b,1):.1f}% kept), n_views={n_views} "
          f"n_consistent={n_consistent} tau_pix={tau_pix} tau_depth={tau_depth}")
    return out


def fuse_tsdf_occupancy_torch(
    mvs: dict, res: int, bound: float, trunc: float, device: str = "cuda",
    shell: bool = False, shell_lo: float = -0.95,
    chunk: int = 16_000_000,
) -> tuple[np.ndarray, dict]:
    """GPU port of fuse_tsdf_occupancy (identical semantics).

    Chunked over voxels so peak GPU memory stays bounded: the full-grid
    accumulators (Wsum, Vsum) are kept once, but each view projects the voxel
    cloud a chunk at a time, so a res^3 = 512^3 grid fits on a ~10 GB GPU."""
    import torch

    dev = torch.device(device)
    lin = torch.linspace(-bound, bound, res, device=dev)
    ZZ, YY, XX = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], 1)  # (N,3)
    del ZZ, YY, XX
    N = pts.shape[0]
    Wsum = torch.zeros(N, device=dev)
    Vsum = torch.zeros(N, device=dev)

    depths, valids, confs = mvs["depths"], mvs["valid"], mvs["conf"]
    Ks, c2ws = mvs["K"], mvs["c2w"]
    V = len(depths)
    for vi in range(V):
        depth = torch.as_tensor(np.asarray(depths[vi], np.float32), device=dev)
        valid = torch.as_tensor(np.asarray(valids[vi], np.bool_), device=dev)
        conf = torch.as_tensor(np.asarray(confs[vi], np.float32), device=dev) * valid
        h, w = depth.shape
        K = torch.as_tensor(Ks[vi].astype(np.float32), device=dev)
        R = torch.as_tensor(c2ws[vi][:3, :3].astype(np.float32), device=dev)
        c = torch.as_tensor(c2ws[vi][:3, 3].astype(np.float32), device=dev)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            cam = (pts[s:e] - c) @ R
            z = cam[:, 2]; front = z > 1e-5
            zsafe = torch.where(front, z, torch.ones_like(z))
            u = cam[:, 0] / zsafe * K[0, 0] + K[0, 2]
            v = cam[:, 1] / zsafe * K[1, 1] + K[1, 2]
            ui = torch.round(u).long(); vj = torch.round(v).long()
            inb = front & (ui >= 0) & (ui < w) & (vj >= 0) & (vj < h)
            if not bool(inb.any()):
                continue
            uii, vjj = ui[inb], vj[inb]
            wgt = conf[vjj, uii]
            d = depth[vjj, uii]
            obs = wgt > 0
            if not bool(obs.any()):
                continue
            sample = torch.clamp((d - z[inb]) / trunc, -1.0, 1.0)
            sel = (torch.arange(e - s, device=dev)[inb] + s)[obs]
            Wsum.index_add_(0, sel, wgt[obs])
            Vsum.index_add_(0, sel, wgt[obs] * sample[obs])

    observed = Wsum > 0
    tsdf = torch.ones(N, device=dev)
    tsdf[observed] = Vsum[observed] / Wsum[observed]
    if shell:
        occ_t = observed & (tsdf < 0.0) & (tsdf > shell_lo)
    else:
        occ_t = tsdf < 0.0
    occ = occ_t.reshape(res, res, res).cpu().numpy()
    info = {
        "trunc": float(trunc),
        "shell": bool(shell),
        "observed_voxels": int(observed.sum().item()),
        "occupied_voxels": int(occ.sum()),
        "total_voxels": int(N),
    }
    print(f"  [tsdf-gpu] observed {info['observed_voxels']}/{N} "
          f"({100*info['observed_voxels']/N:.1f}%), occupied {info['occupied_voxels']} "
          f"({100*info['occupied_voxels']/N:.2f}%)")
    return occ, info


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
    depth_label: str = "MVSFormer++",
    scene_label: str = "scene",
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
        f"{scene_label} sphere init carved by {depth_label} depths "
        f"{title_suffix}({pct_removed:.1f}% voxels removed)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _tsdf_to_occ(tsdf, observed, shell, shell_lo):
    """tsdf<0 = solid-behind fill; shell = thin band -shell_lo<tsdf<0 (surface only)."""
    if shell:
        return observed & (tsdf < 0.0) & (tsdf > shell_lo)
    return tsdf < 0.0


def fuse_tsdf_occupancy(
    mvs: dict,
    res: int,
    bound: float,
    trunc: float,
    chunk_size: int,
    shell: bool = False,
    shell_lo: float = -0.95,
) -> tuple[np.ndarray, dict]:
    """Confidence-weighted, front-truncated signed fusion -> solid occupancy.

    For every grid voxel p and every confident view, the signed sample is
    ``s = clip((d_meas - z_voxel) / trunc, -1, +1)`` where ``z_voxel`` is the
    voxel's camera-space depth and ``d_meas`` the measured depth at its pixel.
    Front of surface (z < d) -> positive (outside); behind (z > d) -> negative
    (inside); the ramp is metric within +-trunc. There is NO back-truncation
    skip, so occluded interior voxels accumulate -1 and the object fills solid.
    Samples are averaged with per-pixel confidence as weight, so a single
    low-confidence wrong depth (textureless wall / sky) is out-weighted by the
    good views instead of hole-punching. A voxel is occupied where the
    confidence-weighted mean signed value is < 0; unobserved voxels (no view)
    default to outside.

    Returns (occupancy[z,y,x] bool, info).
    """
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    # grid axes are (z, y, x) to match sphere_occ / save_sdf_grid
    ZZ, YY, XX = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)  # world xyz
    del ZZ, YY, XX
    N = pts.shape[0]
    Wsum = np.zeros(N, dtype=np.float32)   # sum of confidence weights
    Vsum = np.zeros(N, dtype=np.float32)   # sum of weight * signed sample

    depths, valids, confs = mvs["depths"], mvs["valid"], mvs["conf"]
    Ks, c2ws = mvs["K"], mvs["c2w"]
    V = len(depths)
    for vi in range(V):
        depth = np.asarray(depths[vi], dtype=np.float32)
        valid = np.asarray(valids[vi], dtype=bool)
        conf = np.asarray(confs[vi], dtype=np.float32) * valid  # weight 0 where invalid
        h, w = depth.shape
        K = Ks[vi].astype(np.float64)
        R = c2ws[vi][:3, :3].astype(np.float64)
        c = c2ws[vi][:3, 3].astype(np.float64)
        n_upd = 0
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            cam = (pts[s:e].astype(np.float64) - c[None]) @ R
            z = cam[:, 2]
            front = z > 1e-5
            zsafe = np.where(front, z, 1.0)
            u = cam[:, 0] / zsafe * K[0, 0] + K[0, 2]
            v = cam[:, 1] / zsafe * K[1, 1] + K[1, 2]
            ui = np.rint(u).astype(np.int64)
            vj = np.rint(v).astype(np.int64)
            inb = front & (ui >= 0) & (ui < w) & (vj >= 0) & (vj < h)
            if not inb.any():
                continue
            uii, vjj = ui[inb], vj[inb]
            wgt = conf[vjj, uii]
            d = depth[vjj, uii].astype(np.float64)
            obs = wgt > 0
            if not obs.any():
                continue
            sample = np.clip((d - z[inb]) / trunc, -1.0, 1.0).astype(np.float32)
            # scatter-add into the flat accumulators (global voxel indices = s + local)
            sel = s + np.flatnonzero(inb)[obs]
            np.add.at(Wsum, sel, wgt[obs])
            np.add.at(Vsum, sel, wgt[obs] * sample[obs])
            n_upd += int(obs.sum())
        if vi % 25 == 0:
            print(f"  [tsdf] view {vi:03d}: updated {n_upd} voxel-samples")

    observed = Wsum > 0
    tsdf = np.full(N, 1.0, dtype=np.float32)        # unobserved -> outside
    tsdf[observed] = Vsum[observed] / Wsum[observed]
    occ = _tsdf_to_occ(tsdf, observed, shell, shell_lo).reshape(res, res, res)
    info = {
        "trunc": float(trunc),
        "shell": bool(shell),
        "observed_voxels": int(observed.sum()),
        "occupied_voxels": int(occ.sum()),
        "total_voxels": int(N),
    }
    print(f"  [tsdf] observed {info['observed_voxels']}/{N} voxels "
          f"({100*info['observed_voxels']/N:.1f}%), occupied {info['occupied_voxels']} "
          f"({100*info['occupied_voxels']/N:.2f}%)")
    return occ, info


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
    print(f"{args.depth_label} conf_thr={args.conf_thr} margin={margin:.6f} votes_req={args.votes_req}")
    if args.sfm_clip:
        print(f"SfM ROI clip margin={clip_margin:.6f} ({args.clip_margin_voxels:g} voxels)")

    print(f"\nloading {args.depth_label} depths ...")
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

    if args.drop_seethrough:
        print("\ndropping see-through / far-outlier depth pixels ...")
        mvs["valid"] = drop_seethrough_pixels(
            mvs["depths"], mvs["valid"],
            win=args.seethrough_win,
            factor=args.seethrough_factor,
            down=args.seethrough_down,
        )

    if args.geo_consistency:
        print(f"\napplying cross-view geometric consistency filter (device={args.device}) ...")
        geo_fn = (geometric_consistency_filter_torch if args.device != "cpu"
                  else geometric_consistency_filter)
        geo_kw = {"device": args.device} if args.device != "cpu" else {}
        mvs["valid"] = geo_fn(
            mvs["depths"], mvs["valid"], mvs["K"], mvs["c2w"],
            n_views=args.geo_n_views,
            n_consistent=args.geo_n_consistent,
            tau_pix=args.geo_tau_pix,
            tau_depth=args.geo_tau_depth,
            **geo_kw,
        )

    print("\ninitializing enclosing sphere ...")
    occ = sphere_occ(args.res, args.bound, center, sphere_radius)
    n_occ0 = int(occ.sum())
    print(f"  sphere voxels: {n_occ0} / {occ.size}")
    verts_s, faces_s = occ_to_mesh_world(occ, args.bound)
    save_ply(verts_s, faces_s, args.out_dir / "sphere_init.ply")
    save_sdf_grid(occ, args.bound, args.out_dir / "sdf_sphere_init.npy")

    per_view = []
    n_sfm_protected = 0
    if args.method == "tsdf":
        tsdf_info = None
        print("\nconfidence-weighted TSDF fusion ...")
        trunc = args.tsdf_trunc_voxels * voxel
        print(f"  trunc = {trunc:.6f} ({args.tsdf_trunc_voxels:g} voxels)  device={args.device}")
        if args.device != "cpu":
            carved, tsdf_info = fuse_tsdf_occupancy_torch(
                mvs, args.res, args.bound, trunc, args.device,
                shell=args.tsdf_shell, shell_lo=args.tsdf_shell_lo)
        else:
            carved, tsdf_info = fuse_tsdf_occupancy(
                mvs, args.res, args.bound, trunc, args.chunk_size,
                shell=args.tsdf_shell, shell_lo=args.tsdf_shell_lo)
        n_depth_removed = n_occ0 - int(carved.sum())
    else:
        tsdf_info = None
        print(f"\n{args.depth_label} depth-space free carving ...")
        occ_idx = np.argwhere(occ)
        occ_pts = voxel_world_from_occ_indices(occ_idx, args.bound, args.res)
        print(f"  device={args.device}  occupied voxels to vote: {len(occ_pts)}")
        if args.device != "cpu":
            votes, per_view = depth_vote_all_torch(
                occ_idx, occ_pts, mvs["depths"], mvs["valid"],
                mvs["K"], mvs["c2w"], args.res, margin, args.device)
        else:
            votes = np.zeros_like(occ, dtype=np.uint16)
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
        if args.protect_sfm_radius is not None:
            print(f"\nprotecting voxels within {args.protect_sfm_radius} of SfM points ...")
            from scipy.spatial import cKDTree
            pts = sfm_points_in_bound(args.scene, args.bound)
            if len(pts) == 0:
                print("  [protect-sfm] no in-bound SfM points; skipping")
            else:
                tree = cKDTree(pts.astype(np.float64))
                dist, _ = tree.query(occ_pts.astype(np.float64), k=1, workers=-1)
                near = dist <= float(args.protect_sfm_radius)
                protect = np.zeros_like(occ, dtype=bool)
                nidx = occ_idx[near]
                protect[nidx[:, 0], nidx[:, 1], nidx[:, 2]] = True
                n_sfm_protected = int((remove & protect).sum())
                remove &= ~protect
                print(f"  [protect-sfm] rescued {n_sfm_protected} voted-empty voxels "
                      f"near {len(pts)} SfM points")
        carved = occ & ~remove
        n_depth_removed = int(remove.sum())
        print(f"\nremoved by {args.depth_label} depth votes: {n_depth_removed} / {n_occ0} "
              f"({100 * n_depth_removed / max(n_occ0, 1):.2f}%)")

    n_clip_removed = 0
    clip_info = None
    if args.sfm_clip:
        print(f"\napplying sparse-SfM AABB clip after {args.depth_label} carving ...")
        keep, clip_info = sfm_aabb_clip_mask(args.scene, carved, args.bound, clip_margin)
        before = int(carved.sum())
        carved = carved & keep
        n_clip_removed = before - int(carved.sum())
        print(f"removed by SFM ROI clip: {n_clip_removed}")

    n_roi_dist_removed = 0
    if args.sfm_roi_dist is not None:
        print(f"\napplying SfM distance ROI (drop voxels > {args.sfm_roi_dist} "
              f"from nearest SfM point) ...")
        from scipy.spatial import cKDTree
        pts = sfm_points_in_bound(args.scene, args.bound)
        if len(pts) == 0:
            print("  [roi-dist] no in-bound SfM points; skipping distance ROI")
        else:
            tree = cKDTree(pts.astype(np.float64))
            cidx = np.argwhere(carved)
            cpts = voxel_world_from_occ_indices(cidx, args.bound, args.res)
            dist, _ = tree.query(cpts.astype(np.float64), k=1, workers=-1)
            far = dist > float(args.sfm_roi_dist)
            before = int(carved.sum())
            drop = cidx[far]
            carved[drop[:, 0], drop[:, 1], drop[:, 2]] = False
            n_roi_dist_removed = before - int(carved.sum())
            print(f"removed by SfM distance ROI: {n_roi_dist_removed}")

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
        f"sphere carved by {args.depth_label} + SFM ROI" if args.sfm_clip else f"sphere carved by {args.depth_label}",
        "+ SFM ROI " if args.sfm_clip else "",
        depth_label=args.depth_label,
        scene_label=args.scene.name,
    )

    summary = {
        "scene": str(args.scene),
        "depth_dir": str(args.depth_dir),
        "method": args.method,
        "tsdf_info": tsdf_info,
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
        "voxels_removed_sfm_roi_dist": n_roi_dist_removed,
        "sfm_roi_dist": args.sfm_roi_dist,
        "voxels_rescued_sfm_protect": n_sfm_protected,
        "protect_sfm_radius": args.protect_sfm_radius,
        "drop_seethrough": bool(args.drop_seethrough),
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
