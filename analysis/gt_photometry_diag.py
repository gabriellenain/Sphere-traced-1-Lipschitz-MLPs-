#!/usr/bin/env python3
"""GT-surface NCC photometry diagnostic for DTU.

This probes whether the training PMVS/NCC photometric term has a minimum near
the official DTU reference surface.  The NCC itself is not reimplemented: every
score is produced by lip_tracer.loss.pmvs_ncc_loss with the same oriented 3D
patch construction used during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Title prefix for the figures; set in main() (e.g. "TnT Ignatius") so the
# non-DTU runs are not mislabelled. Falls back to the DTU scan tag.
SCENE_TITLE: str | None = None


def _scene_tag(scan_id) -> str:
    return SCENE_TITLE if SCENE_TITLE else f"DTU scan{scan_id}"


def _obs_inbound(pts: np.ndarray, obs: np.ndarray, bb: np.ndarray,
                 res: float) -> np.ndarray:
    in_bb = np.all((pts >= bb[0]) & (pts <= bb[1]), axis=1)
    idx = np.clip(np.round((pts - bb[0]) / res).astype(int),
                  0, np.array(obs.shape) - 1)
    return in_bb & obs[idx[:, 0], idx[:, 1], idx[:, 2]]


def load_dtu_gt_filtered(dtu_eval_dir: Path, scan_id: int) -> np.ndarray:
    """Official DTU reference points after ObsMask + Plane filtering."""
    from scipy.io import loadmat
    import trimesh

    ply_path = dtu_eval_dir / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
    mat_path = dtu_eval_dir / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    plane_path = dtu_eval_dir / "ObsMask" / f"Plane{scan_id}.mat"
    if not ply_path.exists():
        raise FileNotFoundError(f"missing DTU GT cloud: {ply_path}")
    if not mat_path.exists() or not plane_path.exists():
        raise FileNotFoundError(f"missing DTU ObsMask/Plane files under {dtu_eval_dir / 'ObsMask'}")

    pts = np.asarray(trimesh.load(str(ply_path), process=False).vertices,
                     dtype=np.float32)
    mat = loadmat(str(mat_path))
    obs = mat["ObsMask"].astype(bool)
    bb = mat["BB"].astype(np.float64)
    res = float(mat["Res"].flat[0])
    in_obs = _obs_inbound(pts, obs, bb, res)
    pts_obs = pts[in_obs]

    plane = loadmat(str(plane_path))["P"].reshape(4)
    hom = np.concatenate([pts_obs, np.ones((len(pts_obs), 1), np.float32)], axis=1)
    keep = (hom * plane[None]).sum(axis=1) > 0
    pts_gt = pts_obs[keep]
    print(f"[gt] {len(pts):,} raw -> {len(pts_obs):,} ObsMask -> {len(pts_gt):,} Plane")
    return pts_gt.astype(np.float32)


def to_normalized(pts_world: np.ndarray, scene: Path) -> np.ndarray:
    cam = np.load(scene / "cameras.npz")
    scale_mat = cam["scale_mat_0"].astype(np.float64)
    inv = np.linalg.inv(scale_mat)
    hom = np.concatenate([pts_world, np.ones((len(pts_world), 1), np.float32)], axis=1)
    return (inv @ hom.T).T[:, :3].astype(np.float32)


def load_tnt_gt(gt_ply: Path, max_load: int, seed: int) -> np.ndarray:
    """Official Tanks & Temples GT point cloud in the LiDAR frame.

    The clouds are large (Ignatius ~5M points), so an optional random subsample
    keeps the per-view projection / KDTree work tractable.
    """
    import trimesh

    if not gt_ply.exists():
        raise FileNotFoundError(f"missing TnT GT cloud: {gt_ply}")
    pts = np.asarray(trimesh.load(str(gt_ply), process=False).vertices,
                     dtype=np.float64)
    if max_load > 0 and len(pts) > max_load:
        sel = np.random.default_rng(seed).choice(len(pts), size=max_load, replace=False)
        sel.sort()
        pts = pts[sel]
    print(f"[gt] tnt {gt_ply.name}: {len(pts):,} points")
    return pts.astype(np.float32)


def tnt_to_normalized(pts_world: np.ndarray, scene: Path, trans_path: Path) -> np.ndarray:
    """GT-LiDAR --inv(<scene>_trans.txt)--> COLMAP-SfM --bbox normalize--> NSVF.

    Same chain used by analysis/compare_ignatius_gt_points.py so the GT lands in
    the exact normalized frame the model and load_views() cameras live in.
    """
    trans = np.loadtxt(trans_path, dtype=np.float64).reshape(4, 4)
    hom = np.concatenate([pts_world.astype(np.float64),
                          np.ones((len(pts_world), 1))], axis=1)
    colmap = (np.linalg.inv(trans) @ hom.T).T[:, :3]
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float64).reshape(-1)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    return ((colmap - center[None]) / scale).astype(np.float32)


def estimate_pca_normals(query_pts: np.ndarray, support_pts: np.ndarray,
                         k: int, chunk: int = 50_000,
                         support_tree=None) -> tuple[np.ndarray, np.ndarray]:
    """PCA plane normals for query points using local KNN in support_pts."""
    if support_tree is None:
        from scipy.spatial import cKDTree
        support_tree = cKDTree(support_pts)
    normals = np.empty_like(query_pts, dtype=np.float32)
    curv = np.empty((len(query_pts),), dtype=np.float32)
    for start in range(0, len(query_pts), chunk):
        end = min(start + chunk, len(query_pts))
        _, nn = support_tree.query(query_pts[start:end], k=k, workers=-1)
        nbrs = support_pts[nn]
        centered = nbrs - nbrs.mean(axis=1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", centered, centered) / float(k)
        eig, evec = np.linalg.eigh(cov)
        normals[start:end] = evec[:, :, 0].astype(np.float32)
        curv[start:end] = (eig[:, 0] / np.maximum(eig.sum(axis=1), 1e-12)).astype(np.float32)
    return normals, curv


def project_points(points: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor,
                   H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uv (V,N,2) and foreground-style in-frame mask (V,N)."""
    xc = torch.einsum("vij,nj->vni", w2c[:, :3, :3], points) + w2c[:, None, :3, 3]
    uvh = torch.einsum("vij,vnj->vni", K, xc)
    uv = uvh[..., :2] / uvh[..., 2:3].clamp(min=1e-6)
    inb = ((xc[..., 2] > 0)
           & (uv[..., 0] >= 0) & (uv[..., 0] < W)
           & (uv[..., 1] >= 0) & (uv[..., 1] < H))
    return uv, inb


def foreground_visibility(points: torch.Tensor, masks: torch.Tensor,
                          K: torch.Tensor, w2c: torch.Tensor,
                          H: int, W: int, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    V = masks.shape[0]
    vis_parts, uv_parts = [], []
    rows = torch.arange(V, device=points.device)[:, None]
    for start in range(0, points.shape[0], chunk):
        end = min(start + chunk, points.shape[0])
        uv, inb = project_points(points[start:end], K, w2c, H, W)
        ui = uv[..., 0].long().clamp(0, W - 1)
        vi = uv[..., 1].long().clamp(0, H - 1)
        fg = masks[rows, vi, ui]
        vis_parts.append((inb & fg).cpu())
        uv_parts.append(uv.cpu())
    return torch.cat(vis_parts, dim=1), torch.cat(uv_parts, dim=1)


def build_pairs(vis: torch.Tensor, alt: torch.Tensor,
                max_refs_per_point: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs_p, pairs_ref, pairs_alt = [], [], []
    vis_np = vis.numpy()
    alt_np = alt.cpu().numpy()
    for pi in range(vis_np.shape[1]):
        refs = np.flatnonzero(vis_np[:, pi])
        if refs.size == 0:
            continue
        if max_refs_per_point > 0 and refs.size > max_refs_per_point:
            refs = rng.choice(refs, size=max_refs_per_point, replace=False)
        for ref in refs:
            for src in alt_np[ref]:
                if src != ref and vis_np[src, pi]:
                    pairs_p.append(pi)
                    pairs_ref.append(ref)
                    pairs_alt.append(src)
    return (np.asarray(pairs_p, dtype=np.int64),
            np.asarray(pairs_ref, dtype=np.int64),
            np.asarray(pairs_alt, dtype=np.int64))


def score_offsets(images: torch.Tensor, points: torch.Tensor, normals: torch.Tensor,
                  pidx: np.ndarray, refs: np.ndarray, alts: np.ndarray,
                  offsets: np.ndarray, color: str, K: torch.Tensor, w2c: torch.Tensor,
                  H: int, W: int, patch: int, half_pix: float,
                  ncc_min: float, sample_mode: str, gaussian_sigma: float,
                  gaussian_radius: int, patch_wsigma: float,
                  patch_bilateral_gamma: float, chunk: int,
                  capture_offset: float | None = 0.0) -> dict[str, np.ndarray]:
    n_offsets = len(offsets)
    sum_loss = np.zeros(n_offsets, dtype=np.float64)
    sum_zncc = np.zeros(n_offsets, dtype=np.float64)
    count = np.zeros(n_offsets, dtype=np.int64)
    kept = np.zeros(n_offsets, dtype=np.int64)
    capture_idx = None
    pair_zncc = None
    if capture_offset is not None:
        capture_idx = int(np.argmin(np.abs(offsets - float(capture_offset))))
        pair_zncc = np.full((len(pidx),), np.nan, dtype=np.float32)

    pidx_t = torch.from_numpy(pidx).to(points.device)
    refs_t_all = torch.from_numpy(refs).to(points.device)
    alts_t_all = torch.from_numpy(alts).to(points.device)
    for oi, off in enumerate(offsets):
        for start in range(0, len(pidx), chunk):
            end = min(start + chunk, len(pidx))
            ids = pidx_t[start:end]
            x = points[ids] + float(off) * normals[ids]
            n = normals[ids]
            ref_t = refs_t_all[start:end]
            alt_t = alts_t_all[start:end]
            want_full = capture_idx is not None and oi == capture_idx
            with torch.no_grad():
                out = pmvs_ncc_loss(
                    images, x, n, ref_t, alt_t, K, w2c, H, W,
                    patch=patch, half_pix=half_pix,
                    sample_mode=sample_mode,
                    gaussian_sigma=gaussian_sigma,
                    gaussian_radius=gaussian_radius,
                    ncc_min=ncc_min,
                    return_full=want_full,
                    ncc_color=color,
                    patch_wsigma=patch_wsigma,
                    patch_bilateral_gamma=patch_bilateral_gamma,
                )
            if want_full:
                zncc, keep, n_valid, zfull = out
                pair_zncc[start:end] = zfull.detach().cpu().numpy().astype(np.float32)
            else:
                zncc, keep, n_valid = out
            if zncc.numel() == 0:
                continue
            z = zncc.detach().cpu().numpy().astype(np.float64)
            k = keep.detach().cpu().numpy()
            sum_zncc[oi] += z.sum()
            sum_loss[oi] += (1.0 - z).sum()
            count[oi] += z.size
            kept[oi] += int(k.sum())
        print(f"[{color}] offset {off:+.5f}: pairs={count[oi]:,} kept={kept[oi]:,}")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_zncc = sum_zncc / count
        mean_loss = sum_loss / count
        kept_frac = kept / count
    out = {
        "offsets": offsets.astype(np.float32),
        "mean_zncc": mean_zncc.astype(np.float32),
        "mean_loss": mean_loss.astype(np.float32),
        "count": count,
        "kept": kept,
        "kept_frac": kept_frac.astype(np.float32),
    }
    if pair_zncc is not None:
        out["pair_zncc_at_capture"] = pair_zncc
        out["capture_offset"] = np.float32(offsets[capture_idx])
    return out


def _aggregate_ref_points(refs: np.ndarray, pidx: np.ndarray, pair_zncc: np.ndarray,
                          ref_view: int) -> tuple[np.ndarray, np.ndarray]:
    m = (refs == ref_view) & np.isfinite(pair_zncc)
    if not m.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    pts = pidx[m]
    vals = pair_zncc[m].astype(np.float64)
    uniq, inv = np.unique(pts, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    cnt = np.bincount(inv)
    return uniq.astype(np.int64), (sums / np.maximum(cnt, 1)).astype(np.float32)


def _subsample_overlay(xy: np.ndarray, val: np.ndarray, max_points: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(val) <= max_points:
        return xy, val
    sel = rng.choice(len(val), size=max_points, replace=False)
    return xy[sel], val[sel]


def plot_image_overlays(out_dir: Path, scan_id: int, images: np.ndarray,
                        uv: np.ndarray, pidx: np.ndarray, refs: np.ndarray,
                        gray_pair_zncc: np.ndarray, rgb_pair_zncc: np.ndarray,
                        overlay_views: list[int], n_auto: int,
                        max_points: int, seed: int) -> list[int]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to write the diagnostic PNGs"
        ) from exc

    rng = np.random.default_rng(seed)
    V = images.shape[0]
    if not overlay_views:
        usable = np.isfinite(gray_pair_zncc) | np.isfinite(rgb_pair_zncc)
        counts = np.bincount(refs[usable], minlength=V)
        overlay_views = [int(v) for v in np.argsort(counts)[::-1][:n_auto] if counts[v] > 0]

    written_views: list[int] = []
    for ref_view in overlay_views:
        if ref_view < 0 or ref_view >= V:
            print(f"[overlay] skip invalid view {ref_view}")
            continue
        g_pts, g_z = _aggregate_ref_points(refs, pidx, gray_pair_zncc, ref_view)
        r_pts, r_z = _aggregate_ref_points(refs, pidx, rgb_pair_zncc, ref_view)
        if len(g_pts) == 0 and len(r_pts) == 0:
            print(f"[overlay] view {ref_view:03d}: no finite GT-plane ZNCC")
            continue

        common, gi, ri = np.intersect1d(g_pts, r_pts, return_indices=True)
        d_xy = uv[ref_view, common]
        d_z = np.empty(0, dtype=np.float32)
        if len(common) > 0:
            d_z = (g_z[gi] - r_z[ri]).astype(np.float32)
            d_xy, d_z = _subsample_overlay(d_xy, d_z, max_points, rng)

        g_xy = uv[ref_view, g_pts]
        r_xy = uv[ref_view, r_pts]
        g_xy, g_z = _subsample_overlay(g_xy, g_z, max_points, rng)
        r_xy, r_z = _subsample_overlay(r_xy, r_z, max_points, rng)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        panels = [
            ("gray GT-plane ZNCC", g_xy, g_z, "magma", 0.0, 0.85),
            ("RGB GT-plane ZNCC", r_xy, r_z, "magma", 0.0, 0.85),
            ("gray - RGB ZNCC", d_xy, d_z, "coolwarm", -0.25, 0.25),
        ]
        for a, (title, xy, val, cmap, vmin, vmax) in zip(ax, panels):
            a.imshow(images[ref_view])
            if len(val) > 0:
                sc = a.scatter(xy[:, 0], xy[:, 1], c=val, s=5.0, alpha=0.82,
                               cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
                cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=8)
            a.set_title(f"{title}\nmean={np.nanmean(val):.3f}, n={len(val):,}"
                        if len(val) > 0 else title)
            a.set_axis_off()
        fig.suptitle(f"{_scene_tag(scan_id)} view {ref_view:03d}: GT projected photometry")
        fig.savefig(out_dir / f"overlay_view{ref_view:03d}_gray_rgb.png", dpi=180)
        plt.close(fig)
        written_views.append(ref_view)
        print(f"[overlay] wrote view {ref_view:03d}")
    return written_views


def _visible_in_view(points: torch.Tensor, ref_view: int, masks: torch.Tensor,
                     K: torch.Tensor, w2c: torch.Tensor, H: int, W: int,
                     chunk: int) -> tuple[np.ndarray, np.ndarray]:
    keep, uv_keep = [], []
    for start in range(0, points.shape[0], chunk):
        end = min(start + chunk, points.shape[0])
        uv, inb = project_points(points[start:end], K[ref_view:ref_view + 1],
                                 w2c[ref_view:ref_view + 1], H, W)
        uv = uv[0]
        inb = inb[0]
        ui = uv[:, 0].long().clamp(0, W - 1)
        vi = uv[:, 1].long().clamp(0, H - 1)
        fg = masks[ref_view, vi, ui]
        m = (inb & fg).detach().cpu().numpy()
        if m.any():
            keep.append(np.arange(start, end, dtype=np.int64)[m])
            uv_keep.append(uv.detach().cpu().numpy()[m])
    if not keep:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float32)
    return np.concatenate(keep), np.concatenate(uv_keep).astype(np.float32)


def _visible_in_single_view(points: torch.Tensor, view: int, masks: torch.Tensor,
                            K: torch.Tensor, w2c: torch.Tensor,
                            H: int, W: int, chunk: int) -> np.ndarray:
    parts = []
    for start in range(0, points.shape[0], chunk):
        end = min(start + chunk, points.shape[0])
        uv, inb = project_points(points[start:end], K[view:view + 1],
                                 w2c[view:view + 1], H, W)
        uv = uv[0]
        inb = inb[0]
        ui = uv[:, 0].long().clamp(0, W - 1)
        vi = uv[:, 1].long().clamp(0, H - 1)
        parts.append((inb & masks[view, vi, ui]).detach().cpu().numpy())
    return np.concatenate(parts)


def _plot_dense_overlay(out_path: Path, scan_id: int, ref_view: int,
                        image: np.ndarray, uv_ref: np.ndarray,
                        gray_pts: np.ndarray, gray_z: np.ndarray,
                        rgb_pts: np.ndarray, rgb_z: np.ndarray,
                        max_draw: int, seed: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to write the diagnostic PNGs"
        ) from exc

    rng = np.random.default_rng(seed + ref_view)
    g_xy, g_z = _subsample_overlay(uv_ref[gray_pts], gray_z, max_draw, rng)
    r_xy, r_z = _subsample_overlay(uv_ref[rgb_pts], rgb_z, max_draw, rng)
    common, gi, ri = np.intersect1d(gray_pts, rgb_pts, return_indices=True)
    d_xy = uv_ref[common]
    d_z = (gray_z[gi] - rgb_z[ri]).astype(np.float32) if len(common) else np.empty(0, np.float32)
    d_xy, d_z = _subsample_overlay(d_xy, d_z, max_draw, rng)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        ("gray GT-plane ZNCC", g_xy, g_z, "magma", 0.0, 0.85),
        ("RGB GT-plane ZNCC", r_xy, r_z, "magma", 0.0, 0.85),
        ("gray - RGB ZNCC", d_xy, d_z, "coolwarm", -0.25, 0.25),
    ]
    for a, (title, xy, val, cmap, vmin, vmax) in zip(ax, panels):
        a.imshow(image)
        if len(val) > 0:
            size = 1.2 if len(val) <= 120_000 else 0.45
            alpha = 0.72 if len(val) <= 120_000 else 0.62
            sc = a.scatter(xy[:, 0], xy[:, 1], c=val, s=size, alpha=alpha,
                           cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
            cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=8)
        a.set_title(f"{title}\nmean={np.nanmean(val):.3f}, n={len(val):,}"
                    if len(val) > 0 else title)
        a.set_axis_off()
    fig.suptitle(f"{_scene_tag(scan_id)} view {ref_view:03d}: dense GT projected photometry")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _raster_overlay(image: np.ndarray, xy: np.ndarray, val: np.ndarray,
                    cmap_name: str, vmin: float, vmax: float,
                    alpha: float = 0.78) -> np.ndarray:
    import matplotlib.pyplot as plt

    H, W = image.shape[:2]
    base = np.clip(image.astype(np.float32), 0.0, 1.0)
    if len(val) == 0:
        return base
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & np.isfinite(val)
    if not m.any():
        return base
    flat = yi[m] * W + xi[m]
    vv = val[m].astype(np.float64)
    sums = np.bincount(flat, weights=vv, minlength=H * W)
    cnt = np.bincount(flat, minlength=H * W)
    avg = np.full(H * W, np.nan, dtype=np.float32)
    ok = cnt > 0
    avg[ok] = (sums[ok] / cnt[ok]).astype(np.float32)
    avg = avg.reshape(H, W)
    ok2 = ok.reshape(H, W)

    cmap = plt.get_cmap(cmap_name)
    t = np.clip((avg - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    color = cmap(np.nan_to_num(t, nan=0.0))[..., :3].astype(np.float32)
    out = base.copy()
    a = alpha * ok2[..., None].astype(np.float32)
    out = out * (1.0 - a) + color * a
    return np.clip(out, 0.0, 1.0)


def _plot_paper_grid(out_path: Path, scan_id: int,
                     panels: list[dict], views_per_row: int = 2) -> None:
    if not panels:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n = len(panels)
    rows = int(np.ceil(n / views_per_row))
    cols = views_per_row * 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.05 * cols, 2.45 * rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    headings = ["gray NCC", "RGB NCC", "gray - RGB"]
    for i in range(rows * views_per_row):
        r = i // views_per_row
        g = i % views_per_row
        c0 = 3 * g
        if i >= n:
            for c in range(c0, c0 + 3):
                axes[r, c].set_axis_off()
            continue
        p = panels[i]
        xys = [p["gray_xy"], p["rgb_xy"], p["diff_xy"]]
        vals = [p["gray_z"], p["rgb_z"], p["diff_z"]]
        cmaps = ["magma", "magma", "coolwarm"]
        vmins = [0.0, 0.0, -0.25]
        vmaxs = [0.85, 0.85, 0.25]
        stats = [p["gray_mean"], p["rgb_mean"], p["diff_mean"]]
        counts = [p["gray_total_n"], p["rgb_total_n"], p["diff_total_n"]]
        for j in range(3):
            ax = axes[r, c0 + j]
            ax.imshow(p["image"])
            if len(vals[j]) > 0:
                ax.scatter(xys[j][:, 0], xys[j][:, 1], c=vals[j],
                           s=1.15, alpha=0.72, cmap=cmaps[j],
                           vmin=vmins[j], vmax=vmaxs[j], linewidths=0,
                           rasterized=True)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(headings[j], fontsize=11, pad=4)
            if j == 0:
                ax.text(0.015, 0.965, f"v{p['view']:03d}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=10, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
            ax.text(0.985, 0.035, f"{stats[j]:.2f} / {counts[j]//1000:d}k",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none"))

    zncc_sm = ScalarMappable(norm=Normalize(0.0, 0.85), cmap="magma")
    diff_sm = ScalarMappable(norm=Normalize(-0.25, 0.25), cmap="coolwarm")
    zncc_axes = [axes[r, c] for r in range(rows) for c in range(cols)
                 if c % 3 in (0, 1)]
    diff_axes = [axes[r, c] for r in range(rows) for c in range(cols)
                 if c % 3 == 2]
    cb1 = fig.colorbar(zncc_sm, ax=zncc_axes, fraction=0.012, pad=0.004)
    cb1.set_label("ZNCC", fontsize=9)
    cb2 = fig.colorbar(diff_sm, ax=diff_axes, fraction=0.012, pad=0.004)
    cb2.set_label("gray - RGB", fontsize=9)
    fig.suptitle(f"{_scene_tag(scan_id)}: dense GT-plane photometry overlays",
                 fontsize=14)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _plot_crop_distribution(out_path: Path, scene_tag: str, ref_view: int,
                            gray_box: np.ndarray, rgb_box: np.ndarray,
                            crop_label: str) -> None:
    """Minimal ICLR-style ZNCC histogram for the GT points inside one crop box.

    gray and rgb are the per-point aggregated GT-plane ZNCC of the points whose
    reference-view projection falls inside the box; both come from the unchanged
    pmvs_ncc_loss, so this is just a re-slice of the dense overlay, not a new
    metric.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(-0.2, 1.0, 49)
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    for vals, color, name in ((gray_box, "#1f77b4", "gray"),
                              (rgb_box, "#d62728", "RGB")):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            continue
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8,
                color=color, label=f"{name}  (med {np.median(v):.2f}, n={v.size:,})")
        ax.axvline(np.median(v), color=color, lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("GT-surface ZNCC")
    ax.set_ylabel("density")
    ax.set_xlim(-0.2, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"{scene_tag}  v{ref_view:03d}  {crop_label}".strip(), fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=240)
    plt.close(fig)


def _plot_crop_check(out_path: Path, scene_tag: str, ref_view: int,
                     image: np.ndarray, box: tuple[float, float, float, float],
                     xy_in: np.ndarray, z_in: np.ndarray) -> None:
    """Sanity panel: draw the crop box on the ref image + the in-box points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = box
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.imshow(image)
    if len(z_in):
        ax.scatter(xy_in[:, 0], xy_in[:, 1], c=z_in, s=2.0, cmap="magma",
                   vmin=0.0, vmax=0.85, linewidths=0)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor="red", lw=2.0))
    ax.set_title(f"{scene_tag}  v{ref_view:03d}  crop = ({x0:.0f},{y0:.0f})-"
                 f"({x1:.0f},{y1:.0f})  n_in={len(z_in):,}", fontsize=9)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def dense_image_overlays(out_dir: Path, scan_id: int, gt_norm: np.ndarray,
                         views: dict, images_t: torch.Tensor, masks: torch.Tensor,
                         K: torch.Tensor, w2c: torch.Tensor, alt: torch.Tensor,
                         overlay_views: list[int], max_points_per_view: int,
                         max_draw: int, normal_k: int, seed: int, chunk: int,
                         H: int, W: int, patch: int, half_pix: float,
                         ncc_min: float, sample_mode: str,
                         gaussian_sigma: float, gaussian_radius: int,
                         patch_wsigma: float, patch_bilateral_gamma: float,
                         device: str,
                         crop_box: tuple[float, float, float, float] | None = None,
                         crop_views: tuple[int, ...] = ()) -> list[int]:
    if max_points_per_view == 0:
        return []
    if not overlay_views:
        print("[dense-overlay] no overlay views selected")
        return []

    rng = np.random.default_rng(seed)
    all_points_t = torch.from_numpy(gt_norm).to(device).float()
    cam_mean = views["c2w"][:, :3, 3].mean(dim=0).numpy()
    images_np = views["images"].numpy()
    written = []
    from scipy.spatial import cKDTree
    support_tree = cKDTree(gt_norm)
    grid_panels: list[dict] = []
    # Crop: the box is defined in ONE view's pixel frame (crop_views[0]); it
    # selects a set of 3D GT points (the rings). We then pool those points'
    # ZNCC across EVERY processed reference view into a single distribution.
    crop_box_view = crop_views[0] if (crop_box is not None and crop_views) else None
    crop_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    ring_global: np.ndarray | None = None
    for ref_view in overlay_views:
        keep_global, uv_ref = _visible_in_view(
            all_points_t, ref_view, masks, K, w2c, H, W, chunk)
        if len(keep_global) == 0:
            print(f"[dense-overlay] view {ref_view:03d}: no visible GT points")
            continue
        if max_points_per_view > 0 and len(keep_global) > max_points_per_view:
            sel = rng.choice(len(keep_global), size=max_points_per_view, replace=False)
            keep_global = keep_global[sel]
            uv_ref = uv_ref[sel]

        pts_np = gt_norm[keep_global]
        normals_np, _ = estimate_pca_normals(
            pts_np, gt_norm, k=normal_k, support_tree=support_tree)
        to_cam = cam_mean[None] - pts_np
        flip = (normals_np * to_cam).sum(axis=1) < 0
        normals_np[flip] *= -1.0

        pts_t = torch.from_numpy(pts_np).to(device).float()
        normals_t = torch.from_numpy(normals_np).to(device).float()
        local_p, local_ref, local_alt = [], [], []
        for alt_view_t in alt[ref_view]:
            alt_view = int(alt_view_t)
            vis_alt = _visible_in_single_view(pts_t, alt_view, masks, K, w2c, H, W, chunk)
            idx = np.flatnonzero(vis_alt)
            if len(idx):
                local_p.append(idx)
                local_ref.append(np.full(len(idx), ref_view, dtype=np.int64))
                local_alt.append(np.full(len(idx), alt_view, dtype=np.int64))
        if not local_p:
            print(f"[dense-overlay] view {ref_view:03d}: no visible alt pairs")
            continue
        pidx = np.concatenate(local_p)
        refs = np.concatenate(local_ref)
        alts = np.concatenate(local_alt)

        results = {}
        for color in ("gray", "rgb"):
            results[color] = score_offsets(
                images_t, pts_t, normals_t, pidx, refs, alts,
                np.asarray([0.0], dtype=np.float32), color, K, w2c, H, W,
                patch, half_pix, ncc_min, sample_mode, gaussian_sigma,
                gaussian_radius, patch_wsigma, patch_bilateral_gamma, chunk,
                capture_offset=0.0)
        g_pts, g_z = _aggregate_ref_points(
            refs, pidx, results["gray"]["pair_zncc_at_capture"], ref_view)
        r_pts, r_z = _aggregate_ref_points(
            refs, pidx, results["rgb"]["pair_zncc_at_capture"], ref_view)
        out_path = out_dir / f"dense_overlay_view{ref_view:03d}_gray_rgb.png"
        _plot_dense_overlay(out_path, scan_id, ref_view, images_np[ref_view],
                            uv_ref, g_pts, g_z, r_pts, r_z, max_draw, seed)
        if crop_box is not None:
            # Record this view's per-point ZNCC keyed by GLOBAL GT id so the
            # ring points can be pooled across all views after the loop.
            gid_g = keep_global[g_pts]
            gid_r = keep_global[r_pts]
            crop_records.append((gid_g, g_z, gid_r, r_z))
            if ref_view == crop_box_view:
                x0, y0, x1, y1 = crop_box
                g_uv = uv_ref[g_pts]
                in_box = ((g_uv[:, 0] >= x0) & (g_uv[:, 0] <= x1)
                          & (g_uv[:, 1] >= y0) & (g_uv[:, 1] <= y1))
                ring_global = gid_g[in_box]
                _plot_crop_check(
                    out_dir / f"crop_check_view{ref_view:03d}.png", _scene_tag(scan_id),
                    ref_view, images_np[ref_view], crop_box, g_uv[in_box], g_z[in_box])
                print(f"[crop] box view {ref_view:03d}: {ring_global.size:,} ring "
                      f"GT points selected")
        common, gi, ri = np.intersect1d(g_pts, r_pts, return_indices=True)
        d_z = (g_z[gi] - r_z[ri]).astype(np.float32) if len(common) else np.empty(0, np.float32)
        grid_rng = np.random.default_rng(seed + 10_000 + ref_view)
        grid_max = max_draw if max_draw > 0 else 60_000
        g_xy_grid, g_z_grid = _subsample_overlay(uv_ref[g_pts], g_z, grid_max, grid_rng)
        r_xy_grid, r_z_grid = _subsample_overlay(uv_ref[r_pts], r_z, grid_max, grid_rng)
        d_xy_grid, d_z_grid = _subsample_overlay(uv_ref[common], d_z, grid_max, grid_rng)
        grid_panels.append({
            "view": ref_view,
            "image": images_np[ref_view],
            "gray_xy": g_xy_grid,
            "gray_z": g_z_grid,
            "rgb_xy": r_xy_grid,
            "rgb_z": r_z_grid,
            "diff_xy": d_xy_grid,
            "diff_z": d_z_grid,
            "gray_mean": float(np.nanmean(g_z)) if len(g_z) else float("nan"),
            "rgb_mean": float(np.nanmean(r_z)) if len(r_z) else float("nan"),
            "diff_mean": float(np.nanmean(d_z)) if len(d_z) else float("nan"),
            "gray_total_n": int(len(g_z)),
            "rgb_total_n": int(len(r_z)),
            "diff_total_n": int(len(d_z)),
        })
        written.append(ref_view)
        print(f"[dense-overlay] wrote view {ref_view:03d}: "
              f"{len(keep_global):,} ref pts, {len(pidx):,} pairs")
    _plot_paper_grid(out_dir / "dense_overlay_grid_gray_rgb.png",
                     scan_id, grid_panels)

    if crop_box is not None:
        if ring_global is None:
            print("[crop] box view was not among the dense views; no ring set")
        elif ring_global.size == 0:
            print("[crop] no GT points fell inside the box")
        else:
            ring_set = ring_global
            pooled_g, pooled_r = [], []
            for gid_g, gz, gid_r, rz in crop_records:
                pooled_g.append(gz[np.isin(gid_g, ring_set)])
                pooled_r.append(rz[np.isin(gid_r, ring_set)])
            pooled_g = np.concatenate(pooled_g) if pooled_g else np.empty(0, np.float32)
            pooled_r = np.concatenate(pooled_r) if pooled_r else np.empty(0, np.float32)
            _plot_crop_distribution(
                out_dir / "crop_zncc_dist_allviews", _scene_tag(scan_id),
                crop_box_view, pooled_g, pooled_r,
                crop_label=f"rings, pooled over {len(crop_records)} views")
            print(f"[crop] pooled ring ZNCC over {len(crop_records)} views: "
                  f"gray n={pooled_g.size:,} med={np.median(pooled_g):.3f} | "
                  f"rgb n={pooled_r.size:,} med={np.median(pooled_r):.3f}")
    return written


def plot_results(out_dir: Path, scan_id: int, gray: dict, rgb: dict,
                 pidx: np.ndarray, vis_counts: np.ndarray, curv: np.ndarray,
                 n_points: int, n_pairs: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to write the diagnostic PNGs"
        ) from exc

    offsets = gray["offsets"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(offsets, gray["mean_loss"], "-o", label="gray")
    ax[0].plot(offsets, rgb["mean_loss"], "-o", label="rgb")
    ax[0].axvline(0, color="k", lw=1, alpha=0.45)
    ax[0].set_xlabel("normal offset in normalized scene units")
    ax[0].set_ylabel("mean NCC loss 1 - ZNCC")
    ax[0].set_title("lower is better")
    ax[0].legend()

    ax[1].plot(offsets, gray["mean_zncc"], "-o", label="gray")
    ax[1].plot(offsets, rgb["mean_zncc"], "-o", label="rgb")
    ax[1].axvline(0, color="k", lw=1, alpha=0.45)
    ax[1].set_xlabel("normal offset in normalized scene units")
    ax[1].set_ylabel("mean ZNCC")
    ax[1].set_title("higher is better")
    ax[1].legend()
    fig.suptitle(f"{_scene_tag(scan_id)} GT-plane NCC profile  "
                 f"({n_points:,} GT pts, {n_pairs:,} ref-alt pairs)")
    fig.savefig(out_dir / "gt_ncc_profile_gray_vs_rgb.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(offsets, gray["kept_frac"], "-o", label="gray")
    ax[0].plot(offsets, rgb["kept_frac"], "-o", label="rgb")
    ax[0].set_xlabel("normal offset")
    ax[0].set_ylabel("fraction ZNCC > ncc_min")
    ax[0].set_ylim(0, 1)
    ax[0].legend()

    vc = vis_counts[pidx]
    ax[1].hist(vc, bins=np.arange(vc.min(), vc.max() + 2) - 0.5,
               color="#4c78a8", alpha=0.85)
    ax[1].set_xlabel("foreground-visible views per evaluated pair point")
    ax[1].set_ylabel("pair count")
    ax[1].set_title(f"median curvature {np.median(curv[pidx]):.2e}")
    fig.suptitle("GT visibility and PMVS gate support")
    fig.savefig(out_dir / "gt_ncc_support.png", dpi=180)
    plt.close(fig)


def parse_offsets(text: str) -> np.ndarray:
    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) < 3:
        raise argparse.ArgumentTypeError("provide at least three comma-separated offsets")
    return np.asarray(vals, dtype=np.float32)


def parse_views(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True,
                    help="Scene directory: DTU (cameras.npz, image/, mask/) or "
                         "NSVF/TnT (intrinsics.txt, bbox.txt, pose/, rgb/, mask/).")
    ap.add_argument("--dataset", choices=["dtu", "tnt"], default="dtu",
                    help="GT source: dtu = ObsMask/Plane-filtered stl cloud; "
                         "tnt = official Tanks & Temples GT ply + _trans.txt.")
    ap.add_argument("--dtu-eval-dir", type=Path, default=None,
                    help="DTU eval directory containing Points/stl and ObsMask (dtu only).")
    ap.add_argument("--scan-id", type=int, default=None,
                    help="DTU scan id; optional for tnt (used only for npz/summary labels).")
    ap.add_argument("--gt-ply", type=Path, default=None,
                    help="Official TnT GT point cloud, LiDAR frame (tnt only).")
    ap.add_argument("--gt-trans", type=Path, default=None,
                    help="<scene>_trans.txt aligning GT-LiDAR to COLMAP-SfM (tnt only).")
    ap.add_argument("--gt-max-load", type=int, default=2_000_000,
                    help="Subsample the loaded TnT GT cloud to at most this many "
                         "points before projection/KDTree; 0 = keep all.")
    ap.add_argument("--label", type=str, default=None,
                    help="Figure title prefix (e.g. 'TnT Ignatius'); default per dataset.")
    ap.add_argument("--out", type=Path, default=Path("outputs/gt_photometry_diag"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--down", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=20_000,
                    help="Random GT points to evaluate after visibility filtering; 0 = all.")
    ap.add_argument("--normal-k", type=int, default=32)
    ap.add_argument("--n-alt", type=int, default=6)
    ap.add_argument("--view-selection", choices=["nearest", "pairs", "auto"], default="nearest",
                    help="Alternative-view pool: nearest matches the current training default.")
    ap.add_argument("--pairs", type=Path, default=None,
                    help="Optional MVSNet pair.txt. Explicit --pairs implies --view-selection pairs.")
    ap.add_argument("--max-refs-per-point", type=int, default=3,
                    help="Subsample foreground reference views per GT point; 0 = all.")
    ap.add_argument("--offsets", type=parse_offsets,
                    default=parse_offsets("-0.03,-0.02,-0.01,-0.005,0,0.005,0.01,0.02,0.03"))
    ap.add_argument("--ncc-patch", type=int, default=5)
    ap.add_argument("--ncc-half-pix", type=float, default=2.0)
    ap.add_argument("--ncc-min", type=float, default=0.0)
    ap.add_argument("--sample-mode", choices=["bilinear", "gaussian"], default="bilinear")
    ap.add_argument("--gaussian-sigma", type=float, default=2.0)
    ap.add_argument("--gaussian-radius", type=int, default=2)
    ap.add_argument("--ncc-patch-wsigma", type=float, default=0.0)
    ap.add_argument("--ncc-bilateral-gamma", type=float, default=0.0)
    ap.add_argument("--overlay-views", type=parse_views, default=[],
                    help="Comma-separated reference views for image overlays; empty = auto top views.")
    ap.add_argument("--n-overlays", type=int, default=4,
                    help="Number of auto-selected overlay views.")
    ap.add_argument("--overlay-max-points", type=int, default=12_000,
                    help="Display subsample per overlay panel; 0 = draw all.")
    ap.add_argument("--dense-overlay-points-per-view", type=int, default=0,
                    help="Extra per-view GT points for dense paper overlays; 0 disables, negative = all visible.")
    ap.add_argument("--dense-overlay-max-draw", type=int, default=80_000,
                    help="Display subsample for dense overlays; 0 = draw all.")
    ap.add_argument("--crop-box", type=str, default=None,
                    help="x0,y0,x1,y1 pixel box (ref-view image coords); emits a "
                         "minimal ZNCC distribution plot for GT points inside it.")
    ap.add_argument("--crop-views", type=parse_views, default=[],
                    help="Views the --crop-box applies to; empty = all dense views.")
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()

    global torch, alt_cameras_from_pairs, load_views, precompute_alt_cameras, pmvs_ncc_loss
    import torch
    from lip_tracer.data import alt_cameras_from_pairs, load_views, precompute_alt_cameras
    from lip_tracer.loss import pmvs_ncc_loss

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[data] loading views from {args.scene}")
    views = load_views(args.scene, down=args.down)
    images = views["images"].to(args.device)
    masks = views["masks"].to(args.device).bool()
    K = views["K"].to(args.device).float()
    w2c = torch.linalg.inv(views["c2w"].to(args.device).float())
    H, W = int(views["H"]), int(views["W"])

    global SCENE_TITLE
    if args.dataset == "tnt":
        if args.gt_ply is None or args.gt_trans is None:
            ap.error("--dataset tnt requires --gt-ply and --gt-trans")
        gt_world = load_tnt_gt(args.gt_ply, args.gt_max_load, args.seed)
        gt_norm = tnt_to_normalized(gt_world, args.scene, args.gt_trans)
        SCENE_TITLE = args.label or f"TnT {args.scene.name}"
        if args.scan_id is None:
            args.scan_id = 0
    else:
        if args.dtu_eval_dir is None or args.scan_id is None:
            ap.error("--dataset dtu requires --dtu-eval-dir and --scan-id")
        gt_world = load_dtu_gt_filtered(args.dtu_eval_dir, args.scan_id)
        gt_norm = to_normalized(gt_world, args.scene)
        SCENE_TITLE = args.label

    if args.max_points > 0 and len(gt_norm) > args.max_points:
        eval_idx = rng.choice(len(gt_norm), size=args.max_points, replace=False)
        eval_idx.sort()
    else:
        eval_idx = np.arange(len(gt_norm))
    points_np = gt_norm[eval_idx]
    print(f"[gt] evaluating {len(points_np):,}/{len(gt_norm):,} GT points")

    normals_np, curv = estimate_pca_normals(points_np, gt_norm, k=args.normal_k)
    cam_mean = views["c2w"][:, :3, 3].mean(dim=0).numpy()
    to_cam = cam_mean[None] - points_np
    flip = (normals_np * to_cam).sum(axis=1) < 0
    normals_np[flip] *= -1.0

    points = torch.from_numpy(points_np).to(args.device).float()
    normals = torch.from_numpy(normals_np).to(args.device).float()
    vis, uv = foreground_visibility(points, masks, K, w2c, H, W, args.chunk)
    vis_counts = vis.sum(dim=0).numpy()
    valid_pts = vis_counts >= 2
    if not valid_pts.any():
        raise RuntimeError("no GT points project inside foreground masks in at least two views")

    keep_idx = np.flatnonzero(valid_pts)
    points = points[keep_idx]
    normals = normals[keep_idx]
    curv = curv[keep_idx]
    vis = vis[:, keep_idx]
    vis_counts = vis_counts[keep_idx]
    uv = uv[:, keep_idx]
    print(f"[vis] {len(keep_idx):,} GT points have >=2 foreground views")

    pair_path = args.pairs
    use_pairs = args.view_selection in ("pairs", "auto") or pair_path is not None
    if use_pairs and pair_path is None:
        for candidate in (args.scene / "pairs.txt", args.scene / "pair.txt"):
            if candidate.exists():
                pair_path = candidate
                break
    if use_pairs and pair_path is not None:
        print(f"[views] source views from {pair_path}")
        alt = alt_cameras_from_pairs(pair_path, views["c2w"].shape[0], args.n_alt)
        view_selection_used = "pairs"
    elif args.view_selection == "pairs":
        raise FileNotFoundError("requested --view-selection pairs, but no pair file was found")
    else:
        print("[views] source views from nearest camera centres")
        alt = precompute_alt_cameras(views, args.n_alt)
        view_selection_used = "nearest"

    pidx, refs, alts = build_pairs(vis, alt, args.max_refs_per_point, args.seed)
    if len(pidx) == 0:
        raise RuntimeError("no foreground ref-alt pairs survived view selection")
    print(f"[pairs] {len(pidx):,} point/ref/alt pairs")

    # p_ref = project(x_gt, ref_camera); saved for auditing the exact reference
    # centres used by the unchanged pmvs_ncc_loss patch construction.
    ref_uv = uv[torch.from_numpy(refs), torch.from_numpy(pidx)].numpy()

    results = {}
    for color in ("gray", "rgb"):
        results[color] = score_offsets(
            images, points, normals, pidx, refs, alts, args.offsets, color,
            K, w2c, H, W, args.ncc_patch, args.ncc_half_pix, args.ncc_min,
            args.sample_mode, args.gaussian_sigma, args.gaussian_radius,
            args.ncc_patch_wsigma, args.ncc_bilateral_gamma, args.chunk)

    np.savez_compressed(
        args.out / "gt_ncc_diag.npz",
        points=points.cpu().numpy(),
        normals=normals.cpu().numpy(),
        curvature=curv,
        visibility_count=vis_counts,
        pair_point=pidx,
        pair_ref=refs,
        pair_alt=alts,
        pair_ref_uv=ref_uv,
        gray_pair_zncc_at_zero=results["gray"]["pair_zncc_at_capture"],
        gray_mean_loss=results["gray"]["mean_loss"],
        gray_mean_zncc=results["gray"]["mean_zncc"],
        gray_count=results["gray"]["count"],
        gray_kept=results["gray"]["kept"],
        rgb_pair_zncc_at_zero=results["rgb"]["pair_zncc_at_capture"],
        rgb_mean_loss=results["rgb"]["mean_loss"],
        rgb_mean_zncc=results["rgb"]["mean_zncc"],
        rgb_count=results["rgb"]["count"],
        rgb_kept=results["rgb"]["kept"],
        offsets=args.offsets,
    )
    plot_results(args.out, args.scan_id, results["gray"], results["rgb"],
                 pidx, vis_counts, curv, len(points), len(pidx))
    overlay_views = plot_image_overlays(
        args.out, args.scan_id, views["images"].numpy(), uv.numpy(), pidx, refs,
        results["gray"]["pair_zncc_at_capture"],
        results["rgb"]["pair_zncc_at_capture"],
        args.overlay_views, args.n_overlays, args.overlay_max_points, args.seed)
    crop_box = None
    if args.crop_box:
        vals = [float(x) for x in args.crop_box.split(",") if x.strip()]
        if len(vals) != 4:
            ap.error("--crop-box expects x0,y0,x1,y1")
        crop_box = (min(vals[0], vals[2]), min(vals[1], vals[3]),
                    max(vals[0], vals[2]), max(vals[1], vals[3]))
    dense_overlay_views = dense_image_overlays(
        args.out, args.scan_id, gt_norm, views, images, masks, K, w2c, alt,
        overlay_views, args.dense_overlay_points_per_view,
        args.dense_overlay_max_draw, args.normal_k, args.seed, args.chunk,
        H, W, args.ncc_patch, args.ncc_half_pix, args.ncc_min,
        args.sample_mode, args.gaussian_sigma, args.gaussian_radius,
        args.ncc_patch_wsigma, args.ncc_bilateral_gamma, args.device,
        crop_box=crop_box, crop_views=tuple(args.crop_views))

    summary = {
        "scene": str(args.scene),
        "dtu_eval_dir": str(args.dtu_eval_dir),
        "scan_id": args.scan_id,
        "n_gt_loaded": int(len(gt_norm)),
        "n_gt_evaluated": int(len(points)),
        "n_pairs": int(len(pidx)),
        "offsets": args.offsets.tolist(),
        "gray_best_offset": float(args.offsets[np.nanargmin(results["gray"]["mean_loss"])]),
        "rgb_best_offset": float(args.offsets[np.nanargmin(results["rgb"]["mean_loss"])]),
        "gray_loss_at_zero": float(results["gray"]["mean_loss"][np.argmin(np.abs(args.offsets))]),
        "rgb_loss_at_zero": float(results["rgb"]["mean_loss"][np.argmin(np.abs(args.offsets))]),
        "view_selection": view_selection_used,
        "pairs_path": str(pair_path) if view_selection_used == "pairs" else None,
        "ncc_patch": args.ncc_patch,
        "ncc_half_pix": args.ncc_half_pix,
        "ncc_min": args.ncc_min,
        "sample_mode": args.sample_mode,
        "overlay_views": overlay_views,
        "dense_overlay_views": dense_overlay_views,
        "dense_overlay_points_per_view": args.dense_overlay_points_per_view,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {args.out / 'gt_ncc_profile_gray_vs_rgb.png'}")
    print(f"[done] wrote {args.out / 'gt_ncc_support.png'}")
    print(f"[done] wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
