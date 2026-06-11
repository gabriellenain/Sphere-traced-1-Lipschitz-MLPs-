"""Post-hoc 3D photoconsistency ridge visualization.

This is a diagnostic tool, not a reconstruction method.  It evaluates a
regular voxel grid, scores each voxel by multi-view patch ZNCC, smooths that
score volume, then extracts candidate sheet ridges from the smoothed scalar
field.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lip_tracer.data import load_blender_views, load_views


def _parse_ints(s: str | None, n_total: int) -> list[int]:
    if not s:
        return list(range(n_total))
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) if x else None for x in part.split(":")]
            sl = slice(*bits)
            out.extend(range(n_total)[sl])
        else:
            out.append(int(part))
    return sorted(set(i for i in out if 0 <= i < n_total))


def _make_grid(res: int, bbox_min: np.ndarray, bbox_max: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    axes = [np.linspace(bbox_min[i], bbox_max[i], res, dtype=np.float32) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    spacing = tuple(float((bbox_max[i] - bbox_min[i]) / max(res - 1, 1)) for i in range(3))
    return pts, spacing


def _project_points(x: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points into each camera.

    Returns:
        uv:    (V, N, 2) image coordinates
        front: (V, N) points with positive camera-space z
    """
    w2c = torch.linalg.inv(c2w)
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    xc = torch.einsum("vij,nj->vni", R, x) + t[:, None, :]
    z = xc[..., 2]
    uvh = torch.einsum("vij,vnj->vni", K, xc)
    uv = uvh[..., :2] / uvh[..., 2:3].clamp_min(1e-6)
    return uv, z > 1e-4


def _sample_mask(masks: torch.Tensor, uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Nearest-neighbor mask sampling for uv of shape (V, N, 2)."""
    V, N = uv.shape[:2]
    xs = uv[..., 0].round().long().clamp(0, W - 1)
    ys = uv[..., 1].round().long().clamp(0, H - 1)
    vi = torch.arange(V, device=uv.device)[:, None].expand(V, N)
    return masks[vi, ys, xs]


def _patch_grid(uv: torch.Tensor, H: int, W: int, radius: int, dilation: float) -> torch.Tensor:
    """Build grid_sample coordinates for patches around projected pixels."""
    offsets = torch.arange(-radius, radius + 1, device=uv.device, dtype=uv.dtype) * dilation
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    px = uv[:, 0:1] + ox.reshape(1, -1)
    py = uv[:, 1:2] + oy.reshape(1, -1)
    gx = px / max(W - 1, 1) * 2.0 - 1.0
    gy = py / max(H - 1, 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).view(uv.shape[0], 2 * radius + 1, 2 * radius + 1, 2)


def _zncc_score(
    x: torch.Tensor,
    imgs_chw: torch.Tensor,
    masks: torch.Tensor | None,
    K: torch.Tensor,
    c2w: torch.Tensor,
    H: int,
    W: int,
    patch_radius: int,
    patch_dilation: float,
    min_views: int,
    max_views: int,
    agg: str,
    topk: int,
    mask_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one photoconsistency score per point."""
    uv, front = _project_points(x, K, c2w)
    in_bounds = (
        (uv[..., 0] >= patch_radius * patch_dilation)
        & (uv[..., 0] <= W - 1 - patch_radius * patch_dilation)
        & (uv[..., 1] >= patch_radius * patch_dilation)
        & (uv[..., 1] <= H - 1 - patch_radius * patch_dilation)
    )
    valid = front & in_bounds
    if masks is not None and mask_mode == "foreground":
        valid = valid & _sample_mask(masks, uv, H, W)

    N = x.shape[0]
    scores = torch.full((N,), -1.0, device=x.device)
    valid_counts = valid.sum(dim=0).to(torch.int16)
    pair_counts = torch.zeros((N,), dtype=torch.int16, device=x.device)

    V = imgs_chw.shape[0]
    patch_dim = 3 * (2 * patch_radius + 1) ** 2
    all_patches = torch.empty((V, N, patch_dim), device=x.device, dtype=imgs_chw.dtype)
    for vi in range(V):
        grid = _patch_grid(uv[vi], H, W, patch_radius, patch_dilation)
        ph = 2 * patch_radius + 1
        sampled = F.grid_sample(
            imgs_chw[vi:vi + 1],
            grid.view(1, N * ph, ph, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        all_patches[vi] = sampled.view(3, N, ph, ph).permute(1, 0, 2, 3).reshape(N, -1)

    ph = 2 * patch_radius + 1
    for ni in range(N):
        view_ids = torch.nonzero(valid[:, ni], as_tuple=False).flatten()
        if view_ids.numel() < min_views:
            continue
        if view_ids.numel() > max_views:
            # Deterministic spread over the available cameras avoids a nearest-view bias.
            pick = torch.linspace(0, view_ids.numel() - 1, max_views, device=x.device).round().long()
            view_ids = view_ids[pick]

        p = all_patches[view_ids, ni]                    # (V', 3*ph*ph)
        # Per-channel ZNCC: subtract per-channel mean, then normalise per-channel norm.
        # Mixing channels in a single mean/norm would bias the score toward zero.
        p = p.reshape(view_ids.numel(), 3, ph * ph)      # (V', 3, ph*ph)
        p = p - p.mean(dim=2, keepdim=True)              # per-channel mean
        p = p / p.norm(dim=2, keepdim=True).clamp_min(1e-6)
        p = p.reshape(view_ids.numel(), -1)              # (V', 3*ph*ph)
        sim = p @ p.T / 3.0                              # average over 3 channels
        iu, ju = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1, device=x.device)
        vals = sim[iu, ju]
        if vals.numel() == 0:
            continue
        pair_counts[ni] = vals.numel()
        if agg == "median":
            scores[ni] = vals.median()
        elif agg == "mean":
            scores[ni] = vals.mean()
        else:
            k = min(topk, vals.numel())
            scores[ni] = torch.topk(vals, k=k, largest=True).values.mean()
    return scores, valid_counts, pair_counts


def compute_photoconsistency(
    views: dict,
    points: np.ndarray,
    device: str,
    chunk: int,
    patch_radius: int,
    patch_dilation: float,
    min_views: int,
    max_views: int,
    agg: str,
    topk: int,
    mask_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = views["images"].float().to(device)
    imgs_chw = images.permute(0, 3, 1, 2).contiguous()
    masks = views.get("masks")
    masks_t = masks.to(device) if masks is not None and mask_mode != "none" else None
    K = views["K"].float().to(device)
    c2w = views["c2w"].float().to(device)
    H, W = int(views["H"]), int(views["W"])

    P = np.full((len(points),), -1.0, dtype=np.float32)
    valid_counts = np.zeros((len(points),), dtype=np.int16)
    pair_counts = np.zeros((len(points),), dtype=np.int16)
    t0 = time.time()
    for s in range(0, len(points), chunk):
        e = min(s + chunk, len(points))
        x = torch.from_numpy(points[s:e]).float().to(device)
        with torch.no_grad():
            p, vc, pc = _zncc_score(
                x, imgs_chw, masks_t, K, c2w, H, W,
                patch_radius, patch_dilation, min_views, max_views, agg, topk, mask_mode,
            )
        P[s:e] = p.cpu().numpy()
        valid_counts[s:e] = vc.cpu().numpy()
        pair_counts[s:e] = pc.cpu().numpy()
        done = e / len(points)
        elapsed = time.time() - t0
        eta = elapsed / max(done, 1e-6) - elapsed
        print(f"[photo] {e:,}/{len(points):,} voxels  valid={(P[:e] >= -0.5).mean():.1%}  "
              f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    return P, valid_counts, pair_counts


def analyze_ridges(
    P: np.ndarray,
    res: int,
    spacing: tuple[float, float, float],
    sigma_vox: float,
    p_quantile: float,
    g_quantile: float,
) -> dict[str, np.ndarray | float]:
    from scipy.ndimage import gaussian_filter

    valid = P >= -0.5
    fill = np.where(valid, P, np.nan)
    finite_mean = float(np.nanmean(fill)) if np.isfinite(fill).any() else 0.0
    fill = np.nan_to_num(fill, nan=finite_mean)
    P_grid = fill.reshape(res, res, res)
    valid_grid = valid.reshape(res, res, res)

    P_sigma = gaussian_filter(P_grid, sigma=sigma_vox, mode="nearest")
    gx, gy, gz = np.gradient(P_sigma, *spacing, edge_order=2)
    grad = np.stack([gx, gy, gz], axis=-1)

    hxx, hxy, hxz = np.gradient(gx, *spacing, edge_order=2)
    hyx, hyy, hyz = np.gradient(gy, *spacing, edge_order=2)
    hzx, hzy, hzz = np.gradient(gz, *spacing, edge_order=2)
    H = np.empty((res, res, res, 3, 3), dtype=np.float32)
    H[..., 0, 0] = hxx; H[..., 0, 1] = 0.5 * (hxy + hyx); H[..., 0, 2] = 0.5 * (hxz + hzx)
    H[..., 1, 0] = H[..., 0, 1]; H[..., 1, 1] = hyy; H[..., 1, 2] = 0.5 * (hyz + hzy)
    H[..., 2, 0] = H[..., 0, 2]; H[..., 2, 1] = H[..., 1, 2]; H[..., 2, 2] = hzz

    evals, evecs = np.linalg.eigh(H.reshape(-1, 3, 3))
    lam_min = evals[:, 0].reshape(res, res, res).astype(np.float32)
    e_min = evecs[:, :, 0].reshape(res, res, res, 3).astype(np.float32)
    g = np.einsum("...i,...i->...", grad, e_min).astype(np.float32)

    p_thr = float(np.quantile(P_sigma[valid_grid], p_quantile)) if valid_grid.any() else float("inf")
    base = valid_grid & (P_sigma >= p_thr) & (lam_min < 0.0)
    abs_g = np.abs(g[base])
    g_thr = float(np.quantile(abs_g, g_quantile)) if abs_g.size else 0.0
    candidates = base & (np.abs(g) <= g_thr)

    # centroid of high-score valid voxels → use as default cut plane
    high = valid_grid & (P_sigma >= p_thr)
    if high.any():
        idx = np.argwhere(high)
        # weight by score so the cut lands at the surface peak, not the bbox corner
        w = P_sigma[high]
        centroid_ijk = tuple(int(round(float(np.average(idx[:, d], weights=w)))) for d in range(3))
    else:
        centroid_ijk = (res // 2, res // 2, res // 2)

    return {
        "P_grid": P_grid.astype(np.float32),
        "P_sigma": P_sigma.astype(np.float32),
        "grad": grad.astype(np.float32),
        "lambda_min": lam_min,
        "e_min": e_min,
        "g": g,
        "valid_grid": valid_grid,
        "candidates": candidates,
        "p_thr": p_thr,
        "g_thr": g_thr,
        "centroid_ijk": centroid_ijk,
    }


def render_psigma_3d(
    out_dir: Path,
    P_sigma: np.ndarray,
    valid_grid: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
    top_fraction: float = 0.05,
) -> str | None:
    """Render top-scoring P_sigma voxels as a clean 3D scatter, colored by score.

    Only shows the top `top_fraction` of valid voxels by score, then crops the
    view to a tight box around those points so the skull isn't a dot in a large void.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    valid = valid_grid & (P_sigma > -0.5)
    if not valid.any():
        return None

    scores = P_sigma[valid]
    thresh = float(np.quantile(scores, 1.0 - top_fraction))
    mask = valid & (P_sigma >= thresh)
    idx = np.argwhere(mask)
    pts = bbox_min[None, :] + idx.astype(np.float32) * np.array(spacing)[None, :]
    sc  = P_sigma[mask]
    sc  = (sc - sc.min()) / max(sc.max() - sc.min(), 1e-6)
    col = plt.cm.plasma(sc)[:, :3]

    # tight crop: center on these points + 20% padding
    lo, hi = pts.min(0), pts.max(0)
    pad    = 0.2 * (hi - lo).max() + 1e-6
    center = 0.5 * (lo + hi)
    half   = 0.5 * (hi - lo).max() + pad

    VIEWS = [("front", 0, 0), ("right", 0, 90), ("top", 90, 0), ("iso", 25, 45)]
    BG = "#111111"
    fig = plt.figure(figsize=(9, 9), facecolor=BG)
    fig.suptitle(f"photoconsistency — top {int(top_fraction*100)}% ({len(pts):,} voxels, thresh={thresh:.3f})",
                 color="#aaaaaa", fontsize=9, y=0.98)
    for i, (label, elev, azim) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d", facecolor=BG)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=col, s=3.0, linewidths=0, alpha=0.7, rasterized=True)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, color="#cccccc", fontsize=9, pad=2)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#2a2a2a")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
    fig.tight_layout(pad=0.3)
    path = out_dir / "psigma_3d.png"
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"[render] psigma_3d.png  ({len(pts):,} voxels, thresh={thresh:.3f})", flush=True)
    return "psigma_3d.png"


def _write_point_cloud(path: Path, pts: np.ndarray, colors: np.ndarray | None = None) -> None:
    import trimesh

    if colors is None:
        colors = np.tile(np.array([[60, 180, 255, 255]], dtype=np.uint8), (len(pts), 1))
    trimesh.PointCloud(pts, colors=colors).export(path)


def render_ply_views(out_dir: Path) -> list[str]:
    """Render ridge_candidates.ply and ridge_g0_mesh.ply to clean 3D PNGs.

    Produces candidates_3d.png and mesh_3d.png, each a 2×2 grid of
    (front, right, top, isometric) views.  The mesh is sampled to a shaded
    point cloud so matplotlib's lack of depth-sorting doesn't destroy the render.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    VIEWS = [("front", 0, 0), ("right", 0, 90), ("top", 90, 0), ("iso", 25, 45)]
    BG = "#111111"

    written: list[str] = []

    def _ax_style(ax, center, half, elev, azim, label):
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, color="#cccccc", fontsize=9, pad=2)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#2a2a2a")
        ax.tick_params(colors="#555555", labelsize=5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)

    def _bounds(pts):
        lo, hi = pts.min(0), pts.max(0)
        return 0.5 * (lo + hi), 0.5 * (hi - lo).max() + 1e-6

    def _scatter_fig(pts: np.ndarray, rgba: np.ndarray, path: Path, title: str) -> None:
        if len(pts) == 0:
            return
        center, half = _bounds(pts)
        col = rgba[:, :3] / 255.0 if rgba is not None else None
        fig = plt.figure(figsize=(9, 9), facecolor=BG)
        fig.suptitle(title, color="#aaaaaa", fontsize=10, y=0.98)
        for i, (label, elev, azim) in enumerate(VIEWS, 1):
            ax = fig.add_subplot(2, 2, i, projection="3d", facecolor=BG)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=col, s=1.0, linewidths=0, alpha=0.8, rasterized=True)
            _ax_style(ax, center, half, elev, azim, label)
        fig.tight_layout(pad=0.3)
        fig.savefig(path, dpi=160, facecolor=BG)
        plt.close(fig)

    # --- point cloud — color by ZNCC score via plasma colormap ---
    pc_path = out_dir / "ridge_candidates.ply"
    if pc_path.exists():
        try:
            import trimesh
            pc = trimesh.load(pc_path)
            pts = np.asarray(pc.vertices, dtype=np.float32)
            raw_col = np.asarray(pc.colors) if hasattr(pc, "colors") and pc.colors is not None else None
            if raw_col is not None and raw_col.shape[0] == len(pts):
                # Green channel encodes ZNCC score (80–255 range set at write time)
                score = raw_col[:, 1].astype(np.float32)
                score = (score - score.min()) / max(score.max() - score.min(), 1.0)
                rgba = (cm.plasma(score) * 255).astype(np.uint8)
            else:
                rgba = None
            _scatter_fig(pts, rgba, out_dir / "candidates_3d.png", "ZNCC ridge candidates")
            written.append("candidates_3d.png")
            print(f"[render] candidates_3d.png  ({len(pts):,} pts)", flush=True)
        except Exception as exc:
            print(f"[render] point cloud skipped: {exc}", flush=True)

    # --- mesh — sample surface + shade by normal (avoids Poly3DCollection z-sort mess) ---
    mesh_path = out_dir / "ridge_g0_mesh.ply"
    if mesh_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(mesh_path)
            n_sample = min(80_000, max(10_000, len(mesh.faces) // 4))
            pts_m, face_idx = trimesh.sample.sample_surface(mesh, n_sample)
            pts_m = np.asarray(pts_m, dtype=np.float32)
            # Lambertian shading: two lights for front/back visibility
            fn = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
            light = np.array([0.4, 0.6, 1.0], dtype=np.float32)
            light /= np.linalg.norm(light)
            diff = np.clip(fn @ light, 0.0, 1.0)
            fill = np.clip(-fn @ light * 0.3, 0.0, 1.0)
            shade = 0.18 + 0.72 * diff + fill   # ambient + key + fill
            shade = np.clip(shade, 0.0, 1.0)
            # Map shade → cool blue-white palette
            r = (0.05 + 0.60 * shade * shade)
            g = (0.55 + 0.40 * shade)
            b = np.ones_like(shade)
            rgba_m = (np.stack([r, g, b, np.ones_like(shade)], axis=1) * 255).astype(np.uint8)
            _scatter_fig(pts_m, rgba_m, out_dir / "mesh_3d.png",
                         f"ridge mesh  ({len(mesh.faces):,} faces, {n_sample:,} sampled)")
            written.append("mesh_3d.png")
            print(f"[render] mesh_3d.png  ({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, {n_sample:,} pts sampled)", flush=True)
        except Exception as exc:
            print(f"[render] mesh skipped: {exc}", flush=True)

    return written


def _write_ridge_mesh(
    path: Path,
    g: np.ndarray,
    mask: np.ndarray,
    bbox_min: np.ndarray,
    spacing: tuple[float, float, float],
) -> bool:
    try:
        from skimage.measure import marching_cubes
        import trimesh
    except Exception as exc:
        print(f"[mesh] skipped: {exc}", flush=True)
        return False

    if not mask.any() or g.min() > 0 or g.max() < 0:
        return False

    vol = g.copy()
    outside = ~mask
    vol[outside] = max(float(np.nanmax(np.abs(g))), 1.0)
    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    except ValueError as exc:
        print(f"[mesh] skipped: {exc}", flush=True)
        return False
    verts = verts + bbox_min[None, :]
    trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=False).export(path)
    return True


def _norm01(x: np.ndarray, mask: np.ndarray | None = None, symmetric: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if mask is not None and mask.any():
        vals = x[mask]
    else:
        vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    if symmetric:
        m = float(np.nanpercentile(np.abs(vals), 99.0))
        if m <= 1e-8:
            return np.full_like(x, 0.5, dtype=np.float32)
        return np.clip(0.5 + 0.5 * x / m, 0.0, 1.0)
    lo, hi = np.nanpercentile(vals, [1.0, 99.0])
    if hi - lo <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _upsample_nearest(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    if img.ndim == 2:
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def _write_png(path: Path, img: np.ndarray, scale: int = 1) -> None:
    import imageio.v2 as imageio

    img = _upsample_nearest(img, scale)
    imageio.imwrite(path, np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))


def _overlay(base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    rgb = np.repeat(base[..., None], 3, axis=-1) * 0.8
    if mask.any():
        c = np.asarray(color, dtype=np.float32)
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * c
    return np.clip(rgb, 0.0, 1.0)


def _concat_with_gutters(images: list[np.ndarray], gutter: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    h = max(img.shape[0] for img in images)
    channels = 1 if images[0].ndim == 2 else images[0].shape[2]
    padded = []
    for img in images:
        if img.ndim == 2 and channels == 3:
            img = np.repeat(img[..., None], 3, axis=-1)
        pad_h = h - img.shape[0]
        if pad_h:
            pad_shape = ((0, pad_h), (0, 0)) if img.ndim == 2 else ((0, pad_h), (0, 0), (0, 0))
            img = np.pad(img, pad_shape, constant_values=1.0)
        padded.append(img)
    gut_shape = (h, gutter) if channels == 1 else (h, gutter, channels)
    gutter_img = np.ones(gut_shape, dtype=np.float32)
    out = padded[0]
    for img in padded[1:]:
        out = np.concatenate([out, gutter_img, img], axis=1)
    return out


def write_png_cuts(
    out_dir: Path,
    ridge: dict[str, np.ndarray | float],
    scale: int,
    slice_ijk: tuple[int, int, int] | None = None,
) -> list[str]:
    """Write quick-look central cuts and projections for the score/ridge fields.

    slice_ijk: voxel indices (ix, iy, iz) for cut planes; defaults to volume centre.
    """
    written: list[str] = []
    P = ridge["P_sigma"]
    g = ridge["g"]
    lam = ridge["lambda_min"]
    valid = ridge["valid_grid"]
    cand = ridge["candidates"]
    assert isinstance(P, np.ndarray)
    assert isinstance(g, np.ndarray)
    assert isinstance(lam, np.ndarray)
    assert isinstance(valid, np.ndarray)
    assert isinstance(cand, np.ndarray)

    res = P.shape[0]
    if slice_ijk is not None:
        ix, iy, iz = (int(np.clip(v, 0, res - 1)) for v in slice_ijk)
    else:
        ix = iy = iz = res // 2
    # Per-axis slicers use the individual ix/iy/iz indices, not a single mid.
    axes = [
        ("x", lambda a, _ix=ix: a[_ix, :, :]),
        ("y", lambda a, _iy=iy: a[:, _iy, :]),
        ("z", lambda a, _iz=iz: a[:, :, _iz]),
    ]
    fields = [
        ("P_sigma", P, False),
        ("g", g, True),
        ("lambda_min", lam, True),
        ("candidates", cand.astype(np.float32), False),
    ]
    for name, arr, symmetric in fields:
        mask = valid if name != "candidates" else None
        arr_n = _norm01(arr, mask=mask, symmetric=symmetric)
        for axis, slicer in axes:
            img = np.flipud(slicer(arr_n).T)
            path = out_dir / f"{name}_cut_{axis}.png"
            _write_png(path, img, scale=scale)
            written.append(path.name)

    # Candidate occupancy projections, useful when the exact zero-ridge sheet misses the center cut.
    for axis, proj in [
        ("x", cand.max(axis=0)),
        ("y", cand.max(axis=1)),
        ("z", cand.max(axis=2)),
    ]:
        path = out_dir / f"candidates_proj_{axis}.png"
        _write_png(path, np.flipud(proj.astype(np.float32).T), scale=scale)
        written.append(path.name)

    # Human-readable contact sheets: P in gray with candidates overlaid in cyan.
    overview_tiles = []
    score_tiles = []
    for axis, slicer in axes:
        p_cut = slicer(P)
        v_cut = slicer(valid)
        c_cut = slicer(cand)
        p_img = np.flipud(_norm01(p_cut, mask=v_cut).T)
        c_img = np.flipud(c_cut.T.astype(bool))
        score_tiles.append(p_img)
        overview_tiles.append(_overlay(p_img, c_img, (0.05, 0.85, 1.0)))
    overview = _concat_with_gutters(overview_tiles)
    score_sheet = _concat_with_gutters(score_tiles)
    path = out_dir / "overview_cuts.png"
    _write_png(path, overview, scale=scale)
    written.append(path.name)
    path = out_dir / "P_sigma_cuts.png"
    _write_png(path, score_sheet, scale=scale)
    written.append(path.name)

    proj_tiles = []
    for proj in [cand.max(axis=0), cand.max(axis=1), cand.max(axis=2)]:
        proj_tiles.append(np.flipud(proj.astype(np.float32).T))
    path = out_dir / "candidates_projections.png"
    _write_png(path, _concat_with_gutters(proj_tiles), scale=scale)
    written.append(path.name)
    return written


def _try_load_sfm_centroid(scene: Path | None) -> np.ndarray | None:
    """Return mean of COLMAP sparse points if available, else None."""
    if scene is None:
        return None
    sfm_path = scene / "sparse_sfm_points.txt"
    if not sfm_path.exists():
        return None
    try:
        pts = np.loadtxt(sfm_path, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None]
        centroid = pts.mean(axis=0)
        print(f"[auto-center] COLMAP centroid = {centroid.tolist()}", flush=True)
        return centroid
    except Exception:
        return None


def _resolve_slice_ijk(
    slice_xyz_str: str | None,
    sfm_centroid: np.ndarray | None,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    res: int,
) -> tuple[int, int, int] | None:
    """Convert a world-space point to voxel indices for cut planes."""
    def _world_to_ijk(xyz: np.ndarray) -> tuple[int, int, int]:
        frac = (xyz - bbox_min) / np.maximum(bbox_max - bbox_min, 1e-8)
        ijk = (frac * (res - 1)).round().astype(int)
        return tuple(int(np.clip(v, 0, res - 1)) for v in ijk)

    if slice_xyz_str is not None:
        vals = np.fromstring(slice_xyz_str, sep=",", dtype=np.float32)
        if vals.shape == (3,):
            return _world_to_ijk(vals)
        print(f"[warn] --slice-xyz expects 3 floats, got '{slice_xyz_str}' — using centroid", flush=True)
    if sfm_centroid is not None:
        return _world_to_ijk(sfm_centroid)
    return None   # caller falls back to res // 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from_npz", type=Path, default=None,
                    help="regenerate ridge analysis/PNGs from an existing photo_ridge_volume.npz")
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--dataset", choices=["dtu", "lego"], default="dtu")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/photo_ridge_viz"))
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--bound", type=float, default=1.5, help="symmetric grid bound if --bbox is omitted")
    ap.add_argument("--bbox", type=str, default=None,
                    help="xmin,ymin,zmin,xmax,ymax,zmax; overrides --bound")
    ap.add_argument("--views", type=str, default=None,
                    help="comma list or python-like slice, e.g. '0,8,16' or '0:49:2'")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--patch-radius", type=int, default=3)
    ap.add_argument("--patch-dilation", type=float, default=1.0)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--agg", choices=["median", "mean", "mean-topk"], default="median")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--sigma-vox", type=float, default=1.0)
    ap.add_argument("--p-quantile", type=float, default=0.70,
                    help="top-P fraction of voxels kept as ridge candidates (was 0.80)")
    ap.add_argument("--g-quantile", type=float, default=0.15,
                    help="gradient-near-zero fraction for ridge (was 0.08)")
    ap.add_argument("--slice-xyz", type=str, default=None,
                    help="x,y,z world coords for PNG cut planes (default: COLMAP centroid or bbox centre)")
    ap.add_argument("--png-scale", type=int, default=6,
                    help="nearest-neighbor upsampling factor for diagnostic PNG cuts")
    ap.add_argument("--mask-mode", choices=["foreground", "none"], default="foreground")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_npz is not None:
        data = np.load(args.from_npz)
        P_existing = data["P"].astype(np.float32).reshape(-1)
        bbox_min = data["bbox_min"].astype(np.float32)
        bbox_max = data["bbox_max"].astype(np.float32)
        spacing = tuple(float(x) for x in data["spacing"])
        res = int(round(P_existing.shape[0] ** (1.0 / 3.0)))
        if res ** 3 != P_existing.shape[0]:
            raise ValueError(f"cannot infer cubic grid resolution from {args.from_npz}")
        ridge = analyze_ridges(P_existing, res, spacing, args.sigma_vox, args.p_quantile, args.g_quantile)
        slice_ijk = _resolve_slice_ijk(args.slice_xyz, None, bbox_min, bbox_max, res)
        if slice_ijk is None:
            slice_ijk = ridge["centroid_ijk"]
            print(f"[cuts] auto-centroid at voxel {slice_ijk}", flush=True)
        pngs = write_png_cuts(args.out_dir, ridge, max(args.png_scale, 1), slice_ijk=slice_ijk)
        r = render_psigma_3d(args.out_dir, ridge["P_sigma"], ridge["valid_grid"], bbox_min, spacing)
        if r:
            pngs.append(r)
        meta = {
            "from_npz": str(args.from_npz),
            "res": res,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "spacing": list(spacing),
            "sigma_vox": args.sigma_vox,
            "p_thr": ridge["p_thr"],
            "g_thr": ridge["g_thr"],
            "png_scale": int(max(args.png_scale, 1)),
            "candidate_points": int(ridge["candidates"].sum()),
            "pngs": pngs,
        }
        render3d = render_ply_views(args.out_dir)
        pngs += render3d
        meta["pngs"] = pngs
        (args.out_dir / "meta_regen.json").write_text(json.dumps(meta, indent=2))
        print(f"[done] regenerated {len(pngs)} png cuts/projections + 3d renders from {args.from_npz}", flush=True)
        return

    if args.scene is None:
        raise ValueError("--scene is required unless --from_npz is used")
    views = load_blender_views(args.scene, down=1) if args.dataset == "lego" else load_views(args.scene)
    ids = _parse_ints(args.views, views["c2w"].shape[0])
    if len(ids) < args.min_views:
        raise ValueError(f"need at least {args.min_views} selected views, got {len(ids)}")
    for k in ("images", "masks", "c2w", "K"):
        if k in views:
            views[k] = views[k][ids]
    print(f"[load] scene={args.scene} views={len(ids)} H={views['H']} W={views['W']} device={args.device}", flush=True)

    if args.bbox:
        vals = np.fromstring(args.bbox, sep=",", dtype=np.float32)
        if vals.shape != (6,):
            raise ValueError("--bbox must contain six comma-separated floats")
        bbox_min, bbox_max = vals[:3], vals[3:]
    else:
        bbox_min = np.array([-args.bound, -args.bound, -args.bound], dtype=np.float32)
        bbox_max = np.array([+args.bound, +args.bound, +args.bound], dtype=np.float32)

    points, spacing = _make_grid(args.res, bbox_min, bbox_max)
    print(f"[grid] res={args.res} voxels={len(points):,} bbox={bbox_min.tolist()}..{bbox_max.tolist()}", flush=True)

    P, valid_counts, pair_counts = compute_photoconsistency(
        views, points, args.device, args.chunk, args.patch_radius, args.patch_dilation,
        args.min_views, args.max_views, args.agg, args.topk, args.mask_mode,
    )
    ridge = analyze_ridges(P, args.res, spacing, args.sigma_vox, args.p_quantile, args.g_quantile)

    # Determine cut planes: --slice-xyz > COLMAP centroid > score-weighted centroid > bbox centre
    sfm_pts = _try_load_sfm_centroid(args.scene)
    slice_ijk = _resolve_slice_ijk(args.slice_xyz, sfm_pts, bbox_min, bbox_max, args.res)
    if slice_ijk is None:
        slice_ijk = ridge["centroid_ijk"]
        print(f"[cuts] auto-centroid at voxel {slice_ijk}", flush=True)
    else:
        ix, iy, iz = slice_ijk
        print(f"[cuts] slice at voxel ({ix},{iy},{iz})", flush=True)

    cand_idx = np.argwhere(ridge["candidates"])
    cand_pts = bbox_min[None, :] + cand_idx.astype(np.float32) * np.array(spacing, dtype=np.float32)[None, :]
    p_vals = ridge["P_sigma"][ridge["candidates"]]
    if len(cand_pts):
        p_norm = (p_vals - p_vals.min()) / max(float(p_vals.max() - p_vals.min()), 1e-6)
        colors = np.stack([
            np.full_like(p_norm, 30),
            (80 + 175 * p_norm),
            np.full_like(p_norm, 255),
            np.full_like(p_norm, 255),
        ], axis=-1).astype(np.uint8)
    else:
        colors = None
    _write_point_cloud(args.out_dir / "ridge_candidates.ply", cand_pts, colors)
    mesh_ok = _write_ridge_mesh(
        args.out_dir / "ridge_g0_mesh.ply",
        ridge["g"],
        ridge["valid_grid"] & (ridge["P_sigma"] >= ridge["p_thr"]) & (ridge["lambda_min"] < 0),
        bbox_min,
        spacing,
    )
    pngs = write_png_cuts(args.out_dir, ridge, max(args.png_scale, 1), slice_ijk=slice_ijk)
    r = render_psigma_3d(args.out_dir, ridge["P_sigma"], ridge["valid_grid"], bbox_min, spacing)
    if r:
        pngs.append(r)

    np.savez_compressed(
        args.out_dir / "photo_ridge_volume.npz",
        P=P.reshape(args.res, args.res, args.res),
        valid_counts=valid_counts.reshape(args.res, args.res, args.res),
        pair_counts=pair_counts.reshape(args.res, args.res, args.res),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        spacing=np.asarray(spacing, dtype=np.float32),
        **{k: v for k, v in ridge.items() if isinstance(v, np.ndarray)},
    )
    meta = {
        "scene": str(args.scene),
        "dataset": args.dataset,
        "views": ids,
        "res": args.res,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "spacing": list(spacing),
        "patch_radius": args.patch_radius,
        "patch_dilation": args.patch_dilation,
        "min_views": args.min_views,
        "max_views": args.max_views,
        "agg": args.agg,
        "topk": args.topk,
        "sigma_vox": args.sigma_vox,
        "p_thr": ridge["p_thr"],
        "g_thr": ridge["g_thr"],
        "png_scale": int(max(args.png_scale, 1)),
        "candidate_points": int(len(cand_pts)),
        "mesh_written": bool(mesh_ok),
        "pngs": pngs,
    }
    render3d = render_ply_views(args.out_dir)
    pngs += render3d
    meta["pngs"] = pngs
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {args.out_dir}/photo_ridge_volume.npz", flush=True)
    print(f"[done] wrote {args.out_dir}/ridge_candidates.ply ({len(cand_pts):,} pts)", flush=True)
    if mesh_ok:
        print(f"[done] wrote {args.out_dir}/ridge_g0_mesh.ply", flush=True)
    print(f"[done] wrote {len(pngs)} png cuts/projections + 3d renders", flush=True)


if __name__ == "__main__":
    main()
