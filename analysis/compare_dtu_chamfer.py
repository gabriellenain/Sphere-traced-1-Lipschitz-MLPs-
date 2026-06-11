#!/usr/bin/env python3
"""DTU-style Chamfer comparison against the official GT point cloud.

Inputs are either a repo checkpoint, extracted in normalised SDF coordinates,
or a mesh. Predicted points are mapped into DTU world coordinates, filtered by
the official ObsMask, then compared to the filtered DTU GT point cloud with
bidirectional nearest-neighbour distances.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch


def _infer_scan_id(scene: Path) -> int:
    m = re.search(r"scan(\d+)", str(scene))
    if not m:
        raise ValueError(f"Could not infer scan id from scene path: {scene}")
    return int(m.group(1))


def _checkpoint_run_config(ckpt_path: Path) -> dict | None:
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text())
    except Exception as exc:
        print(f"[warn] could not read checkpoint run config {config_path}: {exc}", flush=True)
        return None


def _is_refined_checkpoint(ckpt_path: Path) -> bool:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return "psi" in ckpt and "coarse_pt" in ckpt


def _resolve_coarse_pt(ckpt_path: Path) -> Path:
    """Return the coarse checkpoint path embedded in a refined checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    coarse = Path(ckpt["coarse_pt"])
    if coarse.is_absolute():
        candidates = [coarse]
    else:
        candidates = [Path.cwd() / coarse, ckpt_path.parent / coarse]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Refined checkpoint references coarse_pt={ckpt['coarse_pt']!r} which does not exist "
        f"(tried: {candidates}). Pass --ckpt and --psi explicitly."
    )


def _resolve_trained_scene(ckpt_path: Path) -> Path | None:
    """Return the scene path stored in the checkpoint's config.json, or None if unavailable."""
    ref = ckpt_path
    if _is_refined_checkpoint(ckpt_path):
        try:
            ref = _resolve_coarse_pt(ckpt_path)
        except FileNotFoundError:
            return None
    cfg = _checkpoint_run_config(ref)
    if not cfg:
        return None
    trained_scene = cfg.get("scene")
    return Path(trained_scene) if trained_scene else None


def _extract_mesh_from_model(ckpt_path: Path, bound: float, res: int, device: str):
    from skimage.measure import marching_cubes

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    architecture = ckpt.get("architecture", "cpl")
    hidden = ckpt.get("hidden", 256)
    depth = ckpt.get("depth", 8)
    group_size = ckpt.get("group_size", 2)
    activation = ckpt.get("activation", "groupsort")
    input_enc = ckpt.get("input_encoding", "identity")
    if input_enc == "neus":
        input_enc = "pe"
    multires = ckpt.get("multires", 6)
    for k, v in state.items():
        if "weight" in k and v.ndim >= 2:
            hidden = v.shape[-1]
            break

    from lip_tracer.model import make_model
    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_enc,
                   multires=multires, architecture=architecture)
    f.load_state_dict(state, strict=False)
    f = f.to(device).eval()

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    chunk = 65536 if device == "cpu" else 4096
    n_total = len(grid)
    n_chunks = (n_total + chunk - 1) // chunk
    log_every = max(1, n_chunks // 20)
    print(f"       evaluating SDF on {n_total:,} grid points "
          f"({n_chunks} chunks of {chunk}, device={device})", flush=True)
    out_chunks = []
    t_eval = time.time()
    with torch.no_grad():
        for ci, i in enumerate(range(0, n_total, chunk)):
            out_chunks.append(f(grid[i:i + chunk]))
            if (ci + 1) % log_every == 0 or (ci + 1) == n_chunks:
                done = min(i + chunk, n_total)
                elapsed = time.time() - t_eval
                eta = elapsed * (n_total / done - 1) if done else 0.0
                print(f"         [{done:>10,} / {n_total:,}] "
                      f"{100.0 * done / n_total:5.1f}%  "
                      f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s", flush=True)
    vals = torch.cat(out_chunks)
    vol = vals.reshape(res, res, res).cpu().numpy()
    del f, grid, vals, out_chunks
    if device != "cpu":
        torch.cuda.empty_cache()
    print(f"       SDF eval done in {time.time() - t_eval:.1f}s; running marching cubes", flush=True)
    if vol.min() > 0 or vol.max() < 0:
        raise ValueError(
            f"No zero-crossing in SDF grid (min={vol.min():.3f}, max={vol.max():.3f}). "
            "Try adjusting --bound."
        )

    spacing = 2 * bound / (res - 1)
    verts, faces, normals, _ = marching_cubes(
        vol, level=0.0, spacing=(spacing,) * 3, gradient_direction="ascent"
    )
    return (verts - bound).astype(np.float32), faces.astype(np.int64), normals.astype(np.float32)


def _extract_mesh_from_refined(ckpt_path: Path, psi_path: Path,
                               bound: float, res: int, device: str):
    """Marching cubes on g(x) = fθ(x) - δψ(x), the deformed surface implicit."""
    from skimage.measure import marching_cubes
    from lip_tracer.model import make_model
    from lip_tracer.deformation_field import DeformationField

    # load coarse fθ
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    architecture = ckpt.get("architecture", "cpl")
    group_size   = ckpt.get("group_size", 2)
    activation   = ckpt.get("activation", "groupsort")
    input_enc    = ckpt.get("input_encoding", "identity")
    if input_enc == "neus":
        input_enc = "pe"
    multires = ckpt.get("multires", 6)
    state = ckpt["f"]
    hidden = next((v.shape[-1] for k, v in state.items()
                   if "weight" in k and v.ndim >= 2), 256)
    depth  = ckpt.get("depth", 8)
    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_enc,
                   multires=multires, architecture=architecture)
    f.load_state_dict(state, strict=False)
    f = f.to(device).eval()

    # load δψ
    psi_ckpt  = torch.load(psi_path, map_location="cpu", weights_only=False)
    delta_max = psi_ckpt.get("delta_max", 0.01)
    psi = DeformationField(delta_max=delta_max)
    psi.load_state_dict(psi_ckpt["psi"])
    psi = psi.to(device).eval()

    vox  = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    chunk = 4096
    out_chunks = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(grid), chunk):
            x = grid[i:i + chunk]
            out_chunks.append(f(x) - psi(x))
    vals = torch.cat(out_chunks)
    vol  = vals.reshape(res, res, res).cpu().numpy()
    del f, psi, grid, vals, out_chunks
    if device != "cpu":
        torch.cuda.empty_cache()
    print(f"       refined SDF eval done in {time.time() - t0:.1f}s; running marching cubes",
          flush=True)
    if vol.min() > 0 or vol.max() < 0:
        raise ValueError(
            f"No zero-crossing in refined SDF grid (min={vol.min():.3f}, max={vol.max():.3f}). "
            "Try adjusting --bound."
        )
    spacing = 2 * bound / (res - 1)
    verts, faces, normals, _ = marching_cubes(
        vol, level=0.0, spacing=(spacing,) * 3, gradient_direction="ascent"
    )
    return (verts - bound).astype(np.float32), faces.astype(np.int64), normals.astype(np.float32)


def _load_mesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh found in {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    return (np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int64),
            np.asarray(mesh.vertex_normals, dtype=np.float32))


def _sample_surface(verts: np.ndarray, faces: np.ndarray, n: int, seed: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if n <= 0:
        return verts.astype(np.float32)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return pts.astype(np.float32)


def _to_world(pts: np.ndarray, scale_mat: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (scale_mat @ pts_h.T).T[:, :3].astype(np.float32)


def _load_point_cloud(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    except ImportError:
        import trimesh
        pc = trimesh.load(str(path), process=False)
        if hasattr(pc, "vertices"):
            return np.asarray(pc.vertices, dtype=np.float32)
        if hasattr(pc, "vertices"):
            return np.asarray(pc.vertices, dtype=np.float32)
        raise


def _load_dtu_gt(dtu_eval_dir: Path, scan_id: int, gt_ply: Path | None = None):
    from scipy.io import loadmat

    ply_path = gt_ply if gt_ply is not None else dtu_eval_dir / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
    if not ply_path.exists():
        available = sorted((dtu_eval_dir / "Points" / "stl").glob("stl*_total.ply"))
        names = ", ".join(p.name for p in available[:20])
        more = " ..." if len(available) > 20 else ""
        raise FileNotFoundError(
            f"GT PLY not found for scan{scan_id}: {ply_path}\n"
            f"Expected the scan-specific DTU cloud, e.g. stl{scan_id:03d}_total.ply. "
            f"Available under {dtu_eval_dir / 'Points' / 'stl'}: {names}{more}"
        )

    mat_path = dtu_eval_dir / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"ObsMask not found: {mat_path}")

    mat = loadmat(str(mat_path))
    obs_mask = mat["ObsMask"].astype(bool)
    bb = mat["BB"].astype(np.float64)
    res = float(mat["Res"].flat[0])
    all_pts = _load_point_cloud(ply_path)
    gt_pts = all_pts[_in_obs(all_pts, obs_mask, bb, res)]
    return gt_pts, {"ply": str(ply_path), "ObsMask": obs_mask, "BB": bb, "Res": res,
                    "n_gt_total": int(len(all_pts))}


def _in_obs(pts: np.ndarray, obs_mask: np.ndarray, bb: np.ndarray, res: float) -> np.ndarray:
    in_bb = np.all((pts >= bb[0]) & (pts <= bb[1]), axis=1)
    idx = np.clip(np.round((pts - bb[0]) / res).astype(int), 0, np.array(obs_mask.shape) - 1)
    return in_bb & obs_mask[idx[:, 0], idx[:, 1], idx[:, 2]]


def _mask_filter_gt(
    gt_pts: np.ndarray,
    scene_dir: Path,
    mode: str = "any",
    min_ratio: float = 0.95,
    min_views: int = 1,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Filter GT points by reprojection into IDR foreground masks.

    mode="any" matches the historical behavior: keep a point if it lands in
    foreground in at least min_views masks. For object-only eval, mode="ratio"
    or "all" is usually cleaner because it approximates a visual-hull test.
    """
    from PIL import Image

    cam_dict = np.load(scene_dir / "cameras.npz")
    mask_dir = scene_dir / "mask"
    mask_paths = sorted(p for p in mask_dir.glob("*.png") if not p.name.startswith("."))
    if not mask_paths:
        raise FileNotFoundError(f"No mask images found in {mask_dir}")

    fg_counts = np.zeros(len(gt_pts), dtype=np.uint16)
    seen_counts = np.zeros(len(gt_pts), dtype=np.uint16)
    gt_h = np.concatenate([gt_pts.astype(np.float64),
                           np.ones((len(gt_pts), 1), dtype=np.float64)], axis=1).T  # 4×N

    for i, mask_path in enumerate(mask_paths):
        P_key = f"world_mat_{i}"
        if P_key not in cam_dict:
            continue
        P = cam_dict[P_key].astype(np.float64)      # 4×4 projection P=K[R|t]
        mask = np.array(Image.open(mask_path))       # H×W or H×W×3
        H, W = mask.shape[:2]

        proj = P @ gt_h                              # 4×N
        w = proj[2]
        pos = w > 0
        u = np.where(pos, proj[0] / np.where(pos, w, 1.0), -1.0)
        v = np.where(pos, proj[1] / np.where(pos, w, 1.0), -1.0)
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        in_bounds = pos & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        idx_v = np.where(in_bounds)[0]
        if not len(idx_v):
            continue
        seen_counts[idx_v] += 1
        m = mask[vi[idx_v], ui[idx_v]]
        vals = m[..., 0] if mask.ndim == 3 else m
        fg_counts[idx_v[vals > 0]] += 1

    min_views = max(1, int(min_views))
    seen_ok = seen_counts >= min_views
    if mode == "any":
        keep = fg_counts >= min_views
    elif mode == "all":
        keep = seen_ok & (fg_counts == seen_counts)
    elif mode == "ratio":
        ratio = np.divide(
            fg_counts,
            np.maximum(seen_counts, 1),
            out=np.zeros(len(gt_pts), dtype=np.float32),
            where=seen_counts > 0,
        )
        keep = seen_ok & (ratio >= float(min_ratio))
    else:
        raise ValueError(f"unknown GT mask filter mode: {mode}")

    stats = {
        "mode": mode,
        "min_ratio": float(min_ratio),
        "min_views": int(min_views),
        "n_projected_min_views": int(seen_ok.sum()),
        "fg_views_mean": float(fg_counts.mean()) if len(fg_counts) else 0.0,
        "seen_views_mean": float(seen_counts.mean()) if len(seen_counts) else 0.0,
    }
    return keep, stats


def _crop_to_bbox(pts: np.ndarray, ref: np.ndarray, padding: float) -> tuple[np.ndarray, dict[str, list[float]]]:
    lo = ref.min(axis=0) - padding
    hi = ref.max(axis=0) + padding
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    return keep, {"min": lo.tolist(), "max": hi.tolist(), "padding": float(padding)}


def _save_point_cloud(path: Path, pts: np.ndarray) -> None:
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        o3d.io.write_point_cloud(str(path), pc)
    except ImportError:
        import trimesh
        trimesh.PointCloud(pts).export(path)


def _nn_distances_gpu(query: np.ndarray, ref: np.ndarray, device: str, chunk: int = 512) -> np.ndarray:
    """Chunked nearest-neighbour distances on GPU using torch.cdist.
    chunk=512 keeps each distance matrix ≤ ~4 GB for typical DTU GT sizes (~1.8 M ref pts).
    Falls back to CPU cKDTree on OOM."""
    try:
        q = torch.from_numpy(query).to(device)
        r = torch.from_numpy(ref).to(device)
        dists = []
        with torch.no_grad():
            for i in range(0, len(q), chunk):
                d = torch.cdist(q[i:i + chunk], r)   # (C, N)
                dists.append(d.min(dim=1).values.cpu())
                del d
        return torch.cat(dists).numpy()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"       [warn] GPU OOM at chunk={chunk}, falling back to CPU cKDTree", flush=True)
        from scipy.spatial import cKDTree
        d, _ = cKDTree(ref).query(query, k=1, workers=-1)
        return d.astype(np.float32)


def _nn_metrics(pred: np.ndarray, gt: np.ndarray, max_dist: float | None,
                device: str = "cpu") -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    if device != "cpu" and torch.cuda.is_available():
        acc = _nn_distances_gpu(pred, gt, device)
        comp = _nn_distances_gpu(gt, pred, device)
    else:
        from scipy.spatial import cKDTree
        acc, _ = cKDTree(gt).query(pred, k=1, workers=-1)
        comp, _ = cKDTree(pred).query(gt, k=1, workers=-1)
    acc = acc.astype(np.float32)
    comp = comp.astype(np.float32)
    if max_dist is not None:
        acc_clip = np.minimum(acc, max_dist)
        comp_clip = np.minimum(comp, max_dist)
    else:
        acc_clip, comp_clip = acc, comp
    accuracy = float(acc_clip.mean())
    completeness = float(comp_clip.mean())
    metrics = {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer": 0.5 * (accuracy + completeness),
        "acc_p50": float(np.median(acc)),
        "acc_p90": float(np.percentile(acc, 90)),
        "comp_p50": float(np.median(comp)),
        "comp_p90": float(np.percentile(comp, 90)),
        "hausdorff": float(max(acc.max(), comp.max())),
        "max_dist_clip": float(max_dist) if max_dist is not None else None,
    }
    return metrics, acc, comp


def _pca_align(pts: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation R such that R @ pts.T has PC1→col0, PC2→col1, PC3→col2.
    The longest axis (usually height for a bust) is mapped to the Y column so the
    XY projection shows a natural front view."""
    c = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)   # rows of Vt are PCs
    # PCs in descending variance order: Vt[0]=largest, Vt[1], Vt[2]=smallest
    # We want: depth→col2 (Z, into screen), height→col1 (Y), width→col0 (X)
    # So map PC0→Y, PC1→X, PC2→Z  ⟹  R columns = [PC1, PC0, PC2]
    R = np.stack([Vt[1], Vt[0], Vt[2]], axis=1).astype(np.float32)  # (3,3)
    # ensure right-handed
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


def _render_error_png(pred: np.ndarray, acc_dist: np.ndarray,
                      gt: np.ndarray, comp_dist: np.ndarray,
                      out_path: Path, vmax: float | None = None,
                      align_pca: bool = False,
                      rotate_deg: float = 0.0) -> None:
    """Save a visual error map: 3×4 grid (3 projections + histogram), rows=accuracy/completeness/overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    plt.style.use("dark_background")

    CMAP = "plasma"
    BG   = "#0d0d0d"
    rng  = np.random.default_rng(0)
    MAX_PTS = 300_000

    def _sub(pts, d, n=MAX_PTS):
        if len(pts) > n:
            idx = rng.choice(len(pts), n, replace=False)
            return pts[idx], d[idx]
        return pts, d

    pred_s, acc_s   = _sub(pred, acc_dist)
    gt_s,   comp_s  = _sub(gt,   comp_dist)

    if align_pca or rotate_deg != 0.0:
        _all = np.concatenate([pred_s, gt_s], axis=0)
        _mean = _all.mean(0)
        _R = _pca_align(_all) if align_pca else np.eye(3, dtype=np.float32)
        if rotate_deg != 0.0:
            th = np.deg2rad(rotate_deg)
            c, s = np.cos(th), np.sin(th)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            _R = _R @ Rz
        def _rot(p): return (p - _mean) @ _R
        pred_s = _rot(pred_s);  gt_s = _rot(gt_s)
        PROJ = [(0, 1, "width", "height"), (0, 2, "width", "depth"), (1, 2, "height", "depth")]
    else:
        _rot = None
        PROJ = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]

    vm = vmax if vmax is not None else max(
        float(np.percentile(acc_s,  95)),
        float(np.percentile(comp_s, 95)), 1e-3)

    norm = Normalize(vmin=0, vmax=vm)
    sm_acc  = ScalarMappable(cmap=CMAP, norm=norm)
    sm_comp = ScalarMappable(cmap=CMAP, norm=norm)

    fig = plt.figure(figsize=(22, 15), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # 3 rows × 4 cols; last col is histograms
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.32,
                           width_ratios=[1, 1, 1, 0.85])

    row_labels = ["ACCURACY  pred→GT", "COMPLETENESS  GT→pred", "OVERLAY  pred=blue  GT=orange"]

    # ── rows 0 & 1: error-coloured projections ──────────────────────────────
    for row, (pts, dists, sm) in enumerate([(pred_s, acc_s, sm_acc),
                                             (gt_s,  comp_s, sm_comp)]):
        for col, (i, j, xl, yl) in enumerate(PROJ):
            ax = fig.add_subplot(gs[row, col], facecolor=BG)
            order = np.argsort(dists)          # draw low-error points last → readable
            sc = ax.scatter(pts[order, i], pts[order, j],
                            c=dists[order], cmap=CMAP, norm=norm,
                            s=0.4, linewidths=0, alpha=0.85, rasterized=True)
            ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\n{yl}", fontsize=8, labelpad=4)
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("mm", fontsize=7); cb.ax.tick_params(labelsize=7)

    # ── row 2: overlap overlay ───────────────────────────────────────────────
    pred_ov, _ = _sub(pred, acc_dist, n=150_000)
    gt_ov,   _ = _sub(gt,  comp_dist, n=150_000)
    if _rot is not None:
        pred_ov = _rot(pred_ov);  gt_ov = _rot(gt_ov)
    for col, (i, j, xl, yl) in enumerate(PROJ):
        ax = fig.add_subplot(gs[2, col], facecolor=BG)
        ax.scatter(gt_ov[:,   i], gt_ov[:,   j], s=0.3, color="#ff7f0e",
                   alpha=0.5, linewidths=0, rasterized=True, label="GT")
        ax.scatter(pred_ov[:, i], pred_ov[:, j], s=0.3, color="#1f77b4",
                   alpha=0.6, linewidths=0, rasterized=True, label="pred")
        ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8); ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(f"{row_labels[2]}\n{yl}", fontsize=8, labelpad=4)
            ax.legend(markerscale=8, fontsize=7, loc="upper right",
                      framealpha=0.3, facecolor=BG)

    # ── col 3: distance histograms ───────────────────────────────────────────
    BINS = 120
    def _hist_ax(ax, d, color, label, mean, p50, p90):
        ax.set_facecolor(BG)
        clip = min(float(d.max()), vm * 3)
        bins = np.linspace(0, clip, BINS)
        ax.hist(d[d <= clip], bins=bins, color=color, alpha=0.75, density=True)
        for v, ls, lbl in [(mean, "--", f"mean {mean:.2f}"),
                            (p50,  ":",  f"p50  {p50:.2f}"),
                            (p90,  "-.", f"p90  {p90:.2f}")]:
            ax.axvline(v, color="white", lw=0.9, ls=ls, alpha=0.8, label=lbl)
        ax.axvline(vm, color="#ff4444", lw=1.0, alpha=0.9, label=f"vmax {vm:.2f}")
        ax.set_xlabel("distance (mm)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(label, fontsize=9, pad=4)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, framealpha=0.25, facecolor=BG)

    ax_ha = fig.add_subplot(gs[0, 3], facecolor=BG)
    _hist_ax(ax_ha, acc_s,  "#00c8ff", "Accuracy distribution",
             float(acc_s.mean()), float(np.median(acc_s)), float(np.percentile(acc_s, 90)))

    ax_hc = fig.add_subplot(gs[1, 3], facecolor=BG)
    _hist_ax(ax_hc, comp_s, "#ff9900", "Completeness distribution",
             float(comp_s.mean()), float(np.median(comp_s)), float(np.percentile(comp_s, 90)))

    # ── summary text in [2,3] ────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[2, 3], facecolor=BG)
    ax_txt.axis("off")
    chamfer = 0.5 * (float(acc_s.mean()) + float(comp_s.mean()))
    summary = (
        f"accuracy    {float(acc_s.mean()):.4f}\n"
        f"completeness {float(comp_s.mean()):.4f}\n"
        f"chamfer     {chamfer:.4f}\n\n"
        f"acc  p50  {float(np.median(acc_s)):.3f}\n"
        f"acc  p90  {float(np.percentile(acc_s, 90)):.3f}\n"
        f"comp p50  {float(np.median(comp_s)):.3f}\n"
        f"comp p90  {float(np.percentile(comp_s, 90)):.3f}\n\n"
        f"pred pts  {len(pred):,}\n"
        f"gt   pts  {len(gt):,}\n"
        f"vmax      {vm:.2f} mm"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=10, va="top", ha="left", family="monospace",
                color="white", linespacing=1.7)

    fig.suptitle(
        f"DTU Chamfer  ·  acc={float(acc_s.mean()):.3f}  comp={float(comp_s.mean()):.3f}"
        f"  chamfer={chamfer:.3f}  (colour scale 0–{vm:.2f} mm)",
        fontsize=13, y=0.995, color="white")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.style.use("default")
    plt.close(fig)
    print(f"     error map saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path)
    src.add_argument("--mesh", type=Path)
    ap.add_argument("--psi", type=Path, default=None,
                    help="refined deformation field .pt (requires --ckpt)")
    ap.add_argument("--mesh-space", choices=["normalized", "world"], default="normalized")
    ap.add_argument("--scene", type=Path, default=None,
                    help="DTU scene directory (contains cameras.npz). When --ckpt is given "
                         "the trained scene is always read from config.json and this value "
                         "serves only as a fallback if config.json is missing.")
    ap.add_argument("--scan-id", type=int, default=None)
    ap.add_argument("--dtu-eval-dir", type=Path, required=True)
    ap.add_argument("--gt-ply", type=Path, default=None,
                    help="override the scan-specific GT cloud path")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=0.8)
    ap.add_argument("--n-pred-points", type=int, default=1_000_000)
    ap.add_argument("--max-dist", type=float, default=20.0,
                    help="DTU-style max distance (mm) clip for accuracy/completeness means; "
                         "set <=0 to disable")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--gt-crop", choices=["none", "pred-bbox"], default="none",
                    help="optionally crop GT after ObsMask filtering for object-focused evaluation")
    ap.add_argument("--gt-crop-padding", type=float, default=0.0,
                    help="world-unit padding around the crop region")
    ap.add_argument("--gt-max-dist-to-pred", type=float, default=None,
                    help="keep only GT points within this distance of the predicted surface")
    ap.add_argument("--gt-z-min", type=float, default=None,
                    help="discard GT points with world-space Z below this value (removes floor/pedestal)")
    ap.add_argument("--gt-z-max", type=float, default=None,
                    help="discard GT points with world-space Z above this value")
    ap.add_argument("--gt-mask-filter", action="store_true",
                    help="keep only GT points visible as foreground in ≥1 training mask "
                         "(scene/mask/*.png + cameras.npz); removes pedestal/background "
                         "analogously to how mask-supervised methods train")
    ap.add_argument("--gt-mask-mode", choices=["any", "ratio", "all"], default="any",
                    help="foreground mask rule for --gt-mask-filter. 'any' preserves the "
                         "old permissive behavior; 'ratio'/'all' are stricter object-only "
                         "visual-hull-like filters.")
    ap.add_argument("--gt-mask-min-ratio", type=float, default=0.95,
                    help="minimum foreground/projected-view ratio for --gt-mask-mode ratio")
    ap.add_argument("--gt-mask-min-views", type=int, default=1,
                    help="minimum projected views for ratio/all, or minimum foreground "
                         "views for any")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-pred-mesh", action="store_true")
    ap.add_argument("--save-eval-clouds", action="store_true",
                    help="save the predicted and GT point clouds used by the metric")
    ap.add_argument("--save-png", action="store_true",
                    help="save a 2x3 error-coloured PNG of accuracy and completeness")
    ap.add_argument("--png-vmax", type=float, default=None,
                    help="colour scale maximum (mm); defaults to 95th percentile of distances")
    ap.add_argument("--png-align-pca", action="store_true",
                    help="PCA-align the point cloud before plotting so the dominant axis faces front")
    ap.add_argument("--png-rotate-deg", type=float, default=0.0,
                    help="extra rotation (degrees) around the depth axis after PCA alignment")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if (args.save_png or args.save_pred_mesh or args.save_eval_clouds) and args.out is None:
        ap.error("--save-png / --save-pred-mesh / --save-eval-clouds require --out")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    # Always use the scene the checkpoint was trained on.
    if args.ckpt is not None:
        trained_scene = _resolve_trained_scene(args.ckpt)
        if trained_scene is not None:
            if args.scene is None:
                print(f"[info] scene auto-resolved from checkpoint config: {trained_scene}", flush=True)
                args.scene = trained_scene
            elif trained_scene.resolve() != args.scene.resolve():
                print(f"[info] using trained scene (overrides --scene):", flush=True)
                print(f"       trained: {trained_scene}", flush=True)
                print(f"       ignored: {args.scene}", flush=True)
                args.scene = trained_scene
        elif args.scene is None:
            ap.error("--scene is required: checkpoint has no config.json with a scene path.")

    if args.scene is None:
        ap.error("--scene is required when using --mesh.")

    scan_id = args.scan_id if args.scan_id is not None else _infer_scan_id(args.scene)
    cam_path = args.scene / "cameras.npz"
    if not cam_path.exists():
        raise FileNotFoundError(
            f"Camera file not found: {cam_path}\n"
            f"Pass --scene to the DTU scene directory containing cameras.npz "
            f"(for example .../data/dtu/scan{scan_id})."
        )
    cam_dict = np.load(cam_path)
    scale_mat = cam_dict["scale_mat_0"].astype(np.float64)

    t0 = time.time()
    if args.ckpt is not None and args.psi is not None:
        coarse_pt, psi_pt = args.ckpt, args.psi
        source_label = f"{coarse_pt} + psi={psi_pt}"
        print(f"[pred] extracting refined mesh: ckpt={coarse_pt}  psi={psi_pt} "
              f"(res={args.res}, bound={args.bound})")
        verts, faces, _ = _extract_mesh_from_refined(coarse_pt, psi_pt,
                                                      args.bound, args.res, device)
        pred_space = "normalized"
    elif args.ckpt is not None and _is_refined_checkpoint(args.ckpt):
        psi_pt = args.ckpt
        coarse_pt = _resolve_coarse_pt(psi_pt)
        print(f"[pred] auto-detected refined checkpoint — coarse={coarse_pt}  psi={psi_pt} "
              f"(res={args.res}, bound={args.bound})")
        source_label = f"{coarse_pt} + psi={psi_pt}"
        verts, faces, _ = _extract_mesh_from_refined(coarse_pt, psi_pt,
                                                      args.bound, args.res, device)
        pred_space = "normalized"
    elif args.ckpt is not None:
        source_label = str(args.ckpt)
        print(f"[pred] extracting mesh from checkpoint {args.ckpt} (res={args.res}, bound={args.bound})")
        verts, faces, _ = _extract_mesh_from_model(args.ckpt, args.bound, args.res, device)
        pred_space = "normalized"
    else:
        source_label = str(args.mesh)
        print(f"[pred] loading mesh {args.mesh} (space={args.mesh_space})")
        verts, faces, _ = _load_mesh(args.mesh)
        pred_space = args.mesh_space
    print(f"       {len(verts):,} verts, {len(faces):,} faces ({time.time() - t0:.1f}s)")

    print(f"[pred] sampling {args.n_pred_points:,} surface points", flush=True)
    t_s = time.time()
    pred_pts = _sample_surface(verts, faces, args.n_pred_points, args.seed)
    print(f"       sampled {len(pred_pts):,} pts in {time.time() - t_s:.1f}s", flush=True)
    if pred_space == "normalized":
        pred_world = _to_world(pred_pts, scale_mat)
    else:
        pred_world = pred_pts

    print(f"[gt] loading DTU GT cloud for scan{scan_id}")
    gt_pts, dtu = _load_dtu_gt(args.dtu_eval_dir, scan_id, args.gt_ply)
    pred_obs = _in_obs(pred_world, dtu["ObsMask"], dtu["BB"], dtu["Res"])
    pred_eval = pred_world[pred_obs]
    if len(pred_eval) == 0:
        raise ValueError("No predicted points remain after DTU ObsMask filtering.")
    if len(gt_pts) == 0:
        raise ValueError("No GT points remain after DTU ObsMask filtering.")
    print(f"     pred: {len(pred_eval):,} / {len(pred_world):,} pts in ObsMask")
    print(f"     gt:   {len(gt_pts):,} / {dtu['n_gt_total']:,} pts in ObsMask ({Path(dtu['ply']).name})")
    n_gt_obsmask = int(len(gt_pts))

    if args.gt_z_min is not None or args.gt_z_max is not None:
        z = gt_pts[:, 2]
        keep_z = np.ones(len(gt_pts), dtype=bool)
        if args.gt_z_min is not None:
            keep_z &= z >= args.gt_z_min
        if args.gt_z_max is not None:
            keep_z &= z <= args.gt_z_max
        gt_pts = gt_pts[keep_z]
        print(f"     gt Z-clip [{args.gt_z_min},{args.gt_z_max}] => {len(gt_pts):,} pts remaining")
        if len(gt_pts) == 0:
            raise ValueError("No GT points remain after Z-clip filter.")

    gt_mask_filter_info = None
    if args.gt_mask_filter:
        print(f"[gt] mask filter: projecting {len(gt_pts):,} pts into {args.scene}/mask/ …", flush=True)
        t_mf = time.time()
        keep_mf, gt_mask_filter_info = _mask_filter_gt(
            gt_pts,
            args.scene,
            mode=args.gt_mask_mode,
            min_ratio=args.gt_mask_min_ratio,
            min_views=args.gt_mask_min_views,
        )
        gt_pts = gt_pts[keep_mf]
        print(f"     mask filter ({args.gt_mask_mode}, min_views={args.gt_mask_min_views}, "
              f"min_ratio={args.gt_mask_min_ratio:g}) => {len(gt_pts):,} pts "
              f"(removed {(~keep_mf).sum():,}) in {time.time() - t_mf:.1f}s", flush=True)
        if len(gt_pts) == 0:
            raise ValueError("No GT points remain after mask filter.")

    gt_eval = gt_pts
    gt_crop_info = {"mode": args.gt_crop, "n_before": int(len(gt_pts)), "n_after": int(len(gt_pts))}
    if args.gt_crop == "pred-bbox":
        keep_gt, bbox = _crop_to_bbox(gt_pts, pred_eval, args.gt_crop_padding)
        gt_eval = gt_pts[keep_gt]
        gt_crop_info.update({"n_after": int(len(gt_eval)), "bbox": bbox})
        if len(gt_eval) == 0:
            raise ValueError("No GT points remain after pred-bbox crop. Increase --gt-crop-padding.")
        print(f"     gt crop: pred bbox + {args.gt_crop_padding:g} => "
              f"{len(gt_eval):,} / {len(gt_pts):,} pts")

    removed_gt = np.empty((0, 3), dtype=gt_eval.dtype)
    gt_pred_dist_filter = None
    if args.gt_max_dist_to_pred is not None:
        from scipy.spatial import cKDTree
        gt_to_pred_dist, _ = cKDTree(pred_eval).query(gt_eval, k=1, workers=-1)
        keep_gt = gt_to_pred_dist <= args.gt_max_dist_to_pred
        removed_gt = gt_eval[~keep_gt]
        gt_eval = gt_eval[keep_gt]
        gt_pred_dist_filter = {
            "max_dist": float(args.gt_max_dist_to_pred),
            "n_before": int(len(gt_to_pred_dist)),
            "n_after": int(len(gt_eval)),
            "n_removed": int((~keep_gt).sum()),
        }
        if len(gt_eval) == 0:
            raise ValueError("No GT points remain after --gt-max-dist-to-pred filter.")
        print(f"     gt pred-dist filter: <= {args.gt_max_dist_to_pred:g} => "
              f"{len(gt_eval):,} / {len(gt_to_pred_dist):,} pts")

    max_dist = args.max_dist if args.max_dist and args.max_dist > 0 else None
    print(f"[metrics] nearest-neighbour Chamfer (max_dist={max_dist}, "
          f"pred={len(pred_eval):,}, gt={len(gt_eval):,}, device={device})", flush=True)
    t_m = time.time()
    metrics, pred_to_gt, gt_to_pred = _nn_metrics(pred_eval, gt_eval, max_dist, device=device)
    print(f"          chamfer+F-score NN done in {time.time() - t_m:.1f}s", flush=True)
    for thr in args.thresholds:
        p = float((pred_to_gt < thr).mean())
        c = float((gt_to_pred < thr).mean())
        metrics[f"precision@{thr:.4g}"] = p
        metrics[f"completeness@{thr:.4g}"] = c
        metrics[f"f_score@{thr:.4g}"] = 2 * p * c / (p + c) if (p + c) > 0 else 0.0

    print("\n=== DTU GT Point Cloud Chamfer ===")
    print(f"  source:       {source_label}")
    print(f"  scene:        {args.scene}")
    print(f"  scan:         {scan_id}")
    print(f"  accuracy:     {metrics['accuracy']:.4f}  pred->gt")
    print(f"  completeness: {metrics['completeness']:.4f}  gt->pred")
    print(f"  chamfer:      {metrics['chamfer']:.4f}")
    print(f"  hausdorff:    {metrics['hausdorff']:.4f}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        if args.save_pred_mesh:
            import trimesh
            mesh_verts = _to_world(verts, scale_mat) if pred_space == "normalized" else verts
            trimesh.Trimesh(vertices=mesh_verts, faces=faces, process=False).export(args.out / "pred_world_mesh.ply")
        if args.save_eval_clouds:
            _save_point_cloud(args.out / "pred_eval_points.ply", pred_eval)
            _save_point_cloud(args.out / "gt_eval_points.ply", gt_eval)
            if len(removed_gt):
                _save_point_cloud(args.out / "gt_removed_points.ply", removed_gt)
        if args.save_png:
            _render_error_png(pred_eval, pred_to_gt, gt_eval, gt_to_pred,
                              args.out / "chamfer_error.png", vmax=args.png_vmax,
                              align_pca=args.png_align_pca,
                              rotate_deg=args.png_rotate_deg)
        out_file = args.out / "dtu_chamfer.json"
        payload = {
            "source": source_label,
            "scene": str(args.scene),
            "scan_id": scan_id,
            "dtu_eval_dir": str(args.dtu_eval_dir),
            "gt_ply": dtu["ply"],
            "res": args.res,
            "bound": args.bound,
            "n_pred_points": args.n_pred_points,
            "n_pred_eval": int(len(pred_eval)),
            "n_gt_eval": int(len(gt_eval)),
            "n_gt_obsmask": n_gt_obsmask,
            "gt_mask_filter": args.gt_mask_filter,
            "gt_mask_filter_info": gt_mask_filter_info,
            "gt_crop": gt_crop_info,
            "gt_pred_dist_filter": gt_pred_dist_filter,
            "thresholds": args.thresholds,
            "metrics": metrics,
        }
        out_file.write_text(json.dumps(payload, indent=2))
        print(f"results saved -> {out_file}")


if __name__ == "__main__":
    main()
