#!/usr/bin/env python3
"""Render a curvature heatmap of the GT DTU mesh projected onto a given view.

Curvature proxy: 1 − |mean unit-normal of 1-ring neighbours|.
High value = neighbour normals diverge = fine detail / sharp feature.

Camera convention: --view is 0-based (matches world_mat_{i} in cameras.npz
and image/00000{i}.png).  DTU scan65/scan122 have 49 views → valid range 0-48.

Example:
    python curvature_heatmap.py \\
        --dtu-eval-dir /scratch/.../MVS\ Data \\
        --scan-id 65 \\
        --scene /scratch/.../dtu/scan65 \\
        --view 48
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ── curvature ─────────────────────────────────────────────────────────────────

def _curvature_pca(pts: np.ndarray, k: int = 20, chunk: int = 50_000,
                   subsample: int | None = None, seed: int = 0,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """PCA curvature + normals on a point cloud.

    Returns (pts_out [M,3], curvature [M], normals [M,3]).
    curvature = λ_min / (λ_min + λ_mid + λ_max)  (surface variation).
    normal    = eigenvector of λ_min.
    Subsamples to `subsample` random points before computing if given.
    """
    from scipy.spatial import cKDTree

    if subsample is not None and len(pts) > subsample:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), subsample, replace=False)]

    tree  = cKDTree(pts)
    curv  = np.zeros(len(pts), dtype=np.float32)
    norms = np.zeros((len(pts), 3), dtype=np.float32)

    for start in range(0, len(pts), chunk):
        end = min(start + chunk, len(pts))
        _, nn_idx   = tree.query(pts[start:end], k=k, workers=-1)  # (C, k)
        neighbors   = pts[nn_idx]                                   # (C, k, 3)
        centered    = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov         = np.einsum("...ni,...nj->...ij", centered, centered) / k
        ev, evec    = np.linalg.eigh(cov)                          # ascending
        total       = ev.sum(axis=1) + 1e-10
        curv[start:end]  = (ev[:, 0] / total).astype(np.float32)
        norms[start:end] = evec[:, :, 0].astype(np.float32)       # λ_min eigenvec

    return pts, curv, norms


# ── projection ────────────────────────────────────────────────────────────────

def _project(pts: np.ndarray, P34: np.ndarray):
    """Project world-space pts (N,3) through 3×4 P=K[R|t].
    Returns (u, v, depth) where depth = P[2] · x_hom.
    """
    pts_h = np.concatenate([pts.astype(np.float64),
                             np.ones((len(pts), 1), np.float64)], axis=1).T  # 4×N
    proj  = P34.astype(np.float64) @ pts_h   # 3×N
    depth = proj[2]
    valid = depth > 0
    u = np.where(valid, proj[0] / np.where(valid, depth, 1.0), np.nan)
    v = np.where(valid, proj[1] / np.where(valid, depth, 1.0), np.nan)
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32)


# ── optional ObsMask filter ───────────────────────────────────────────────────

def _apply_obsmask(verts: np.ndarray, dtu_eval_dir: Path, scan_id: int) -> np.ndarray:
    """Return boolean mask: True where GT vertex passes DTU ObsMask+Plane filter."""
    from scipy.io import loadmat

    mat = loadmat(str(dtu_eval_dir / "ObsMask" / f"ObsMask{scan_id}_10.mat"))
    obs, BB, Res = (mat["ObsMask"].astype(bool),
                    mat["BB"].astype(np.float64),
                    float(mat["Res"].flat[0]))
    in_bb  = np.all((verts >= BB[0]) & (verts <= BB[1]), axis=1)
    idx    = np.clip(np.round((verts - BB[0]) / Res).astype(int),
                     0, np.array(obs.shape) - 1)
    in_obs = in_bb & obs[idx[:, 0], idx[:, 1], idx[:, 2]]

    pts_obs = verts[in_obs]
    P_plane = loadmat(str(dtu_eval_dir / "ObsMask" / f"Plane{scan_id}.mat"))["P"]
    hom     = np.concatenate([pts_obs, np.ones((len(pts_obs), 1), np.float32)], axis=1)
    above   = (P_plane.reshape(1, 4) * hom).sum(-1) > 0

    keep = np.zeros(len(verts), dtype=bool)
    keep[np.where(in_obs)[0][above]] = True
    return keep


# ── render ────────────────────────────────────────────────────────────────────

def _render(u: np.ndarray, v: np.ndarray, depth: np.ndarray, curv: np.ndarray,
            W: int, H: int, img_bg: np.ndarray | None,
            scan_id: int, view_idx: int, n_total: int,
            vmax_pct: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    BG, CMAP = "#0d0d0d", "plasma"
    vm   = float(np.percentile(curv, vmax_pct)) if len(curv) else 0.05
    norm = Normalize(vmin=0, vmax=vm)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})
    fig.patch.set_facecolor(BG)

    titles = ["all GT vertices", "ObsMask + Plane filtered"] if n_total != len(curv) else ["all GT vertices", ""]
    datasets = [(u, v, depth, curv)]

    for ax_i, ax in enumerate(axes):
        ax.set_facecolor(BG)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("u  (px)", fontsize=9, color="white")
        if ax_i == 0:
            ax.set_ylabel("v  (px)", fontsize=9, color="white")

        if ax_i == 1 and img_bg is not None:
            ax.imshow(img_bg, extent=[0, W, H, 0], aspect="auto", alpha=1.0)
            ax.set_title(f"image + curvature overlay", fontsize=10, color="white", pad=6)
        else:
            ax.set_title(titles[ax_i] if ax_i < len(titles) else "", fontsize=10, color="white", pad=6)

        order = np.argsort(-depth)   # far-to-near: near on top
        sc = ax.scatter(u[order], v[order],
                        c=curv[order], cmap=CMAP, norm=norm,
                        s=0.25, linewidths=0, alpha=0.95, rasterized=True)

        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("curvature proxy", fontsize=8, color="white")
        cb.ax.tick_params(colors="white", labelsize=7)
        cb.ax.yaxis.label.set_color("white")

    n_vis = len(curv)
    fig.suptitle(
        f"GT curvature heatmap  ·  DTU scan{scan_id}  ·  view {view_idx} (0-based)\n"
        f"{n_vis:,} visible vertices  ·  vmax = {vm:.4f} @ p{vmax_pct:.0f}  "
        f"·  curvature = 1 − |mean 1-ring normal|",
        color="white", fontsize=11, y=1.01)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[done] {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtu-eval-dir", type=Path, required=True,
                    help="Path to DTU SampleSet/MVS Data (contains ObsMask/, Points/stl/)")
    ap.add_argument("--scan-id",      type=int,  required=True)
    ap.add_argument("--scene",        type=Path, required=True,
                    help="DTU scene dir with cameras.npz and image/")
    ap.add_argument("--view",         type=int,  default=48,
                    help="Camera index (0-based, default 48 = last of the 49-view split)")
    ap.add_argument("--out",          type=Path, default=None)
    ap.add_argument("--vmax-pct",     type=float, default=95.0,
                    help="Percentile for colormap vmax (default 95)")
    ap.add_argument("--subsample",    type=int,   default=400_000,
                    help="Random subsample of GT pts for curvature computation "
                         "(default 400000; use 0 for all — slow on 3M+ pts)")
    ap.add_argument("--knn",          type=int,   default=20,
                    help="Neighbours for PCA curvature (default 20)")
    ap.add_argument("--no-obsmask",   action="store_true",
                    help="Skip ObsMask+Plane filter and show all GT vertices")
    args = ap.parse_args()

    out = args.out or Path(f"curv_heatmap_scan{args.scan_id}_view{args.view:02d}.png")

    # ── load GT point cloud ───────────────────────────────────────────────────
    import trimesh
    ply_path = args.dtu_eval_dir / "Points" / "stl" / f"stl{args.scan_id:03d}_total.ply"
    print(f"[gt] loading {ply_path} …", flush=True)
    obj   = trimesh.load(str(ply_path), process=False)
    verts = np.asarray(obj.vertices, dtype=np.float32)
    print(f"[gt] {len(verts):,} points", flush=True)

    # ── optional ObsMask+Plane filter ─────────────────────────────────────────
    if args.no_obsmask:
        keep = np.ones(len(verts), dtype=bool)
    else:
        print("[gt] applying ObsMask + Plane filter…", flush=True)
        keep = _apply_obsmask(verts, args.dtu_eval_dir, args.scan_id)
        print(f"[gt] {keep.sum():,} / {len(verts):,} points pass filter", flush=True)

    verts_f = verts[keep]

    # ── curvature (PCA on k-NN, optional subsample) ───────────────────────────
    sub = args.subsample if args.subsample > 0 else None
    print(f"[curv] PCA curvature  k={args.knn}"
          + (f"  subsample={sub:,}" if sub else "  (all points)") + "…", flush=True)
    verts_f, curv_f, _ = _curvature_pca(verts_f, k=args.knn, subsample=sub)
    print(f"[curv] range [{curv_f.min():.4f}, {curv_f.max():.4f}]  "
          f"p95={float(np.percentile(curv_f, 95)):.4f}", flush=True)

    # ── load camera ───────────────────────────────────────────────────────────
    cam    = np.load(args.scene / "cameras.npz")
    P_key  = f"world_mat_{args.view}"
    if P_key not in cam:
        wm_keys = sorted(k for k in cam if k.startswith("world_mat_") and "inv" not in k)
        indices = sorted(int(k.split("_")[-1]) for k in wm_keys)
        ap.error(f"{P_key} not in cameras.npz.  Valid range: {indices[0]}–{indices[-1]}")
    P34 = cam[P_key][:3, :4].astype(np.float64)

    # ── load reference image ──────────────────────────────────────────────────
    img_bg = None
    img_dir = args.scene / "image"
    H_img = W_img = None
    if img_dir.exists():
        img_paths = sorted(p for p in img_dir.glob("*.png") if not p.name.startswith("._"))
        if args.view < len(img_paths):
            from PIL import Image
            img_bg = np.array(Image.open(img_paths[args.view]))
            H_img, W_img = img_bg.shape[:2]
            print(f"[img] {img_paths[args.view].name}  {W_img}×{H_img}", flush=True)

    # ── project ───────────────────────────────────────────────────────────────
    print(f"[proj] projecting {len(verts_f):,} vertices through {P_key}…", flush=True)
    u, v, depth = _project(verts_f, P34)

    if H_img is None or W_img is None:
        valid_any = np.isfinite(u) & np.isfinite(v) & (depth > 0)
        W_img = int(np.nanmax(u[valid_any])) + 10 if valid_any.any() else 1600
        H_img = int(np.nanmax(v[valid_any])) + 10 if valid_any.any() else 1200

    in_frame = (depth > 0) & np.isfinite(u) & np.isfinite(v) & \
               (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)

    u_vis     = u[in_frame]
    v_vis     = v[in_frame]
    depth_vis = depth[in_frame]
    curv_vis  = curv_f[in_frame]
    print(f"[proj] {in_frame.sum():,} vertices visible in frame", flush=True)

    if in_frame.sum() == 0:
        print("[warn] no vertices project into frame — check --view index", flush=True)
        return

    # ── render ────────────────────────────────────────────────────────────────
    _render(u_vis, v_vis, depth_vis, curv_vis,
            W_img, H_img, img_bg,
            args.scan_id, args.view, len(verts_f),
            args.vmax_pct, out)


if __name__ == "__main__":
    main()
