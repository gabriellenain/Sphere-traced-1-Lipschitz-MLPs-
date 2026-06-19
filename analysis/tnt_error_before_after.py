#!/usr/bin/env python3
"""One clean before/after TnT error-map figure from two official eval outputs.

Each eval dir (analysis/eval_tnt_official.py output) holds the cropped+aligned
clouds the TnT toolbox wrote:
    <scene>.precision.ply  pred points  (for ACCURACY  pred->GT)
    <scene>.recall.ply     GT points    (for COMPLETENESS GT->pred)
plus fscore.json (precision / recall / fscore + the recon->GT transform).

We recompute per-point distances with a KD-tree (so colour is on ONE controlled
scale across both runs) and render a 2x2:

        ACCURACY      | BEFORE | AFTER |
        COMPLETENESS  | BEFORE | AFTER |

styled to match the toolbox's chamfer_error.png: plasma colormap on a dark
ground, colour 0..3*tau mm, tau marked. By default the projection is the world
x-z plane (as chamfer_error.png); with --orient-view <N> --scene <dir> the cloud
is instead projected onto camera N's image basis (rotated into the GT frame by
each run's own recon->GT transform), so the error map matches a hero render of
that view. Less colour after BA = the gain made visible.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

BG, CMAP = "#0d0d0d", "plasma"          # matches tnt_error_pngs.render_pngs
DIFF_CMAP = "RdBu"                       # +Δ (error dropped → improvement) = blue


def _xyz(p: Path) -> np.ndarray:
    return np.asarray(o3d.io.read_point_cloud(str(p)).points)


def _transform_R(d: Path) -> np.ndarray:
    """De-scaled rotation of this run's recon->GT transform (for view orientation)."""
    T = np.asarray(json.loads((d / "fscore.json").read_text())["transform"])
    R = T[:3, :3]
    return R / np.cbrt(abs(np.linalg.det(R)))


def _scores(d: Path):
    j = json.loads((d / "fscore.json").read_text())
    return j["precision"], j["recall"], j["fscore"]


def _cam_basis(args, ap):
    """Image (right, up) basis in the model frame, or None for world x-z."""
    if args.orient_view is None:
        return None
    if args.scene_dir is None:
        ap.error("--orient-view requires --scene-dir")
    from lip_tracer import data as D
    c2w = D.load_views(args.scene_dir)["c2w"][args.orient_view].numpy()
    return c2w[:3, 0], -c2w[:3, 1]          # image x, image up(-y)


def _basis_for(d: Path, cam):
    """This run's GT-frame projection basis (world x-z, or rotated cam basis)."""
    if cam is None:
        return np.array([1., 0, 0]), np.array([0, 0, 1.])
    R = _transform_R(d)
    return R @ cam[0], R @ cam[1]


def _rasterize(x, y, err, extent, nx, ny):
    """Mean per-cell error on a shared grid; empty cells = NaN."""
    xmin, xmax, ymin, ymax = extent
    ix = np.clip(((x - xmin) / (xmax - xmin) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / (ymax - ymin) * ny).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    summ = np.bincount(flat, weights=err, minlength=nx * ny)
    cnt = np.bincount(flat, minlength=nx * ny)
    out = np.full(nx * ny, np.nan)
    nz = cnt > 0
    out[nz] = summ[nz] / cnt[nz]
    return out.reshape(ny, nx)


def run_diff(args) -> None:
    """3-column error-DIFFERENCE figure from exactly 3 eval dirs (before R,t R,t+int)."""
    ap = argparse.ArgumentParser()          # only for ap.error()
    if len(args.dirs) != 3:
        ap.error("--diff needs exactly 3 dirs: before R,t R,t+int")
    cam = _cam_basis(args, ap)
    vd = args.diff_vmax if args.diff_vmax is not None else 2.0 * args.tau * 1000.0
    norm = Normalize(vmin=-vd, vmax=vd)
    cmap = plt.get_cmap(DIFF_CMAP).copy(); cmap.set_bad(BG)

    def load(d: Path):
        pred = _xyz(d / f"{args.scene}.precision.ply")
        gt = _xyz(d / f"{args.scene}.recall.ply")
        acc = cKDTree(gt).query(pred, k=1, workers=-1)[0] * 1000.0
        comp = cKDTree(pred).query(gt, k=1, workers=-1)[0] * 1000.0
        basis = _basis_for(d, cam)
        return dict(pred=pred, gt=gt, acc=acc, comp=comp, basis=basis, s=_scores(d))

    cols = [load(d) for d in args.dirs]
    names = ["before", "R,t", "R,t+intr"]
    for nm, col in zip(names, cols):
        print(f"[diff] {nm}: P/R/F = {col['s'][0]:.3f}/{col['s'][1]:.3f}/{col['s'][2]:.3f}")

    rows = [("ACCURACY  (pred → GT)", "pred", "acc"),
            ("COMPLETENESS  (GT → pred)", "gt", "comp")]
    pairs = [(0, 1), (0, 2), (1, 2)]        # before-R,t  before-R,t+int  R,t-R,t+int
    titles = [f"{names[a]} − {names[b]}" for a, b in pairs]

    fig, axes = plt.subplots(2, 3, figsize=(5.2 * 3, 12), facecolor=BG, squeeze=False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    nx = args.diff_grid
    for r, (rlabel, pk, ek) in enumerate(rows):
        # shared projected extent + grid for this row (1,99 pctile, padded)
        xs = [c[pk] @ c["basis"][0] for c in cols]
        ys = [c[pk] @ c["basis"][1] for c in cols]
        ax_ = np.concatenate(xs); ay_ = np.concatenate(ys)
        xmin, xmax = np.percentile(ax_, [0.5, 99.5])
        ymin, ymax = np.percentile(ay_, [0.5, 99.5])
        px, py = 0.03 * (xmax - xmin), 0.03 * (ymax - ymin)
        extent = (xmin - px, xmax + px, ymin - py, ymax + py)
        ny = int(nx * (extent[3] - extent[2]) / (extent[1] - extent[0]))
        grids = [_rasterize(x, y, c[ek], extent, nx, ny)
                 for x, y, c in zip(xs, ys, cols)]
        for cidx, (a, b) in enumerate(pairs):
            ax = axes[r, cidx]
            d = grids[a] - grids[b]          # +Δ = a worse than b
            ax.imshow(d, origin="lower", extent=extent, cmap=cmap, norm=norm,
                      aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[cidx], color="white", fontsize=12, pad=8)
        cb = fig.colorbar(sm, ax=axes[r], fraction=0.035, pad=0.02)
        cb.set_label(f"{rlabel}   Δdistance (mm)   +blue = error ↓ (better)",
                     color="white", fontsize=9)
        cb.ax.tick_params(colors="white", labelsize=7)
        cb.outline.set_edgecolor("white")
    plt.subplots_adjust(left=0.01, right=0.93, top=0.96, bottom=0.01,
                        wspace=0.02, hspace=0.04)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, facecolor=BG)
    print(f"[diff] wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True,
                    help="eval dirs as columns, in display order (e.g. baseline R,t full)")
    ap.add_argument("--scene", default="Ignatius", help="scene name (PLY/json prefix)")
    ap.add_argument("--tau", type=float, default=0.003, help="TnT threshold (scene units)")
    ap.add_argument("--max-points", type=int, default=300000, help="scatter subsample")
    ap.add_argument("--orient-view", type=int, default=None,
                    help="project onto this camera's image basis instead of world x-z")
    ap.add_argument("--scene-dir", type=Path, default=None,
                    help="scene dir with poses (needed for --orient-view), e.g. data/tnt/Ignatius")
    ap.add_argument("--diff", action="store_true",
                    help="render error-DIFFERENCE maps instead of per-run error maps. "
                         "Needs exactly 3 dirs (before R,t R,t+int); each row gives the "
                         "3 pairwise differences (before-R,t), (before-R,t+int), "
                         "(R,t-R,t+int) as binned Δerror images. +Δ (blue) = error dropped.")
    ap.add_argument("--diff-vmax", type=float, default=None,
                    help="symmetric colour cap for --diff in mm (default 2*tau)")
    ap.add_argument("--diff-grid", type=int, default=640,
                    help="raster width (px) for --diff binning")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.diff:
        return run_diff(args)

    vm = 3.0 * args.tau * 1000.0                 # colour cap in mm (== toolbox)
    norm = Normalize(vmin=0, vmax=vm)
    rng = np.random.default_rng(0)

    # camera image basis in the model frame (rotated to GT per-run below)
    cam_right = cam_up = None
    if args.orient_view is not None:
        if args.scene_dir is None:
            ap.error("--orient-view requires --scene-dir")
        from lip_tracer import data as D
        c2w = D.load_views(args.scene_dir)["c2w"][args.orient_view].numpy()
        cam_right, cam_up = c2w[:3, 0], -c2w[:3, 1]   # image x, image up(-y)

    def load(d: Path):
        pred = _xyz(d / f"{args.scene}.precision.ply")   # pred points
        gt   = _xyz(d / f"{args.scene}.recall.ply")      # GT points
        acc  = cKDTree(gt).query(pred, k=1, workers=-1)[0] * 1000.0
        comp = cKDTree(pred).query(gt, k=1, workers=-1)[0] * 1000.0
        if cam_right is None:
            basis = (np.array([1., 0, 0]), np.array([0, 0, 1.]))   # world x, z
        else:
            R = _transform_R(d)
            basis = (R @ cam_right, R @ cam_up)
        return dict(pred=pred, gt=gt, acc=acc, comp=comp, basis=basis, s=_scores(d))

    cols = [load(d) for d in args.dirs]
    for d, col in zip(args.dirs, cols):
        print(f"[fig] {d.name}: P/R/F = {col['s'][0]:.3f}/{col['s'][1]:.3f}/{col['s'][2]:.3f}")

    def draw(ax, pts, err, basis):
        u, v = basis
        x, y = pts @ u, pts @ v
        idx = rng.choice(len(pts), min(len(pts), args.max_points), replace=False)
        o = np.argsort(err[idx])                      # worst points drawn last (on top)
        ax.scatter(x[idx][o], y[idx][o], c=err[idx][o], cmap=CMAP, norm=norm,
                   s=0.4, linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_visible(False)

    n = len(cols)
    rows = [("ACCURACY  (pred → GT)", "pred", "acc"),
            ("COMPLETENESS  (GT → pred)", "gt", "comp")]
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 12), facecolor=BG, squeeze=False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    for r, (rlabel, pk, ek) in enumerate(rows):
        for c, col in enumerate(cols):
            draw(axes[r, c], col[pk], col[ek], col["basis"])
        cb = fig.colorbar(sm, ax=axes[r], fraction=0.035, pad=0.02)
        cb.set_label(f"{rlabel}   distance (mm)", color="white", fontsize=9)
        cb.ax.tick_params(colors="white", labelsize=7)
        cb.outline.set_edgecolor("white")
        cb.ax.axhline(args.tau * 1000.0, color="white", lw=1.2)   # tau line
    plt.subplots_adjust(left=0.01, right=0.93, top=0.99, bottom=0.01, wspace=0.02, hspace=0.04)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, facecolor=BG)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
