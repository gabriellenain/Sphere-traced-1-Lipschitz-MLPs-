#!/usr/bin/env python3
"""DTU-style error PNGs for an existing analysis/eval_tnt_official.py output.

Reads <out>/<scene>.precision.ply (pred points after crop+ICP+voxel_down) and
<out>/<scene>.recall.ply (GT points after crop+voxel_down), recomputes per-point
NN distances, and writes chamfer_error.png + worst10_error.png + threshold.png.
Mirrors the visual style of eval_dtu_official.save_chamfer_png; units are mm.

Stats box / histograms / threshold curve use FULL-cloud NN distances, so they
agree with the official precision/recall in fscore.json. --distance-sample only
thins the scatter plots (rendering millions of points is useless), never stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_ply_xyz(path: Path) -> np.ndarray:
    """Stream a PLY file's xyz vertex coords. Works on point clouds AND meshes.

    Avoids open3d (not in training venv); uses trimesh which is available.
    For huge clouds (537 MB recall.ply), trimesh.load(process=False) skips
    expensive cleanup.
    """
    import trimesh
    obj = trimesh.load(str(path), process=False)
    # trimesh returns PointCloud (with .vertices) for points-only PLYs and
    # Trimesh (also .vertices) for meshes. Either case has .vertices.
    return np.asarray(obj.vertices, dtype=np.float32)


def _dark_ax(ax, labelsize: int = 7) -> None:
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(colors="white", labelsize=labelsize)
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _dark_cb(cb, vmax: float, label: str, fmt: str = "{:.2f}") -> None:
    cb.outline.set_edgecolor("white")
    cb.ax.tick_params(colors="white", labelsize=7)
    cb.set_label(label, color="white", fontsize=8)


def _subsample_points(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or len(points) <= n:
        return points
    idx = rng.choice(len(points), n, replace=False)
    return points[idx]


def render_pngs(pred: np.ndarray, gt: np.ndarray, out_dir: Path,
                scene: str, tau_mm: float, fscore_meta: dict,
                distance_sample: int = 300_000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    print(f"[png] computing full-cloud NN distances  "
          f"pred={len(pred):,}  gt={len(gt):,}", flush=True)
    # Stats box, histograms and the threshold curve all use FULL-cloud NN
    # distances so they agree with the official precision/recall in fscore.json
    # (querying a subsampled GT thins it and inflates every NN distance above τ).
    # Subsampling is applied ONLY to the scatter plots below, which can't render
    # millions of points usefully.
    acc = cKDTree(gt).query(pred, k=1, workers=-1)[0].astype(np.float32) * 1000.0
    comp = cKDTree(pred).query(gt, k=1, workers=-1)[0].astype(np.float32) * 1000.0

    # Plot-only subsamples; keep each point's distance aligned with its position.
    def _sub(points: np.ndarray, dists: np.ndarray):
        if distance_sample <= 0 or len(points) <= distance_sample:
            return points, dists
        idx = rng.choice(len(points), distance_sample, replace=False)
        return points[idx], dists[idx]
    pred_s, acc_s = _sub(pred, acc)
    gt_s, comp_s = _sub(gt, comp)

    BG, CMAP = "#0d0d0d", "plasma"
    # colour scale: 3× tau (matches the TnT toolbox's color_distances cap).
    vm = 3.0 * tau_mm
    norm = Normalize(vmin=0, vmax=vm)
    # Three axis-pair projections.
    PROJ = [(0, 2, "x", "z"), (1, 2, "y", "z"), (0, 1, "x", "y")]

    # ----------------- chamfer_error.png ----------------------------------
    fig = plt.figure(figsize=(22, 15), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.32,
                            width_ratios=[1, 1, 1, 0.85])
    ROW_LABELS = ["ACCURACY  pred→GT", "COMPLETENESS  GT→pred",
                  "OVERLAY  pred=blue  GT=orange"]

    for row, (pts, dists) in enumerate([(pred_s, acc_s), (gt_s, comp_s)]):
        for col, (i, j, xl, yl) in enumerate(PROJ):
            ax = fig.add_subplot(gs[row, col], facecolor=BG)
            order = np.argsort(dists)
            sc = ax.scatter(pts[order, i], pts[order, j],
                            c=dists[order], cmap=CMAP, norm=norm,
                            s=0.4, linewidths=0, alpha=0.85, rasterized=True)
            ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
            lbl = f"{ROW_LABELS[row]}\n{yl}" if col == 0 else yl
            ax.set_ylabel(lbl, fontsize=8, labelpad=4)
            _dark_ax(ax)
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            _dark_cb(cb, vm, "distance (mm)")

    pred_ov = _subsample_points(pred_s, min(150_000, len(pred_s)), rng)
    gt_ov = _subsample_points(gt_s, min(150_000, len(gt_s)), rng)
    for col, (i, j, xl, yl) in enumerate(PROJ):
        ax = fig.add_subplot(gs[2, col], facecolor=BG)
        ax.scatter(gt_ov[:,   i], gt_ov[:,   j], s=0.3, color="#ff7f0e",
                   alpha=0.5, linewidths=0, rasterized=True, label="GT")
        ax.scatter(pred_ov[:, i], pred_ov[:, j], s=0.3, color="#1f77b4",
                   alpha=0.6, linewidths=0, rasterized=True, label="pred")
        ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
        lbl = f"{ROW_LABELS[2]}\n{yl}" if col == 0 else yl
        ax.set_ylabel(lbl, fontsize=8, labelpad=4)
        _dark_ax(ax)

    BINS = 120
    for ax_col, (d, color, label) in enumerate([
        (acc,  "#00c8ff", "Accuracy"),
        (comp, "#ff9900", "Completeness"),
    ]):
        ax = fig.add_subplot(gs[ax_col, 3], facecolor=BG)
        clip = min(float(d.max()), vm * 3)
        bins = np.linspace(0, clip, BINS)
        ax.hist(d[d <= clip], bins=bins, color=color, alpha=0.75, density=True)
        for v, ls, lbl in [
            (float(d.mean()),               "--", f"mean {d.mean():.1f}"),
            (float(np.median(d)),           ":",  f"p50  {np.median(d):.1f}"),
            (float(np.percentile(d, 90)),   "-.", f"p90  {np.percentile(d,90):.1f}"),
        ]:
            ax.axvline(v, color="white", lw=0.9, ls=ls, alpha=0.8, label=lbl)
        ax.axvline(vm, color="#ff4444", lw=1.0, alpha=0.9, label=f"vmax {vm:.1f}")
        ax.axvline(tau_mm, color="#33ff33", lw=1.0, alpha=0.9, label=f"τ {tau_mm:.1f}")
        ax.set_xlabel("distance (mm)", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"{label} distribution", fontsize=9, pad=4)
        _dark_ax(ax); ax.legend(fontsize=7, framealpha=0.3)

    ax_txt = fig.add_subplot(gs[2, 3], facecolor=BG); ax_txt.axis("off")
    p = fscore_meta.get("precision", float("nan"))
    r = fscore_meta.get("recall",    float("nan"))
    fs = fscore_meta.get("fscore",    float("nan"))
    ax_txt.text(0.05, 0.95,
        f"precision  {p:.4f}\nrecall     {r:.4f}\nF-score    {fs:.4f}\n\n"
        f"acc  mean {float(acc.mean()):.2f} mm\nacc  p50  {float(np.median(acc)):.2f} mm\nacc  p90  {float(np.percentile(acc,90)):.2f} mm\n\n"
        f"comp mean {float(comp.mean()):.2f} mm\ncomp p50  {float(np.median(comp)):.2f} mm\ncomp p90  {float(np.percentile(comp,90)):.2f} mm\n\n"
        f"pred pts  {len(pred):,}\ngt   pts  {len(gt):,}\nτ        {tau_mm:.1f} mm",
        transform=ax_txt.transAxes, fontsize=9, va="top", ha="left",
        family="monospace", color="white", linespacing=1.7)

    fig.suptitle(
        f"TnT {scene}  ·  precision={p:.3f}  recall={r:.3f}  F={fs:.3f}"
        f"  (colour 0–{vm:.1f} mm,  τ={tau_mm:.1f} mm)",
        fontsize=13, y=0.995, color="white")
    out = out_dir / "chamfer_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] error map   -> {out}", flush=True)

    # ----------------- worst10_error.png ----------------------------------
    fig2 = plt.figure(figsize=(22, 7), facecolor=BG)
    fig2.patch.set_facecolor(BG)
    gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.32)
    worst_pred = np.argsort(acc_s)[-int(0.1 * len(acc_s)):]
    worst_gt   = np.argsort(comp_s)[-int(0.1 * len(comp_s)):]
    for col, (i, j, xl, yl) in enumerate(PROJ):
        ax = fig2.add_subplot(gs2[0, col], facecolor=BG)
        ax.scatter(pred_s[worst_pred, i], pred_s[worst_pred, j], s=0.4,
                   c=acc_s[worst_pred], cmap=CMAP, norm=norm, linewidths=0,
                   alpha=0.85, rasterized=True, label="pred top-10% acc")
        ax.scatter(gt_s[worst_gt, i], gt_s[worst_gt, j], s=0.4,
                   c=comp_s[worst_gt], cmap=CMAP, norm=norm, linewidths=0,
                   alpha=0.85, rasterized=True, label="GT top-10% comp")
        ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8, labelpad=4)
        _dark_ax(ax)
    fig2.suptitle(f"TnT {scene} · worst 10% points (acc + comp)",
                  fontsize=13, y=1.02, color="white")
    out2 = out_dir / "worst10_error.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"[png] worst10 map -> {out2}", flush=True)

    # ----------------- threshold_curve.png --------------------------------
    fig3 = plt.figure(figsize=(8, 5), facecolor=BG)
    fig3.patch.set_facecolor(BG)
    ax = fig3.add_subplot(111, facecolor=BG)
    ths_mm = np.linspace(0.0, max(vm, tau_mm * 3), 200)
    p_curve = [(acc <= th).mean() for th in ths_mm]
    r_curve = [(comp <= th).mean() for th in ths_mm]
    f_curve = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(p_curve, r_curve)]
    ax.plot(ths_mm, p_curve, color="#00c8ff", label="precision")
    ax.plot(ths_mm, r_curve, color="#ff9900", label="recall")
    ax.plot(ths_mm, f_curve, color="#aaffaa", label="F-score")
    ax.axvline(tau_mm, color="#33ff33", ls="--", lw=1.0, alpha=0.9, label=f"τ={tau_mm:.1f} mm")
    ax.set_xlabel("threshold (mm)"); ax.set_ylabel("rate")
    ax.set_title(f"TnT {scene}  precision / recall / F vs threshold")
    ax.set_ylim(0, 1)
    _dark_ax(ax); ax.legend(fontsize=8, framealpha=0.3)
    out3 = out_dir / "threshold_curve.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig3)
    print(f"[png] threshold   -> {out3}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path,
                    help="analysis/eval_tnt_official.py output dir (contains <scene>.precision.ply etc.)")
    ap.add_argument("--scene", default="Barn",
                    help="scene name (e.g. Barn) — must match the .ply file prefix")
    ap.add_argument("--tau-mm", type=float, default=None,
                    help="tau in mm (defaults: Barn=10)")
    ap.add_argument("--distance-sample", type=int, default=300_000,
                    help="deterministic point subsample per side for the SCATTER "
                         "plots only; stats/histograms/threshold use full clouds")
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser()
    pred_ply = out_dir / f"{args.scene}.precision.ply"
    gt_ply   = out_dir / f"{args.scene}.recall.ply"
    if not pred_ply.exists() or not gt_ply.exists():
        raise SystemExit(f"missing {pred_ply} or {gt_ply}")

    # default tau (in mm) — official TnT thresholds per scene
    tau_default = {"Barn": 10.0, "Truck": 5.0, "Courthouse": 25.0,
                   "Caterpillar": 5.0, "Ignatius": 3.0, "Meetingroom": 10.0,
                   "Church": 25.0}
    tau_mm = args.tau_mm if args.tau_mm is not None else tau_default.get(args.scene, 10.0)

    fscore_meta = {}
    fjson = out_dir / "fscore.json"
    if fjson.exists():
        fscore_meta = json.loads(fjson.read_text())

    print(f"[load] {pred_ply}")
    pred = _read_ply_xyz(pred_ply)
    print(f"[load] {gt_ply}")
    gt   = _read_ply_xyz(gt_ply)

    render_pngs(pred, gt, out_dir, args.scene, tau_mm, fscore_meta,
                distance_sample=args.distance_sample)


if __name__ == "__main__":
    main()
