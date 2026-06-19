#!/usr/bin/env python3
"""3-panel ICLR comparison: no-PE vs PE L=6 per-band vs PE L=6 uniform.

Each panel shades the largest connected component of the marching-cubes mesh
from a 1M-step buddha_sweep config and captions it with the Chamfer of that
largest component (consistent with results_biggest.csv protocol).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_MESH = Path("data/gt_meshes/happy_buddha_norm.ply")
SWEEP = Path("outputs/buddha_sweep")

PRESETS = {
    # original 4-panel band-comparison figure
    "default": [
        ("no-PE  W256 D16", "none_W256_D16_1M", 1_000_000),
        ("PE L=6 per-band  W256 D16", "pe6_per_band_W256_D16_1M", 1_000_000),
        ("PE L=6 uniform  W256 D16", "pe6_uniform_W256_D16_1M", 1_000_000),
        ("PE L=6  W256 D16", "pe_W256_D16", 50_000),
    ],
    # 3-panel: best no-PE vs best PE vs the output-div regularised PE run
    "outputdiv": [
        ("no-PE  W256 D16", "none_W256_D16_1M", 1_000_000),
        ("PE L=6  W256 D16", "pe_W256_D16", 50_000),
        ("PE L=6 + output-div  W256 D16",
         "pe6_outputdiv_lambda_W256_D16_300k", 300_000),
    ],
    # 5-panel: original 4 + the output-div regularised PE run
    "all5": [
        ("no-PE  W256 D16", "none_W256_D16_1M", 1_000_000),
        ("PE L=6 per-band  W256 D16", "pe6_per_band_W256_D16_1M", 1_000_000),
        ("PE L=6 uniform  W256 D16", "pe6_uniform_W256_D16_1M", 1_000_000),
        ("PE L=6  W256 D16", "pe_W256_D16", 50_000),
        ("PE L=6 + output-div  W256 D16",
         "pe6_outputdiv_lambda_W256_D16_300k", 300_000),
    ],
}


def render_shaded(mesh: trimesh.Trimesh, res: int, az: float, el: float,
                  bound: np.ndarray) -> np.ndarray:
    a, e = np.radians(az), np.radians(el)
    fwd = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)

    lo, hi = bound
    centre = 0.5 * (lo + hi)
    radius = 0.55 * float(np.linalg.norm(hi - lo))
    g = np.linspace(-radius, radius, res)
    gx, gy = np.meshgrid(g, g[::-1])
    plane = (centre + gx[..., None] * right + gy[..., None] * upv)
    origins = (plane - 4.0 * radius * fwd).reshape(-1, 3)
    dirs = np.broadcast_to(fwd, origins.shape)

    loc, idr, idt = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=False)
    img = np.ones((res * res, 3), np.float32)
    if len(loc):
        n = mesh.face_normals[idt].copy()
        n[(n * fwd).sum(-1) > 0] *= -1
        key = -fwd
        fill = right * 0.6 + upv * 0.5; fill /= np.linalg.norm(fill)
        sh = (0.75 * (n @ key).clip(0, 1) +
              0.30 * (n @ fill).clip(0, 1) + 0.12).clip(0, 1)
        img[idr] = sh[:, None] * np.array([0.80, 0.78, 0.74], np.float32)
    return img.reshape(res, res, 3)


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: m.area)


def chamfer_largest(gt_pts: np.ndarray, gt_tree: cKDTree,
                    pred_main: trimesh.Trimesh, n: int) -> float:
    pred_pts = np.asarray(pred_main.sample(n), np.float32)
    pred_tree = cKDTree(pred_pts)
    acc_d, _ = pred_tree.query(gt_pts, workers=-1)
    comp_d, _ = gt_tree.query(pred_pts, workers=-1)
    return 0.5 * (float(comp_d.mean()) + float(acc_d.mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default")
    ap.add_argument("--res", type=int, default=1100)
    ap.add_argument("--az", type=float, default=35.0)
    ap.add_argument("--el", type=float, default=15.0)
    ap.add_argument("--n-points", type=int, default=100_000)
    ap.add_argument("--full", action="store_true",
                    help="render the WHOLE mesh (floaters included) and Chamfer "
                         "the full mesh, instead of the largest connected component")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    configs = PRESETS[args.preset]
    if args.out is None:
        tag = "" if args.preset == "default" else f"_{args.preset}"
        args.out = Path(f"figs/buddha_pe6_compare_W256_D16_1M{tag}"
                        + ("_floaters" if args.full else "") + ".png")

    gt = trimesh.load(GT_MESH, process=False)
    bound = gt.bounds.copy()
    gt_pts = np.asarray(gt.sample(args.n_points), np.float32)
    gt_tree = cKDTree(gt_pts)

    cells = []
    for label, sub, steps in configs:
        mp = SWEEP / sub / "pred_mesh.ply"
        m = trimesh.load(mp, process=False)
        main = largest_component(m)
        kept = 100.0 * len(main.faces) / max(len(m.faces), 1)
        # Chamfer is ALWAYS on the largest component (the honest, floater-free
        # number); --full only renders the whole mesh so the floaters are visible.
        cd = chamfer_largest(gt_pts, gt_tree, main, args.n_points)
        img = render_shaded(m if args.full else main, args.res, args.az, args.el, bound)
        cells.append((label, img, cd, kept, steps))
        print(f"  {label:<28}  steps={steps:>9,}  chamfer={cd:.5f}  kept={kept:.1f}%")

    best = min(c[2] for c in cells)
    n = len(cells)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2))
    for ax, (label, img, cd, kept, steps) in zip(axes, cells):
        ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        is_best = cd == best
        title = f"{label}\n{steps:,} steps   chamfer {cd:.5f}"
        ax.set_title(title, fontsize=11.5,
                     color="#1a7a1a" if is_best else "#1a1a1a",
                     fontweight="bold" if is_best else "normal")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
