#!/usr/bin/env python3
"""Grid of shaded renders for the Buddha W=256,D=16 PE-bands sweep (L=0..10).

For each config:
  - load pred_mesh.ply, keep largest connected component (by area),
  - recompute bidirectional Chamfer vs the GT mesh (100k samples, seed 0),
  - render a shaded orthographic view with the same camera as analysis/buddha_sweep_grid.py.

Emits one grid figure (GT + L=0..10 = 12 cells) and a Chamfer-vs-L line chart.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("outputs/buddha_sweep")
GT_MESH = Path("data/gt_meshes/happy_buddha_norm.ply")


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: m.area)


def chamfer(gt_pts, gt_tree, pred_mesh, n_points, seed):
    rng = np.random.default_rng(seed)
    pred_pts = np.asarray(pred_mesh.sample(n_points), np.float32)
    pred_tree = cKDTree(pred_pts)
    acc_d, _ = pred_tree.query(gt_pts, workers=-1)
    comp_d, _ = gt_tree.query(pred_pts, workers=-1)
    accuracy = float(comp_d.mean())
    completeness = float(acc_d.mean())
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer": 0.5 * (accuracy + completeness),
        "hausdorff": float(max(comp_d.max(), acc_d.max())),
    }


def render_shaded(mesh, res, az, el, bound):
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

    loc, idr, idt = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    img = np.ones((res * res, 3), np.float32)
    if len(loc):
        n = mesh.face_normals[idt].copy()
        n[(n * fwd).sum(-1) > 0] *= -1
        key = -fwd
        fill = right * 0.6 + upv * 0.5; fill /= np.linalg.norm(fill)
        sh = (0.75 * (n @ key).clip(0, 1)
              + 0.30 * (n @ fill).clip(0, 1) + 0.12).clip(0, 1)
        img[idr] = sh[:, None] * np.array([0.80, 0.78, 0.74], np.float32)
    return img.reshape(res, res, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=420)
    ap.add_argument("--az", type=float, default=35.0)
    ap.add_argument("--el", type=float, default=15.0)
    ap.add_argument("--n-points", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-out", type=Path,
                    default=Path("figs/buddha_pe_L_sweep_grid.png"))
    ap.add_argument("--curve-out", type=Path,
                    default=Path("figs/buddha_pe_L_chamfer.png"))
    ap.add_argument("--csv-out", type=Path,
                    default=SWEEP / "results_L_sweep_biggest.csv")
    args = ap.parse_args()

    configs = [(0, SWEEP / "none_W256_D16")]
    for L in range(1, 11):
        configs.append((L, SWEEP / f"pe{L}_none_W256_D16_50k"))

    gt = trimesh.load(GT_MESH, process=False)
    bound = gt.bounds.copy()
    gt_pts = np.asarray(gt.sample(args.n_points), np.float32)
    gt_tree = cKDTree(gt_pts)

    cells = [("GT", render_shaded(gt, args.res, args.az, args.el, bound), None, None)]
    rows = []
    for L, d in configs:
        mp = d / "pred_mesh.ply"
        if not mp.exists():
            cells.append((f"L={L}", np.ones((args.res, args.res, 3), np.float32),
                          None, None))
            print(f"missing: {mp}")
            continue
        raw = trimesh.load(mp, process=False)
        main_mesh = largest_component(raw)
        main_mesh.export(d / "pred_mesh_main.ply")
        cd = chamfer(gt_pts, gt_tree, main_mesh, args.n_points, args.seed)
        kept = 100.0 * len(main_mesh.faces) / max(len(raw.faces), 1)
        print(f"L={L:>2}  chamfer={cd['chamfer']:.5f}  "
              f"haus={cd['hausdorff']:.4f}  kept {kept:5.1f}%  faces={len(main_mesh.faces)}")
        img = render_shaded(main_mesh, args.res, args.az, args.el, bound)
        cells.append((f"L={L}", img, cd["chamfer"], cd["hausdorff"]))
        rows.append({
            "L": L,
            "chamfer_biggest": cd["chamfer"],
            "accuracy_biggest": cd["accuracy"],
            "completeness_biggest": cd["completeness"],
            "hausdorff_biggest": cd["hausdorff"],
            "n_faces_biggest": len(main_mesh.faces),
            "kept_pct": kept,
            "out_dir": str(d),
        })

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {args.csv_out}")

    # --- grid figure --------------------------------------------------------
    n = len(cells)              # 12
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    chvals = [c[2] for c in cells if c[2] is not None]
    best = min(chvals) if chvals else None
    for ax, cell in zip(axes.flat, cells + [(None, None, None, None)] * (nrows*ncols - n)):
        name, img, ch, hd = cell
        if img is None:
            ax.axis("off"); continue
        ax.imshow(img)
        if ch is None:
            title, color, bold = name, "#1a1a1a", False
        else:
            title = f"{name}\nchamfer {ch:.5f}"
            is_best = (ch == best)
            color = "#1a7a1a" if is_best else "#1a1a1a"
            bold = is_best
        ax.set_title(title, fontsize=9, color=color,
                     fontweight="bold" if bold else "normal")
        ax.axis("off")
    fig.tight_layout()
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.grid_out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.grid_out}")

    # --- chamfer-vs-L curve -------------------------------------------------
    Ls = [r["L"] for r in rows]
    chs = [r["chamfer_biggest"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(Ls, chs, "o-", color="#1a4f8a", lw=1.5, ms=5)
    ax.set_xlabel("PE bands L  (multires)")
    ax.set_ylabel("Chamfer")
    ax.set_xticks(Ls)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.curve_out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.curve_out}")


if __name__ == "__main__":
    main()
