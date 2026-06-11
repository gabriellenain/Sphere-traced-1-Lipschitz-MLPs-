#!/usr/bin/env python3
"""5x5 grid of shaded renders: GT Buddha + every buddha_sweep config.

Cell 0 is the ground-truth mesh; the remaining 24 cells are one shaded view of
each swept config's predicted (marching-cubes) mesh, captioned with its Chamfer
distance. All meshes are rendered with the same orthographic camera so shape
quality is directly comparable.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("outputs/buddha_sweep")
GT_MESH = Path("data/gt_meshes/happy_buddha_norm.ply")


def render_shaded(mesh: trimesh.Trimesh, res: int, az: float, el: float,
                  bound: np.ndarray | None = None) -> np.ndarray:
    """Orthographic head-light render of a mesh; light grey clay on white."""
    a, e = np.radians(az), np.radians(el)
    fwd = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)

    lo, hi = (mesh.bounds if bound is None else bound)
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
        n[(n * fwd).sum(-1) > 0] *= -1               # face the camera
        key = -fwd
        fill = right * 0.6 + upv * 0.5; fill /= np.linalg.norm(fill)
        sh = (0.75 * (n @ key).clip(0, 1) +
              0.30 * (n @ fill).clip(0, 1) + 0.12).clip(0, 1)
        img[idr] = sh[:, None] * np.array([0.80, 0.78, 0.74], np.float32)
    return img.reshape(res, res, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=420, help="per-cell render side")
    ap.add_argument("--az", type=float, default=35.0)
    ap.add_argument("--el", type=float, default=15.0)
    ap.add_argument("--results", type=Path, default=SWEEP / "results.csv")
    ap.add_argument("--mesh-name", default="pred_mesh.ply",
                    help="mesh file to render in each config dir")
    ap.add_argument("--out", type=Path, default=Path("figs/buddha_sweep_grid.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.results)))
    rows.sort(key=lambda r: (r["encoding"], int(r["width"]), int(r["depth"])))
    print(f"{len(rows)} configs in results.csv")

    # render GT first to fix a common camera bound for every cell
    gt = trimesh.load(GT_MESH, process=False)
    bound = gt.bounds.copy()
    cells = [("GT", render_shaded(gt, args.res, args.az, args.el, bound), None)]
    for r in rows:
        mp = Path(r["out_dir"]) / args.mesh_name
        enc = "PE" if r["encoding"] == "pe" else "no-PE"
        name = f"{enc}  W{r['width']} D{r['depth']}"
        if not mp.exists():
            cells.append((name, np.ones((args.res, args.res, 3), np.float32), None))
            print(f"  missing mesh: {mp}")
            continue
        m = trimesh.load(mp, process=False)
        cells.append((name, render_shaded(m, args.res, args.az, args.el, bound),
                       float(r["chamfer"])))
        print(f"  rendered {name}  chamfer={float(r['chamfer']):.5f}")

    n = len(cells)
    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(2.5 * side, 2.7 * side))
    best = min((c[2] for c in cells if c[2] is not None), default=None)
    for ax, cell in zip(axes.flat, cells + [("", None, None)] * (side * side - n)):
        name, img, ch = cell
        if img is None:
            ax.axis("off"); continue
        ax.imshow(img)
        if ch is None:
            title = name
            color = "#1a1a1a"
        else:
            title = f"{name}\nchamfer {ch:.5f}"
            color = "#1a7a1a" if ch == best else "#1a1a1a"
        ax.set_title(title, fontsize=8, color=color,
                     fontweight="bold" if ch == best else "normal")
        ax.axis("off")
    fig.suptitle("Happy Buddha SDF regression -- GT + 24 swept configs "
                 "(green = best Chamfer)", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
