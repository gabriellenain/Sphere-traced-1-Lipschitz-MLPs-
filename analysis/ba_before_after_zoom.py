#!/usr/bin/env python3
"""2x2 before/after BA figure (phong top, normals bottom) with a zoom inset.

Reads the per-view sphere-traced buffers (st_view<V>_world_{phong,normals}.png)
that render_st_blender.slurm / sphere_traced_screen_mesh.py dump for a BEFORE dir
(calibrated poses + pre-BA theta) and an AFTER dir (BA poses/intrinsics + BA
theta), and composites the paper figure: each panel is cropped to the subject,
with a gold zoom box and a magnified corner inset + connector lines.

The inset region defaults to the location of largest before->after change in the
normals buffer (within a central-lower search band, where drapery/silhouette
refine), so it lands where there is actually something to see; override with
--inset. Crop defaults to the full frame; pass --crop for a portrait framing.

Example (Ignatius view 52):
    python analysis/ba_before_after_zoom.py \
        --before <run>/ba_compare_view52/before \
        --after  <run>/ba_compare_view52/after_percam_intr \
        --view 52 --crop 330,980,150,950 \
        --out   <run>/ba_compare_view52/fig_before_after_zoom.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch


def _load(d: Path, view: int, kind: str) -> np.ndarray:
    p = d / f"st_view{view}_world_{kind}.png"
    if not p.exists():
        raise FileNotFoundError(p)
    return np.asarray(Image.open(p).convert("RGB"))


def _auto_inset(diff: np.ndarray, crop, side: int):
    """Max-|Δ| square within the central-lower band of the crop."""
    x0, x1, y0, y1 = crop
    sx0, sx1 = int(x0 + 0.05 * (x1 - x0)), int(x1 - 0.05 * (x1 - x0))
    sy0, sy1 = int(y0 + 0.45 * (y1 - y0)), int(y1 - 0.02 * (y1 - y0))
    best = (-1.0, sx0, sy0)
    for yy in range(sy0, max(sy0 + 1, sy1 - side), 16):
        for xx in range(sx0, max(sx0 + 1, sx1 - side), 16):
            m = diff[yy:yy + side, xx:xx + side].mean()
            if m > best[0]:
                best = (m, xx, yy)
    return best[1], best[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True,
                    help="buffer dirs as columns, in display order (e.g. baseline R,t full)")
    ap.add_argument("--view", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop", type=str, default=None,
                    help="x0,x1,y0,y1 portrait crop (default: full frame)")
    ap.add_argument("--inset", type=str, default=None,
                    help="x0,y0,side of the zoom box (default: auto max-Δ over first vs last)")
    ap.add_argument("--inset-side", type=int, default=None,
                    help="square side in px when auto-placing (default: crop_h/4)")
    args = ap.parse_args()

    # rows = (phong, normals); columns = the N buffer dirs.
    phong = [_load(d, args.view, "phong") for d in args.dirs]
    norm = [_load(d, args.view, "normals") for d in args.dirs]
    H, W, _ = norm[0].shape

    crop = ([int(v) for v in args.crop.split(",")] if args.crop else [0, W, 0, H])
    cx0, cx1, cy0, cy1 = crop
    side = args.inset_side or (cy1 - cy0) // 4
    diff = np.abs(norm[-1].astype(float) - norm[0].astype(float)).mean(2)  # first vs last
    if args.inset:
        ix, iy, side = (int(v) for v in args.inset.split(","))
    else:
        ix, iy = _auto_inset(diff, crop, side)
    # Pick the panel corner with the LEAST foreground (most white/empty space) so
    # the magnified inset never hides the sculpture. Evaluated on the union of ALL
    # buffers' foreground so the inset sits in the same place in every panel.
    iw = 0.40
    fg = np.minimum.reduce([im.min(2) for im in (*phong, *norm)]) < 235
    corners = {"tl": (0.015, 1 - 0.015 - iw),       # top corners only (sky is up)
               "tr": (1 - 0.015 - iw, 1 - 0.015 - iw)}
    def _fg_frac(fx0, fy0):
        px0 = int(cx0 + fx0 * (cx1 - cx0)); px1 = int(cx0 + (fx0 + iw) * (cx1 - cx0))
        py1 = int(cy1 - fy0 * (cy1 - cy0)); py0 = int(cy1 - (fy0 + iw) * (cy1 - cy0))
        return fg[max(py0, 0):py1, max(px0, 0):px1].mean()
    cname = min(corners, key=lambda c: _fg_frac(*corners[c]))
    fx0, fy0 = corners[cname]
    print(f"[fig] {len(args.dirs)} cols  inset x[{ix},{ix+side}] y[{iy},{iy+side}]  "
          f"meanΔnorm(first→last)={diff[iy:iy+side, ix:ix+side].mean():.1f}/255  corner={cname}")

    # inset is above the box → connect the box's top corners to the inset's bottom.
    conns = [((ix, iy), (0, 0)), ((ix + side, iy), (1, 0))]

    def panel(ax, img):
        ax.imshow(img)
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy1, cy0); ax.axis("off")
        ax.add_patch(Rectangle((ix, iy), side, side, ec="gold", fc="none", lw=2.2))
        iax = ax.inset_axes([fx0, fy0, iw, iw])
        iax.imshow(img[iy:iy + side, ix:ix + side])
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_color("gold"); s.set_linewidth(2.2)
        for bxy, ixy in conns:
            ax.add_artist(ConnectionPatch(xyA=bxy, coordsA=ax.transData,
                                          xyB=ixy, coordsB=iax.transAxes,
                                          color="gold", lw=1.1, alpha=0.9))

    n = len(args.dirs)
    fig, axes = plt.subplots(2, n, figsize=(8 * n, 2 * 8 * (cy1 - cy0) / (cx1 - cx0)),
                             squeeze=False)
    for c in range(n):
        panel(axes[0, c], phong[c])
        panel(axes[1, c], norm[c])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, facecolor="white")
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
