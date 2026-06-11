#!/usr/bin/env python3
"""Minimal HH-coverage test for a hull/SDF grid against object masks.

Projects every occupied voxel into each view, compares the resulting silhouette
to the object mask, and reports the fraction of object-mask pixels the hull
covers. A genuine hole (e.g. carved-away cab roof) shows up as a large
contiguous red region; sub-pixel discretization speckle is removed with a 1px
morphological close so it is NOT mistaken for a hole.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("data/tnt/Caterpillar"))
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("/tmp/cat_hh_coverage.png"))
    ap.add_argument("--n-show", type=int, default=8)
    args = ap.parse_args()

    sdf = np.load(args.grid)
    res = sdf.shape[0]
    occ = sdf < 0.0
    print(f"grid {args.grid}  res={res}  occupied={int(occ.sum())}")

    lin = np.linspace(-args.bound, args.bound, res, dtype=np.float32)
    iz, iy, ix = np.nonzero(occ)
    pts = np.stack([lin[ix], lin[iy], lin[iz]], axis=-1).astype(np.float64)
    print(f"projecting {len(pts)} voxels")

    v = load_views(args.scene, down=args.down)
    K = v["K"].numpy().astype(np.float64)
    c2w = v["c2w"].numpy().astype(np.float64)
    masks = v["masks"].numpy().astype(bool)
    H, W = int(v["H"]), int(v["W"])
    n = len(c2w)

    cover_frac = np.zeros(n)
    overlays = {}
    show_idx = set(np.linspace(0, n - 1, args.n_show).round().astype(int).tolist())

    for i in range(n):
        R = c2w[i, :3, :3]
        c = c2w[i, :3, 3]
        cam = (pts - c[None]) @ R
        z = cam[:, 2]
        front = z > 1e-5
        u = cam[:, 0] / np.where(front, z, 1.0) * K[i, 0, 0] + K[i, 0, 2]
        vv = cam[:, 1] / np.where(front, z, 1.0) * K[i, 1, 1] + K[i, 1, 2]
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(vv).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        covered = np.zeros((H, W), dtype=bool)
        covered[vi[inb], ui[inb]] = True
        covered = binary_closing(covered, iterations=1)

        m = masks[i]
        mtot = int(m.sum())
        if mtot == 0:
            cover_frac[i] = np.nan
            continue
        hit = int((m & covered).sum())
        cover_frac[i] = hit / mtot
        if i in show_idx:
            ov = np.zeros((H, W, 3), dtype=np.float32)
            ov[m & covered] = (0.2, 0.8, 0.2)      # covered object  -> green
            ov[m & ~covered] = (0.9, 0.1, 0.1)      # missing object  -> red (hole)
            ov[~m & covered] = (0.55, 0.55, 0.7)    # hull beyond mask -> grey
            overlays[i] = ov

    valid = ~np.isnan(cover_frac)
    print(f"\nper-view object coverage: "
          f"min={np.nanmin(cover_frac)*100:.2f}%  "
          f"mean={np.nanmean(cover_frac)*100:.2f}%  "
          f"median={np.nanmedian(cover_frac[valid])*100:.2f}%")
    worst = np.argsort(np.where(valid, cover_frac, 2.0))[:8]
    print("worst views (coverage %):")
    for w in worst:
        print(f"  view {w:3d}: {cover_frac[w]*100:6.2f}%  "
              f"missing={int(masks[w].sum()*(1-cover_frac[w]))} px")

    idxs = sorted(overlays)
    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for k, i in enumerate(idxs):
        a = ax[k // cols, k % cols]
        a.imshow(overlays[i])
        a.set_title(f"view {i}  cov={cover_frac[i]*100:.1f}%", fontsize=10)
    fig.suptitle(f"HH coverage (green=object covered, red=object MISSING, "
                 f"grey=hull beyond mask)\n{args.grid.name}  "
                 f"mean={np.nanmean(cover_frac)*100:.2f}%", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
