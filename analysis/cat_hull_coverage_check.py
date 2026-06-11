#!/usr/bin/env python3
"""Minimal HH-coverage test for a hull/SDF grid against object masks."""
import argparse
import sys
from pathlib import Path

# Importable when run as `python analysis/<this>.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    ap.add_argument("--show-views", type=str, default="",
                    help="comma-separated explicit view ids to render")
    ap.add_argument("--dump-good", type=Path, default=None,
                    help="write kept view indices (coverage >= --good-thresh) here")
    ap.add_argument("--good-thresh", type=float, default=0.90)
    args = ap.parse_args()

    sdf = np.load(args.grid)
    res = sdf.shape[0]
    occ = sdf < 0.0
    print(f"grid {args.grid}  res={res}  occupied={int(occ.sum())}")

    lin = np.linspace(-args.bound, args.bound, res, dtype=np.float32)
    iz, iy, ix = np.nonzero(occ)
    pts = np.stack([lin[ix], lin[iy], lin[iz]], axis=-1).astype(np.float64)

    v = load_views(args.scene, down=args.down)
    K = v["K"].numpy().astype(np.float64)
    c2w = v["c2w"].numpy().astype(np.float64)
    masks = v["masks"].numpy().astype(bool)
    H, W = int(v["H"]), int(v["W"])
    n = len(c2w)

    cover_frac = np.zeros(n)
    overlays = {}
    if args.show_views:
        show_idx = set(int(s) for s in args.show_views.split(","))
    else:
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
        cover_frac[i] = int((m & covered).sum()) / mtot
        if i in show_idx:
            img = v["images"][i].numpy() if "images" in v else None
            ov = np.zeros((H, W, 3), dtype=np.float32)
            if img is not None:
                ov = img.copy() * 0.5
            ov[m & covered] = (0.2, 0.8, 0.2)
            ov[m & ~covered] = (0.95, 0.1, 0.1)
            ov[~m & covered] = (0.55, 0.55, 0.75)
            overlays[i] = ov

    valid = ~np.isnan(cover_frac)
    print(f"\nper-view object coverage: "
          f"min={np.nanmin(cover_frac)*100:.2f}%  "
          f"mean={np.nanmean(cover_frac)*100:.2f}%  "
          f"median={np.nanmedian(cover_frac[valid])*100:.2f}%")
    worst = np.argsort(np.where(valid, cover_frac, 2.0))[:12]
    print("worst views (coverage %):")
    for w in worst:
        print(f"  view {w:3d}: {cover_frac[w]*100:6.2f}%  "
              f"missing={int(masks[w].sum()*(1-cover_frac[w]))} px")
    # how many views below thresholds
    for thr in (0.90, 0.92, 0.95):
        print(f"  views < {thr*100:.0f}%: {int((cover_frac[valid] < thr).sum())}")

    if args.dump_good is not None:
        keep = [int(i) for i in range(n) if valid[i] and cover_frac[i] >= args.good_thresh]
        args.dump_good.write_text("\n".join(str(i) for i in keep) + "\n")
        print(f"\nkept {len(keep)}/{n} views (cov >= {args.good_thresh:.2f}); "
              f"dropped {n - len(keep)} -> {args.dump_good}")
        print("dropped:", [int(i) for i in range(n) if not (valid[i] and cover_frac[i] >= args.good_thresh)])

    idxs = sorted(overlays)
    cols = 3
    rows = int(np.ceil(len(idxs) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for k, i in enumerate(idxs):
        a = ax[k // cols, k % cols]
        a.imshow(np.clip(overlays[i], 0, 1))
        a.set_title(f"view {i}  cov={cover_frac[i]*100:.1f}%", fontsize=10)
    fig.suptitle(f"red=object MISSING from hull, green=covered, grey=hull beyond mask\n"
                 f"{args.grid.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
