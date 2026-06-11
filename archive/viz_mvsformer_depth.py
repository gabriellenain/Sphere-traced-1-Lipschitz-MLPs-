#!/usr/bin/env python3
"""Make a quick PNG mosaic of MVSFormer++ depth + confidence per view."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path.home() / "scratch" / "MVSFormerPlusPlus"))
from datasets.data_io import read_pfm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, required=True,
                    help="e.g. .../mvsformer_depth_1536x1152_1536x1152/scan65")
    ap.add_argument("--out", type=Path, default=Path("mvsformer_viz.png"))
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--conf-thr", type=float, default=0.5)
    args = ap.parse_args()

    pfms = sorted((args.scan_dir / "depth_est").glob("*.pfm"))
    rows = (len(pfms) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(2.4 * args.cols, 2.0 * rows))
    axes = np.atleast_2d(axes).reshape(rows, args.cols)

    for ax in axes.ravel():
        ax.axis("off")

    for i, p in enumerate(pfms):
        depth = np.asarray(read_pfm(str(p))[0], dtype=np.float32)
        conf = np.load(args.scan_dir / "confidence" / f"{p.stem}.npy")
        if conf.dtype == np.uint8:
            conf = conf.astype(np.float32) / 255.0
        d = depth.copy()
        d[conf < args.conf_thr] = np.nan
        finite = d[np.isfinite(d)]
        vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))
        ax = axes[i // args.cols, i % args.cols]
        ax.imshow(d, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(p.stem, fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


def stats(args):
    import cv2
    pfms = sorted((args.scan_dir / "depth_est").glob("*.pfm"))
    scene = args.scan_dir.parent.parent  # .../dtu/scan65
    mask_dir = scene / "mask"

    per_view = []
    all_c_obj, all_d_obj = [], []
    for p in pfms:
        d = np.asarray(read_pfm(str(p))[0], dtype=np.float32)
        c = np.load(args.scan_dir / "confidence" / f"{p.stem}.npy")
        if c.dtype == np.uint8:
            c = c.astype(np.float32) / 255.0
        H, W = d.shape
        mp = mask_dir / f"{int(p.stem):03d}.png"
        from PIL import Image
        m = cv2.resize(
            (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.uint8),
            (W, H), cv2.INTER_NEAREST).astype(bool)
        valid = (c > args.conf_thr) & (d > 1e-3) & m
        per_view.append(100 * valid.sum() / max(m.sum(), 1))
        all_c_obj.append(c[m]); all_d_obj.append(d[valid] if valid.any() else np.array([]))

    all_c = np.concatenate(all_c_obj)
    all_d = np.concatenate([x for x in all_d_obj if x.size])
    print(f"conf_thr={args.conf_thr}  views={len(pfms)}")
    print(f"valid% inside mask  mean={np.mean(per_view):.1f}  "
          f"min={np.min(per_view):.1f}  max={np.max(per_view):.1f}")
    print(f"depth range (valid obj): [{all_d.min():.4f}, {all_d.max():.4f}]  "
          f"median={np.median(all_d):.4f}")
    print("conf histogram (inside mask):")
    for lo, hi in [(0,.1),(.1,.2),(.2,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.01)]:
        pct = 100 * ((all_c >= lo) & (all_c < hi)).mean()
        print(f"  [{lo:.1f},{hi:.1f})  {pct:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("mvsformer_viz.png"))
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.stats:
        stats(args)
    else:
        _viz(args)


def _viz(args):
    pfms = sorted((args.scan_dir / "depth_est").glob("*.pfm"))
    rows = (len(pfms) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(2.4 * args.cols, 2.0 * rows))
    axes = np.atleast_2d(axes).reshape(rows, args.cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i, p in enumerate(pfms):
        depth = np.asarray(read_pfm(str(p))[0], dtype=np.float32)
        conf = np.load(args.scan_dir / "confidence" / f"{p.stem}.npy")
        if conf.dtype == np.uint8:
            conf = conf.astype(np.float32) / 255.0
        d = depth.copy()
        d[conf < args.conf_thr] = np.nan
        finite = d[np.isfinite(d)]
        vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))
        ax = axes[i // args.cols, i % args.cols]
        ax.imshow(d, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(p.stem, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
