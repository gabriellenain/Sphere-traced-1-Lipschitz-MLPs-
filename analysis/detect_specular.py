#!/usr/bin/env python3
"""Robust specular-highlight detector.

A specular highlight is a *localized* bright spot where the illuminant colour
adds roughly equally to all three channels, washing out the surface albedo.
Three conditions must hold together; any one alone gives false positives:

  1. Bright          -- HSV value (max channel) is high.
  2. Desaturated     -- (max-min)/max is low: the albedo is washed out.
  3. Locally peaked  -- it stands out from the surrounding *diffuse* baseline.

Condition 3 is what separates a true highlight from intrinsically bright,
pale albedo (e.g. a light-blue glaze or white snow), which is bright and
desaturated everywhere but is *flat*. We test it with a white top-hat on the
min channel:

    min channel        ~ illuminant-only "specular-free" proxy (Tan/Yoon):
                         specular lifts every channel, so it lifts the min;
                         saturated diffuse colour keeps the min low.
    opening(min, r)     = the slowly-varying diffuse min level (large radius r
                          removes bright structures smaller than r).
    tophat = min-opening = the residual spike left by compact highlights.

Flat bright albedo: its min channel is uniformly high, the opening tracks it,
tophat ~ 0 -> rejected. Sharp glint: min spikes above baseline -> kept.

Usage:
    python analysis/detect_specular.py IMG [IMG ...] -o OUTDIR
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def specular_mask(rgb, v_thr=0.80, sat_thr=0.35, tophat_thr=0.06, radius=15):
    """rgb: HxWx3 float in [0,1]. Returns bool mask of specular pixels.

    v_thr      minimum brightness (HSV value).
    sat_thr    maximum saturation (highlights are washed out).
    tophat_thr minimum local prominence of the min channel above its diffuse
               baseline -- the test that rejects flat bright albedo.
    radius     structuring-element radius (px); larger than any real highlight,
               smaller than broad bright regions.
    """
    rgb = rgb.astype(np.float32)
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    sat = np.where(cmax > 1e-6, (cmax - cmin) / np.maximum(cmax, 1e-6), 0.0)

    # White top-hat on the specular-free (min) channel: local bright residual.
    fp = _disk(radius)
    tophat = cmin - ndimage.grey_opening(cmin, footprint=fp)

    return (cmax >= v_thr) & (sat <= sat_thr) & (tophat >= tophat_thr)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--outdir", default="outputs/specular")
    ap.add_argument("--v-thr", type=float, default=0.80)
    ap.add_argument("--sat-thr", type=float, default=0.35)
    ap.add_argument("--tophat-thr", type=float, default=0.06)
    ap.add_argument("--radius", type=int, default=15)
    ap.add_argument("--mask-dir", default=None,
                    help="dir of DTU NNN.png masks; restrict detection to foreground")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for p in args.images:
        p = Path(p)
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        mask = specular_mask(rgb, args.v_thr, args.sat_thr, args.tophat_thr, args.radius)
        if args.mask_dir:
            # DTU masks are indexed NNN.png; image stems are 000000.png -> 000.
            mp = Path(args.mask_dir) / f"{int(p.stem):03d}.png"
            fg = np.asarray(Image.open(mp).convert("L"), np.float32) / 255.0
            mask &= fg > 0.5

        frac = mask.mean()
        overlay = (rgb * 255).astype(np.uint8).copy()
        overlay[mask] = [255, 0, 0]

        Image.fromarray((rgb * 255).astype(np.uint8)).save(out / f"{p.stem}_rgb.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{p.stem}_mask.png")
        Image.fromarray(overlay).save(out / f"{p.stem}_overlay.png")
        print(f"{p.name}: {frac*100:5.2f}% specular  ({mask.sum()} px)")


if __name__ == "__main__":
    main()
