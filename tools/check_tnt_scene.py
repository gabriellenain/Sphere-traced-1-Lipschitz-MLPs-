#!/usr/bin/env python3
"""Coherence check for an NSVF-format TnT scene before it replaces the old one.

Validates that intrinsics + poses + bbox are mutually consistent the way
lip_tracer's loader will interpret them. Exits non-zero (so a driver script can
refuse the swap) if anything looks wrong. Checks:

  1. equal #poses / #rgb, at least MIN_VIEWS registered frames
  2. finite, proper-rotation (det~+1) camera-to-world matrices
  3. intrinsics positive; principal point inside the image
  4. normalised camera radii sane (cameras sit outside the unit cube, looking in)
  5. framing: the bbox centre (= object centre) projects in-front + in-bounds
     for most views -- the real test that K, poses and bbox agree
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MIN_VIEWS = 20
MIN_FRAMED_FRAC = 0.6


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    args = ap.parse_args()
    s = args.scene

    for f in ("intrinsics.txt", "bbox.txt"):
        if not (s / f).exists():
            fail(f"missing {f}")
    poses = sorted((s / "pose").glob("0_*.txt"))
    rgbs = sorted((s / "rgb").glob("0_*.png"))
    if len(poses) != len(rgbs):
        fail(f"#poses {len(poses)} != #rgb {len(rgbs)}")
    if len(poses) < MIN_VIEWS:
        fail(f"only {len(poses)} registered views (< {MIN_VIEWS}); SfM likely failed")

    K = np.loadtxt(s / "intrinsics.txt")[:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not (fx > 0 and fy > 0):
        fail(f"non-positive focal length fx={fx} fy={fy}")

    from PIL import Image
    W, H = Image.open(rgbs[0]).size
    if not (0 < cx < W and 0 < cy < H):
        fail(f"principal point ({cx:.1f},{cy:.1f}) outside image {W}x{H}")

    bbox = np.loadtxt(s / "bbox.txt")
    bb_min, bb_max = bbox[:3], bbox[3:6]
    center = 0.5 * (bb_min + bb_max)
    scale = float(np.max(0.5 * (bb_max - bb_min)))
    if not np.isfinite(scale) or scale <= 0:
        fail(f"degenerate bbox extent (scale={scale})")

    radii, framed, det_bad = [], 0, 0
    for pp in poses:
        c2w = np.loadtxt(pp).reshape(4, 4)
        if not np.isfinite(c2w).all():
            fail(f"non-finite pose {pp.name}")
        R = c2w[:3, :3]
        if abs(np.linalg.det(R) - 1.0) > 1e-2:
            det_bad += 1
        C = c2w[:3, 3]
        radii.append(np.linalg.norm((C - center) / scale))
        # project object centre (world `center`) into this camera
        w2c = np.linalg.inv(c2w)
        Xc = w2c[:3, :3] @ center + w2c[:3, 3]
        if Xc[2] > 0:
            u = fx * Xc[0] / Xc[2] + cx
            v = fy * Xc[1] / Xc[2] + cy
            if 0 <= u < W and 0 <= v < H:
                framed += 1

    if det_bad:
        fail(f"{det_bad}/{len(poses)} poses are not proper rotations (det != 1)")

    med_r = float(np.median(radii))
    framed_frac = framed / len(poses)
    print(f"  views={len(poses)}  K=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f})  {W}x{H}")
    print(f"  median normalised cam radius={med_r:.2f}  (sane ~1-10)")
    print(f"  object-centre framed in {framed_frac*100:.0f}% of views "
          f"(need >= {MIN_FRAMED_FRAC*100:.0f}%)")

    if not (0.3 < med_r < 50):
        fail(f"camera radii implausible (median {med_r:.2f}); bbox/poses gauge mismatch")
    if framed_frac < MIN_FRAMED_FRAC:
        fail(f"object centre framed in only {framed_frac*100:.0f}% of views; "
             f"bbox probably bounds the whole scene, not the object -- "
             f"re-run colmap_to_nsvf.py with a larger --clip or edit bbox.txt")

    print("  [OK] scene is coherent")


if __name__ == "__main__":
    main()
