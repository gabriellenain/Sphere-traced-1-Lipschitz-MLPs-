#!/usr/bin/env python3
"""Convert ACMMP (Xu & Tao) per-view .dmb output into the depth_est/*.pfm +
confidence/*.npy layout consumed by lip_tracer.geomvs.load_mvsformer_depths_idr
and tools/carve_visual_hull_with_mvsformer_scan24.py.

ACMMP writes, per reference view, under <dense>/ACMMP/2333_<id>/:
    depths_geom.dmb   geometric-consistency-filtered z-depth (invalid px = 0)
    costs.dmb         per-pixel aggregated matching cost (lower = better)
    normals.dmb       (unused here)

Because the ACMMP `cams/` were staged in the IDR-normalised frame (same as the
MVSFormer++ pipeline, see tools/precompute_mvsformer_depths.py:_stage_idr), the
emitted z-depths are already in IDR-normalised cam-z — no scale alignment needed,
exactly like the MVSFormer++ depths the carve loader expects.

Confidence is derived from the ACMMP cost and gated by the geometric-consistency
mask (depth>0):  conf = clip(1 - cost / cost_scale, 0, 1) * (depth > 0).
This is a non-learning, DTU-leakage-free analogue of MVSFormer++'s probability.
The downstream carve's votes-req multi-view agreement is the dominant filter, so
the exact cost->conf mapping is not critical (conf-thr 0.5 keeps cost<cost_scale/2
geometric-consistent pixels).

Output: <out>/<scan>/{depth_est/<id>.pfm, confidence/<id>.npy}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_dmb(path: Path) -> np.ndarray:
    """Read a Gipuma/ACMMP .dmb map: int32 (type,h,w,nb) header + float32 data."""
    with open(path, "rb") as f:
        type_, h, w, nb = np.fromfile(f, dtype="<i4", count=4)
        data = np.fromfile(f, dtype="<f4", count=int(h) * int(w) * int(nb))
    arr = data.reshape((int(h), int(w)) if nb == 1 else (int(h), int(w), int(nb)))
    return arr


def write_pfm(path: Path, image: np.ndarray) -> None:
    """Write a single-channel little-endian PFM matching geomvs._read_pfm."""
    image = np.asarray(image, dtype=np.float32)
    assert image.ndim == 2, "grayscale PFM only"
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        f.write(b"-1.0\n")  # negative scale => little-endian
        np.flipud(image).astype("<f4").tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense", type=Path, required=True,
                    help="ACMMP dense folder containing ACMMP/2333_<id>/ subdirs")
    ap.add_argument("--out", type=Path, required=True,
                    help="output <out>/<scan>/{depth_est,confidence} root")
    ap.add_argument("--scan", type=str, required=True, help="scan name, e.g. scan24")
    ap.add_argument("--cost-scale", type=float, default=2.0,
                    help="cost mapped to confidence via clip(1 - cost/scale, 0, 1)")
    args = ap.parse_args()

    acmmp_root = args.dense / "ACMMP"
    prob_dirs = sorted(acmmp_root.glob("2333_*"))
    if not prob_dirs:
        raise FileNotFoundError(f"no 2333_* result dirs under {acmmp_root}")

    scan_out = args.out / args.scan
    (scan_out / "depth_est").mkdir(parents=True, exist_ok=True)
    (scan_out / "confidence").mkdir(parents=True, exist_ok=True)

    n = 0
    for pd in prob_dirs:
        vid = pd.name.split("_")[-1]  # zero-padded 8-digit id
        depth_dmb = pd / "depths_geom.dmb"
        if not depth_dmb.exists():
            depth_dmb = pd / "depths.dmb"  # fall back if geom pass absent
        cost_dmb = pd / "costs.dmb"
        if not depth_dmb.exists():
            print(f"  [skip] {pd.name}: no depth dmb")
            continue

        depth = read_dmb(depth_dmb).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0] = 0.0

        if cost_dmb.exists():
            cost = read_dmb(cost_dmb).astype(np.float32)
            # ACMMP's costs.dmb can be at a different multi-scale resolution than
            # depths_geom.dmb; resize to the depth grid so the gate aligns.
            if cost.shape != depth.shape:
                import cv2
                cost = cv2.resize(cost, (depth.shape[1], depth.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            cost[~np.isfinite(cost)] = args.cost_scale
            conf = np.clip(1.0 - cost / args.cost_scale, 0.0, 1.0)
        else:
            conf = np.ones_like(depth)
        conf = (conf * (depth > 1e-6)).astype(np.float32)

        write_pfm(scan_out / "depth_est" / f"{vid}.pfm", depth)
        np.save(scan_out / "confidence" / f"{vid}.npy", conf)
        n += 1
        if n % 8 == 1:
            valid = depth > 1e-6
            zr = depth[valid] if valid.any() else np.array([0.0])
            print(f"  [conv] {args.scan} view {vid}: valid={100*valid.mean():.1f}%  "
                  f"z=[{zr.min():.3f},{zr.max():.3f}]  conf_mean={conf[valid].mean() if valid.any() else 0:.2f}")

    print(f"[done] converted {n} views -> {scan_out}")


if __name__ == "__main__":
    main()
