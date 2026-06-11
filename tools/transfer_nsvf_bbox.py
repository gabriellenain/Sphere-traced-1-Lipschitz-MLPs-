#!/usr/bin/env python3
"""Transfer the curated NSVF object bbox into a fresh-COLMAP scene's gauge.

A from-scratch COLMAP SfM on the raw frames gives better cameras but an
arbitrary world gauge, and its sparse cloud spans the whole *scene* -- so an
auto bbox bounds the background, not the object. The NSVF release already ships
a hand-tuned object bbox, just in a different gauge. Since both reconstructions
are the *same physical cameras*, we recover the similarity (scale+R+t) between
the two camera-centre sets (matched by original frame index) via Umeyama, then
map NSVF's bbox corners into the COLMAP gauge and write bbox.txt.

Usage:
  python tools/transfer_nsvf_bbox.py --nsvf data/tnt/Barn \
         --colmap data/tnt/Barn_raw/colmap --out data/tnt/Barn_colmap
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np


def nsvf_centers(nsvf: Path) -> dict:
    """frame_idx -> camera centre (c2w translation)."""
    out = {}
    for p in glob.glob(str(nsvf / "pose" / "0_*.txt")):
        fr = int(os.path.basename(p)[:-4].split("_")[-1])
        c2w = np.loadtxt(p).reshape(4, 4)
        out[fr] = c2w[:3, 3]
    return out


def colmap_centers(txt: Path) -> dict:
    """frame_idx -> camera centre (-R^T t from world->cam pose)."""
    lines = [l for l in (txt / "images.txt").read_text().splitlines()
             if l and not l.startswith("#")]
    out = {}
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        qw, qx, qy, qz = map(float, t[1:5])
        tv = np.array(list(map(float, t[5:8])))
        name = t[9]
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        n = np.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])
        out[int(m.group(1))] = -R.T @ tv
    return out


def umeyama(src: np.ndarray, dst: np.ndarray):
    """similarity mapping src->dst: returns s, R, t with dst ~= s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = (Dc.T @ Sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsvf", type=Path, required=True, help="NSVF scene dir")
    ap.add_argument("--colmap", type=Path, required=True, help="COLMAP workspace")
    ap.add_argument("--out", type=Path, required=True, help="target _colmap scene")
    args = ap.parse_args()

    nc = nsvf_centers(args.nsvf)
    cc = colmap_centers(args.colmap / "sparse_txt")
    common = sorted(set(nc) & set(cc))
    if len(common) < 10:
        raise SystemExit(f"only {len(common)} matched frames -- cannot align")
    src = np.array([nc[f] for f in common])   # NSVF
    dst = np.array([cc[f] for f in common])   # COLMAP

    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
    print(f"  matched {len(common)} cameras  scale={s:.4f}  "
          f"align RMS={resid.mean():.4g} (max {resid.max():.4g}) in COLMAP units")

    bbox = np.loadtxt(args.nsvf / "bbox.txt")
    bmin, bmax = bbox[:3], bbox[3:6]
    corners = np.array([[x, y, z] for x in (bmin[0], bmax[0])
                        for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])])
    cc_col = s * (corners @ R.T) + t          # corners in COLMAP gauge
    lo, hi = cc_col.min(0), cc_col.max(0)
    voxel = float((hi - lo).max()) / 128.0

    with open(args.out / "bbox.txt", "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in (*lo, *hi, voxel)) + "\n")
    print(f"  wrote object bbox -> {args.out/'bbox.txt'}")
    print(f"  min={lo.round(3).tolist()} max={hi.round(3).tolist()} "
          f"extent={(hi-lo).round(3).tolist()}")


if __name__ == "__main__":
    main()
