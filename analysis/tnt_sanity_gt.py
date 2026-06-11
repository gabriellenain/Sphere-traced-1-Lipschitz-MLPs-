#!/usr/bin/env python3
"""TnT eval sanity check: push the GT cloud into the COLMAP-SfM frame.

We take the official GT point cloud and apply inv(<scene>_trans.txt) to move it
*out* of the GT-LiDAR frame and *into* the COLMAP-SfM frame. Feeding the result
back through `analysis/eval_tnt_official.py --frame colmap-sfm` re-applies <scene>_trans.txt,
landing GT exactly back on GT. The F-score should therefore be ~1.0 — any
shortfall is a bug in the crop / transform / EvaluateHisto path, not in a recon.

Output: outputs/sanity_tnt_gt_in_colmap_frame.ply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gt_ply = args.gt_dir / f"{args.scene}.ply"
    trans_p = args.gt_dir / f"{args.scene}_trans.txt"

    gt = o3d.io.read_point_cloud(str(gt_ply))
    T = np.loadtxt(trans_p)              # colmap-sfm -> GT-LiDAR
    assert T.shape == (4, 4)
    Tinv = np.linalg.inv(T)             # GT-LiDAR -> colmap-sfm
    gt.transform(Tinv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out), gt)
    aabb = gt.get_axis_aligned_bounding_box()
    print(f"wrote {args.out} verts {len(gt.points)} "
          f"AABB {aabb.min_bound} {aabb.max_bound}", flush=True)


if __name__ == "__main__":
    main()
