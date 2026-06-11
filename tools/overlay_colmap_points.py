"""Overlay COLMAP sparse points on a few NSVF training views as a PNG grid.

Usage:
  python tools/overlay_colmap_points.py \
      --scene  data/tnt/Barn \
      --points data/tnt/Barn/colmap/sparse/0/points3D.bin \
      --out    outputs/.../overlay.png \
      [--views 0 96 192 288]  [--min-track 3]  [--max-err 1.5]

Filters points by track length and reprojection error, colours them on the
image by 1/error, and prints simple "kept/total" stats.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio


def read_points3d_full(path: Path):
    """Return dict with xyz (N,3), err (N,), track_len (N,)."""
    xyz, err, tlen = [], [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                                # point3D_id
            xyz.append(struct.unpack("<3d", f.read(24)))
            f.read(3)                                # rgb
            err.append(struct.unpack("<d", f.read(8))[0])
            tl = struct.unpack("<Q", f.read(8))[0]
            tlen.append(tl)
            f.read(tl * 8)                           # (image_id, point2D_idx) pairs
    return {
        "xyz":  np.asarray(xyz,  dtype=np.float32),
        "err":  np.asarray(err,  dtype=np.float32),
        "tlen": np.asarray(tlen, dtype=np.int32),
    }


def project(pts_world, c2w, K):
    R_c2w = c2w[:3, :3]; t = c2w[:3, 3]
    R_w2c = R_c2w.T
    Xc = (R_w2c @ (pts_world - t).T).T          # (N, 3) in camera frame
    z  = Xc[:, 2]
    visible = z > 1e-3
    uv = (K @ Xc[visible].T).T
    uv = uv[:, :2] / uv[:, 2:]
    return uv, visible, Xc[visible, 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",   type=Path, required=True)
    ap.add_argument("--points",  type=Path, required=True)
    ap.add_argument("--out",     type=Path, required=True)
    ap.add_argument("--views",   type=int, nargs="+", default=None,
                    help="train view indices (0..N-1) to overlay; default: 4 evenly spaced")
    ap.add_argument("--min-track", type=int,   default=3)
    ap.add_argument("--max-err",   type=float, default=1.5)
    ap.add_argument("--marker-size", type=float, default=2.0)
    args = ap.parse_args()

    info = read_points3d_full(args.points)
    n_total = len(info["xyz"])
    keep = (info["tlen"] >= args.min_track) & (info["err"] <= args.max_err)
    pts  = info["xyz"][keep]; err = info["err"][keep]; tlen = info["tlen"][keep]
    print(f"[overlay] {n_total} total -> {len(pts)} kept "
          f"(track>={args.min_track}, err<={args.max_err})")
    print(f"[overlay] track len: min={info['tlen'].min()} med={int(np.median(info['tlen']))} "
          f"max={info['tlen'].max()}")
    print(f"[overlay] err: min={info['err'].min():.3f} med={np.median(info['err']):.3f} "
          f"max={info['err'].max():.3f}")

    pose_paths = sorted((args.scene / "pose").glob("0_*.txt"))
    K = np.loadtxt(args.scene / "intrinsics.txt", dtype=np.float64)[:3, :3]
    n_views = len(pose_paths)
    if args.views is None:
        args.views = list(np.linspace(0, n_views - 1, 4, dtype=int))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), squeeze=False)
    axes = axes.flatten()
    for ax, vi in zip(axes, args.views):
        if vi >= n_views:
            ax.axis("off"); continue
        pp = pose_paths[vi]
        c2w = np.loadtxt(pp, dtype=np.float64).reshape(4, 4)
        img_path = args.scene / "rgb" / (pp.stem + ".png")
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[-1] == 4: img = img[..., :3]
        H, W = img.shape[:2]

        uv, vis, z = project(pts.astype(np.float64), c2w, K)
        on_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        uv = uv[on_img]; z = z[on_img]
        ax.imshow(img)
        if len(uv):
            sc = ax.scatter(uv[:, 0], uv[:, 1], c=1.0 / z, cmap="plasma",
                            s=args.marker_size, alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, fraction=0.03, label="1/depth")
        ax.set_title(f"view {vi}  ({pp.stem})  "
                     f"{int(vis.sum())}/{len(pts)} in front, "
                     f"{len(uv)} in image")
        ax.axis("off")
    fig.suptitle(f"COLMAP sparse overlay  "
                 f"|  {len(pts)} pts after track>={args.min_track}, err<={args.max_err}  "
                 f"|  {n_total} total")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlay] -> {args.out}")


if __name__ == "__main__":
    main()
