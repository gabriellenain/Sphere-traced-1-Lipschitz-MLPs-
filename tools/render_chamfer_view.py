#!/usr/bin/env python3
"""Reproject the DTUeval error point clouds (vis_<scan>_s2d.ply / _d2s.ply) into
a real DTU camera view.

`chamfer_error.png` shows orthographic scatter projections; this renders the SAME
per-point error colours as seen from one of the dataset cameras, z-buffered onto
the image plane (optionally over the photo).

  s2d = pred surface points coloured by ACCURACY    (pred -> GT distance)
  d2s = GT data points   coloured by COMPLETENESS    (GT  -> pred distance)

The DTUeval vis PLYs live in raw DTU world (mm); cameras.npz world_mat_<i> is the
3x4 projection K[R|t] for raw mm coords, so projection is a direct matmul.

Example:
  python tools/render_chamfer_view.py \
      --out-dir outputs/dtu_official_5002886 \
      --scene /scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan63 \
      --view 11
"""
import argparse
from pathlib import Path

import numpy as np


def _read_ply_xyzrgb(path: Path):
    """Minimal reader for the Open3D binary_little_endian PLY DTUeval writes:
    double x,y,z + uchar r,g,b."""
    with open(path, "rb") as fh:
        n = None
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        arr = np.fromfile(fh, dtype=dt, count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    rgb = np.stack([arr["r"], arr["g"], arr["b"]], axis=1).astype(np.uint8)
    return xyz, rgb


def _splat(xyz, rgb, P, H, W, radius, bg=None):
    """Z-buffered point splat into an (H,W,3) image using projection P (3x4)."""
    Xh = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)      # (N,4)
    uvw = Xh @ P.T                                                  # (N,3)
    z = uvw[:, 2]
    front = z > 1e-6
    u = uvw[front, 0] / z[front]
    v = uvw[front, 1] / z[front]
    z = z[front]
    col = rgb[front]
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui = np.rint(u[keep]).astype(np.int64)
    vi = np.rint(v[keep]).astype(np.int64)
    z = z[keep]
    col = col[keep]

    img = (np.zeros((H, W, 3), np.uint8) if bg is None else bg.copy())
    zbuf = np.full((H, W), np.inf, np.float64)
    # paint far -> near so nearer points win (cheap z-buffer over a disk splat)
    order = np.argsort(-z)
    ui, vi, z, col = ui[order], vi[order], z[order], col[order]
    r = radius
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            if du * du + dv * dv > r * r:
                continue
            uu = (ui + du).clip(0, W - 1)
            vv = (vi + dv).clip(0, H - 1)
            better = z < zbuf[vv, uu]
            img[vv[better], uu[better]] = col[better]
            zbuf[vv[better], uu[better]] = z[better]
    return img, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="eval output dir containing vis_<scan>_{s2d,d2s}.ply")
    ap.add_argument("--scene", type=Path, required=True,
                    help="DTU scene dir with cameras.npz + image/")
    ap.add_argument("--view", type=int, required=True, help="camera index i (world_mat_i, image/00000i.png)")
    ap.add_argument("--scan", type=int, default=None,
                    help="scan id (default: parse from vis_*.ply name)")
    ap.add_argument("--radius", type=int, default=2, help="splat radius in px")
    ap.add_argument("--overlay", action="store_true",
                    help="composite over the camera photo (dim) instead of black")
    args = ap.parse_args()

    if args.scan is None:
        cands = sorted(args.out_dir.glob("vis_*_s2d.ply"))
        if not cands:
            raise SystemExit(f"no vis_*_s2d.ply in {args.out_dir}")
        args.scan = int(cands[0].name.split("_")[1])

    cams = np.load(args.scene / "cameras.npz")
    P = cams[f"world_mat_{args.view}"][:3, :4].astype(np.float64)

    from PIL import Image
    img_path = args.scene / "image" / f"{args.view:06d}.png"
    photo = np.array(Image.open(img_path)) if img_path.exists() else None
    if photo is not None:
        H, W = photo.shape[:2]
    else:
        H, W = 1200, 1600
        print(f"[warn] {img_path} missing; assuming {W}x{H}")

    bg = None
    if args.overlay and photo is not None:
        bg = (photo[..., :3].astype(np.float32) * 0.35).astype(np.uint8)

    for kind, label in [("s2d", "accuracy"), ("d2s", "completeness")]:
        ply = args.out_dir / f"vis_{args.scan:03d}_{kind}.ply"
        if not ply.exists():
            print(f"[skip] {ply} not found")
            continue
        xyz, rgb = _read_ply_xyzrgb(ply)
        img, n = _splat(xyz, rgb, P, H, W, args.radius, bg=bg)
        out = args.out_dir / f"chamfer_view{args.view:03d}_{kind}_{label}.png"
        Image.fromarray(img).save(out)
        print(f"[{kind}] {label}: {n:,}/{len(xyz):,} pts in frame -> {out}")


if __name__ == "__main__":
    main()
