#!/usr/bin/env python3
"""Convert a COLMAP reconstruction (TXT model) of *raw* images into an
NSVF-format scene dir that lip_tracer's `_load_tnt_views` reads unchanged.

This is the camera path used by Neuralangelo / Geo-NeuS on Tanks & Temples:
run our own COLMAP SfM on the unmasked frames, then express the result in the
`rgb/ + pose/ + intrinsics.txt + bbox.txt` layout the training loader expects.

Expected input (produced by colmap_sfm_raw.slurm, after image_undistorter so
the camera is a distortion-free PINHOLE):

    <colmap>/sparse_txt/cameras.txt    one PINHOLE camera (single_camera=1)
    <colmap>/sparse_txt/images.txt     registered poses (world->cam)
    <colmap>/sparse_txt/points3D.txt   sparse cloud (used only for the bbox)
    <colmap>/images/<name>             the undistorted frames

Output (a fresh scene, leaving the NSVF one untouched for comparison):

    <out>/intrinsics.txt   4x4 K (shared)
    <out>/bbox.txt         xmin ymin zmin xmax ymax zmax voxel_size
    <out>/pose/0_XXXX.txt  4x4 camera-to-world
    <out>/rgb/0_XXXX.png   the matching frame (symlink unless --copy)

The COLMAP world gauge is arbitrary -- that's fine: the loader recenters and
rescales by bbox.txt into the unit cube. The one knob that actually needs your
eye is the bbox: a raw SfM cloud spans the whole *scene*, not just the object,
so we percentile-clip the points. Inspect the printed extent and tighten
--clip / edit bbox.txt by hand if the object is a small part of the cloud.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP (qw,qx,qy,qz) world->cam rotation -> 3x3 matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _read_cameras(path: Path) -> dict:
    """camera_id -> (model, fx, fy, cx, cy). Requires PINHOLE."""
    cams = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        t = line.split()
        cid, model = int(t[0]), t[1]
        params = list(map(float, t[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params
            fx = fy = f
        else:
            raise ValueError(
                f"camera model {model!r} still has distortion -- run "
                f"image_undistorter first (colmap_sfm_raw.slurm does this).")
        cams[cid] = (model, fx, fy, cx, cy)
    return cams


def _read_images(path: Path) -> list:
    """-> list of (name, camera_id, c2w 4x4), sorted by name. world->cam in
    images.txt, inverted here to camera-to-world."""
    lines = [ln for ln in path.read_text().splitlines()
             if ln and not ln.startswith("#")]
    out = []
    for i in range(0, len(lines), 2):          # entries are (pose, points2d) pairs
        t = lines[i].split()
        qw, qx, qy, qz = map(float, t[1:5])
        tx, ty, tz = map(float, t[5:8])
        cam_id, name = int(t[8]), t[9]
        R = _quat_to_rot(qw, qx, qy, qz)        # world->cam
        tvec = np.array([tx, ty, tz])
        c2w = np.eye(4)
        c2w[:3, :3] = R.T                       # cam->world
        c2w[:3, 3] = -R.T @ tvec                # camera centre in world
        out.append((name, cam_id, c2w))
    out.sort(key=lambda e: e[0])
    return out


def _read_points_xyz(path: Path) -> np.ndarray:
    xyz = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        t = line.split()
        xyz.append((float(t[1]), float(t[2]), float(t[3])))
    return np.asarray(xyz, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", type=Path, required=True,
                    help="COLMAP workspace with sparse_txt/ and images/")
    ap.add_argument("--out", type=Path, required=True, help="output scene dir")
    ap.add_argument("--clip", type=float, default=2.0,
                    help="percentile clipped off each end per axis for bbox "
                         "(0 = full cloud). Raise if the scene dwarfs the object.")
    ap.add_argument("--copy", action="store_true",
                    help="copy frames instead of symlinking into rgb/")
    args = ap.parse_args()

    txt = args.colmap / "sparse_txt"
    img_src = args.colmap / "images"
    for p in (txt / "cameras.txt", txt / "images.txt", txt / "points3D.txt", img_src):
        if not p.exists():
            raise FileNotFoundError(p)

    cams = _read_cameras(txt / "cameras.txt")
    if len(cams) != 1:
        print(f"  [warn] {len(cams)} cameras found; loader uses ONE shared K. "
              f"Re-run SfM with --ImageReader.single_camera 1 for clean results.")
    _, fx, fy, cx, cy = next(iter(cams.values()))
    K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                 dtype=np.float64)

    images = _read_images(txt / "images.txt")
    pts = _read_points_xyz(txt / "points3D.txt")

    # ---- bbox from the (clipped) sparse cloud --------------------------------
    lo = np.percentile(pts, args.clip, axis=0)
    hi = np.percentile(pts, 100 - args.clip, axis=0)
    extent = hi - lo
    voxel = float(extent.max()) / 128.0

    # ---- write scene ---------------------------------------------------------
    (args.out / "pose").mkdir(parents=True, exist_ok=True)
    (args.out / "rgb").mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out / "intrinsics.txt", K, fmt="%.10f")
    with open(args.out / "bbox.txt", "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in (*lo, *hi, voxel)) + "\n")

    import re
    n_written = 0
    for idx, (name, _cid, c2w) in enumerate(images):
        src = img_src / name
        if not src.exists():
            print(f"  [skip] {name}: not in {img_src}")
            continue
        # Preserve the original frame index in the filename
        # (`0_<seq>_<frame>`): analysis/eval_tnt_official.py pairs poses to the official
        # COLMAP-SfM log via `stem.split("_")[2]`, and the loader is agnostic.
        m = re.search(r"(\d+)", Path(name).stem)
        frame = m.group(1) if m else f"{idx:06d}"
        stem = f"0_{idx:04d}_{frame}"
        np.savetxt(args.out / "pose" / f"{stem}.txt", c2w, fmt="%.10f")
        dst = args.out / "rgb" / f"{stem}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy(src, dst)
        else:
            dst.symlink_to(src.resolve())
        n_written += 1

    print(f"\n  wrote {n_written} views -> {args.out}")
    print(f"  K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"  sparse cloud: {len(pts)} pts")
    print(f"  bbox (clip={args.clip}%): min={lo.round(3).tolist()} "
          f"max={hi.round(3).tolist()} extent={extent.round(3).tolist()}")
    print(f"  -> after unit-cube fit the object should fill most of the cube; "
          f"if it's tiny, the scene cloud is dominating -- raise --clip or edit bbox.txt.")


if __name__ == "__main__":
    main()
