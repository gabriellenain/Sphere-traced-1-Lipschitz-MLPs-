#!/usr/bin/env python3
"""Generate MVSFormer++ depth maps for an IDR-style DTU/TnT/Blender scene.

Stages the scene cameras/images into a DTU-eval layout
(images/00000000.jpg, cams/00000000_cam.txt, pair.txt) and invokes
MVSFormer++'s test.py with the pcd filter (pure Python, no fusibile).

Per-view outputs land at <out>/<scan>/{depth_est,confidence}/00000000.{pfm,npy}.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.linalg import rq


MVSF_REPO = Path(os.environ.get(
    "MVSFORMER_REPO", Path.home() / "scratch" / "MVSFormerPlusPlus"))
# dtu_ckpt lives in the package at the repo root; this script sits in tools/, so
# resolve from the repo root (parent.parent), not the script's own dir.
CKPT_DIR  = Path(__file__).resolve().parent.parent / "lip_tracer" / "dtu_ckpt"


def _decompose_idr_camera(world_mat: np.ndarray, scale_mat: np.ndarray):
    """IDR cameras.npz → (K, R, t_norm, cam_center_norm) in normalised world frame.

    world_mat = K @ [R|t_metric] is decomposed in metric coords; the camera
    centre is then converted to the IDR-normalised frame via scale_mat^{-1}.
    """
    P = world_mat[:3, :4].astype(np.float64)
    K, R = rq(P[:, :3])
    s = np.sign(np.diag(K)); s[s == 0] = 1.0
    K = K @ np.diag(s); R = np.diag(s) @ R
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1; R[2, :] *= -1
    K /= K[2, 2]

    t_metric = np.linalg.solve(K, P[:, 3])
    cam_center_metric = -R.T @ t_metric
    scale_mat_inv = np.linalg.inv(scale_mat.astype(np.float64))
    cam_center_norm = (scale_mat_inv @ np.append(cam_center_metric, 1.0))[:3]
    t_norm = -R @ cam_center_norm
    return K, R, t_norm, cam_center_norm


def _write_cam(path: Path, K: np.ndarray, R: np.ndarray, t: np.ndarray,
               depth_min: float, depth_interval: float, num_depth: int) -> None:
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
    lines = ["extrinsic"]
    lines += [" ".join(f"{v:.10f}" for v in row) for row in E]
    lines += ["", "intrinsic"]
    lines += [" ".join(f"{v:.10f}" for v in row) for row in K]
    lines += ["", f"{depth_min:.6f} {depth_interval:.6f} {num_depth}", ""]
    path.write_text("\n".join(lines))


def _build_pair_txt(centers: np.ndarray, n_src: int) -> str:
    V = len(centers)
    out = [str(V)]
    for i in range(V):
        d = np.linalg.norm(centers - centers[i], axis=1)
        d[i] = np.inf
        order = np.argsort(d)[:n_src]
        scores = 1.0 / np.maximum(d[order], 1e-3)
        out.append(str(i))
        toks = []
        for j, sc in zip(order, scores):
            toks += [str(int(j)), f"{sc:.4f}"]
        out.append(f"{n_src} " + " ".join(toks))
    return "\n".join(out) + "\n"


def stage_scene(scene: Path, scan_name: str, staging: Path, n_src: int,
                blender: bool = False) -> None:
    if blender:
        _stage_blender(scene, scan_name, staging, n_src)
    elif (scene / "intrinsics.txt").exists() and (scene / "pose").is_dir():
        _stage_tnt(scene, scan_name, staging, n_src)
    else:
        _stage_idr(scene, scan_name, staging, n_src)


def _stage_idr(scene: Path, scan_name: str, staging: Path, n_src: int) -> None:
    img_dir = scene / "image"
    image_paths = sorted(p for p in img_dir.iterdir()
                         if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                         and not p.name.startswith("._"))
    if not image_paths:
        raise FileNotFoundError(f"no images under {img_dir}")

    W_orig, H_orig = Image.open(image_paths[0]).size
    cam_path = scene / "cameras.npz"
    if not cam_path.exists():
        cam_path = scene / "cameras_sphere.npz"   # NeuS-BlendedMVS archive name
    cams = np.load(cam_path)

    scan_dir = staging / scan_name
    (scan_dir / "images").mkdir(parents=True, exist_ok=True)
    (scan_dir / "cams").mkdir(parents=True, exist_ok=True)

    centers = []
    view_names = []
    for i, ip in enumerate(image_paths):
        K, R, t, c = _decompose_idr_camera(cams[f"world_mat_{i}"], cams[f"scale_mat_{i}"])
        centers.append(c)
        d_c = float(np.linalg.norm(c))
        d_min = max(0.1, d_c - 1.1)
        d_max = d_c + 1.1
        num_depth = 192
        _write_cam(scan_dir / "cams" / f"{i:08d}_cam.txt", K, R, t,
                   d_min, (d_max - d_min) / num_depth, num_depth)
        Image.open(ip).convert("RGB").save(scan_dir / "images" / f"{i:08d}.jpg", quality=95)

    centers = np.stack(centers)
    (scan_dir / "pair.txt").write_text(_build_pair_txt(centers, n_src))
    print(f"[stage] {len(image_paths)} views → {scan_dir}", flush=True)


def _stage_blender(scene: Path, scan_name: str, staging: Path, n_src: int) -> None:
    import json, math
    meta = json.loads((scene / "transforms_train.json").read_text())
    frames = meta["frames"]

    sample_img = scene / (frames[0]["file_path"].lstrip("./") + ".png")
    W_orig, H_orig = Image.open(sample_img).size
    fl = W_orig / (2.0 * math.tan(meta["camera_angle_x"] / 2.0))
    K = np.array([[fl, 0, W_orig / 2.0],
                  [0, fl, H_orig / 2.0],
                  [0,  0,           1.0]], dtype=np.float64)
    # OpenGL→OpenCV: flip y and z axes
    flip_yz = np.diag([1.0, -1.0, -1.0])

    scan_dir = staging / scan_name
    (scan_dir / "images").mkdir(parents=True, exist_ok=True)
    (scan_dir / "cams").mkdir(parents=True, exist_ok=True)

    centers = []
    for i, fr in enumerate(frames):
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)[:3, :4]
        R_gl = c2w[:3, :3]
        cam_center = c2w[:3, 3]
        R = flip_yz @ R_gl.T          # world-to-cam rotation (OpenCV)
        t = -R @ cam_center
        centers.append(cam_center)

        d_c = float(np.linalg.norm(cam_center))
        d_min = max(0.1, d_c - 1.5)
        d_max = d_c + 1.5
        num_depth = 192
        _write_cam(scan_dir / "cams" / f"{i:08d}_cam.txt", K, R, t,
                   d_min, (d_max - d_min) / num_depth, num_depth)

        src = scene / (fr["file_path"].lstrip("./") + ".png")
        Image.open(src).convert("RGB").save(scan_dir / "images" / f"{i:08d}.jpg", quality=95)

    centers = np.stack(centers)
    (scan_dir / "pair.txt").write_text(_build_pair_txt(centers, n_src))
    print(f"[stage blender] {len(frames)} views → {scan_dir}", flush=True)


def _normalise_tnt_center_scale(scene: Path) -> tuple[np.ndarray, float]:
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float64)
    bb_min, bb_max = bbox[:3], bbox[3:6]
    center = 0.5 * (bb_min + bb_max)
    scale = float(np.max(0.5 * (bb_max - bb_min)))
    return center, scale


def _depth_range_from_cube(R: np.ndarray, t: np.ndarray, bound: float,
                           num_depth: int) -> tuple[float, float]:
    corners = np.array([[x, y, z]
                        for x in (-bound, bound)
                        for y in (-bound, bound)
                        for z in (-bound, bound)], dtype=np.float64)
    z = (R @ corners.T + t.reshape(3, 1))[2]
    z = z[z > 0]
    if z.size == 0:
        return 0.1, 4.0 / num_depth
    z_min, z_max = float(z.min()), float(z.max())
    pad = max(0.1, 0.05 * (z_max - z_min))
    d_min = max(0.05, z_min - pad)
    d_max = z_max + pad
    return d_min, (d_max - d_min) / num_depth


def _stage_tnt(scene: Path, scan_name: str, staging: Path, n_src: int) -> None:
    K = np.loadtxt(scene / "intrinsics.txt", dtype=np.float64)[:3, :3]
    center, scale = _normalise_tnt_center_scale(scene)
    pose_paths = sorted((scene / "pose").glob("0_*.txt"))
    if not pose_paths:
        raise FileNotFoundError(f"no train poses under {scene / 'pose'}")

    scan_dir = staging / scan_name
    (scan_dir / "images").mkdir(parents=True, exist_ok=True)
    (scan_dir / "cams").mkdir(parents=True, exist_ok=True)

    centers = []
    view_names = []
    staged = 0
    num_depth = 192
    for pp in pose_paths:
        ip = scene / "rgb" / (pp.stem + ".png")
        if not ip.exists():
            continue
        c2w = np.loadtxt(pp, dtype=np.float64).reshape(4, 4)
        cam_center = (c2w[:3, 3] - center) / scale
        R = c2w[:3, :3].T
        t = -R @ cam_center
        centers.append(cam_center)
        d_min, d_interval = _depth_range_from_cube(R, t, bound=1.5,
                                                   num_depth=num_depth)
        _write_cam(scan_dir / "cams" / f"{staged:08d}_cam.txt", K, R, t,
                   d_min, d_interval, num_depth)
        Image.open(ip).convert("RGB").save(scan_dir / "images" / f"{staged:08d}.jpg",
                                           quality=95)
        view_names.append(pp.stem)
        staged += 1

    if not centers:
        raise FileNotFoundError(f"no matching train RGB images under {scene / 'rgb'}")
    centers = np.stack(centers)
    (scan_dir / "pair.txt").write_text(_build_pair_txt(centers, n_src))
    (scan_dir / "view_names.txt").write_text("\n".join(view_names) + "\n")
    print(f"[stage tnt] {staged} train views → {scan_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path,
                    default=Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65"))
    ap.add_argument("--scan-name", default=None,
                    help="Subdir name in staging dir (defaults to scene basename).")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--staging", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=CKPT_DIR / "model_best.pth")
    ap.add_argument("--config", type=Path, default=CKPT_DIR / "config.json")
    ap.add_argument("--max-h", type=int, default=1152)
    ap.add_argument("--max-w", type=int, default=1536)
    ap.add_argument("--num-view", type=int, default=5)
    ap.add_argument("--n-src", type=int, default=10,
                    help="Source views per ref written into pair.txt.")
    ap.add_argument("--numdepth", type=int, default=192)
    ap.add_argument("--interval-scale", type=float, default=1.06)
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument("--blender", action="store_true",
                    help="parse transforms_train.json instead of cameras.npz")
    args = ap.parse_args()

    scan_name = args.scan_name or args.scene.name
    staging = args.staging or (args.scene / "_mvsf_staging")
    out_dir = args.out_dir or (args.scene / f"mvsformer_depth_{args.max_w}x{args.max_h}")
    staging.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_scene(args.scene, scan_name, staging, args.n_src, blender=args.blender)

    # Patch DINOv2 backbone path in config so the MVSFormer++ loader finds it.
    cfg_local = staging / "config.json"
    cfg = args.config.read_text().replace(
        "./pretrained_models/dinov2_vitb14_pretrain.pth",
        str(MVSF_REPO / "pretrained_models" / "dinov2_vitb14_pretrain.pth"))
    cfg_local.write_text(cfg)

    cmd = [
        # MVSFormer++'s entrypoint is test.py at its repo root (run with cwd=MVSF_REPO);
        # NOT this project's analysis/ — a stray reorg rename had broken this.
        sys.executable, "-u", "test.py",
        "--dataset", "tt",                      # tt path skips DTU GT-depth lookup
        "--batch_size", "1",
        "--testpath_single_scene", str(staging / scan_name),
        "--testlist", "all",
        "--config", str(cfg_local),
        "--resume", str(args.ckpt),
        "--outdir", str(out_dir),
        "--interval_scale", str(args.interval_scale),
        "--num_view", str(args.num_view),
        "--numdepth", str(args.numdepth),
        "--max_h", str(args.max_h),
        "--max_w", str(args.max_w),
        "--filter_method", "pcd",
        "--conf", "0.5",
        "--thres_view", "2",
        "--thres_disp", "1.0",
    ]
    print("[run]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MVSF_REPO}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=MVSF_REPO, env=env, check=True)

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[done] depth_est + confidence under {out_dir}")


if __name__ == "__main__":
    main()
