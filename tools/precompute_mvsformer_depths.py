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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lip_tracer.data import _find_epfl_strecha_urd, load_views  # noqa: E402


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
    # Canonical 4-token MVSNet depth line: "min interval num max". MVSFormer++'s
    # parser only reads the first two tokens, but ACMMP's ReadCamera requires the
    # 4th (depth_max) — a 3-token line leaves it 0 and degenerates the search.
    depth_max = depth_min + depth_interval * num_depth
    lines += ["", f"{depth_min:.6f} {depth_interval:.6f} {num_depth} {depth_max:.6f}", ""]
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
                blender: bool = False, down: int = 1) -> None:
    if blender:
        _stage_blender(scene, scan_name, staging, n_src)
    elif (scene / "dslr_calibration_undistorted" / "cameras.txt").exists():
        _stage_eth3d(scene, scan_name, staging, n_src, down=down)
    elif (scene / "intrinsics.txt").exists() and (scene / "pose").is_dir():
        _stage_tnt(scene, scan_name, staging, n_src)
    elif _find_epfl_strecha_urd(scene) is not None:
        _stage_epfl_strecha(scene, scan_name, staging, n_src)
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


def _depth_range_from_sfm(R: np.ndarray, t: np.ndarray, pts: np.ndarray,
                          num_depth: int,
                          lo_pct: float = 2.0, hi_pct: float = 98.0,
                          lo_mul: float = 0.8, hi_mul: float = 1.2,
                          min_pts: int = 50) -> tuple[float, float] | None:
    """Per-view depth range bracketing the sparse SfM surface seen by this view.

    Projects the sparse cloud into the camera and takes a padded percentile band
    of the in-front depths. Returns None when too few points project (caller then
    falls back to the origin-cube heuristic). This is robust for off-origin scenes
    where the camera sits inside a fixed origin-centred cube, whose near plane can
    otherwise land *behind* the true surface and clip it out of the hypothesis
    range (see fountain-P11: surface ~0.35 but cube d_min up to 1.84).
    """
    z = (R @ pts.T + t.reshape(3, 1))[2]
    z = z[z > 1e-4]
    if z.size < min_pts:
        return None
    z_lo, z_hi = np.percentile(z, [lo_pct, hi_pct])
    d_min = max(0.05, lo_mul * float(z_lo))
    d_max = max(d_min + 1e-3, hi_mul * float(z_hi))
    return d_min, (d_max - d_min) / num_depth


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


def _stage_epfl_strecha(scene: Path, scan_name: str, staging: Path, n_src: int) -> None:
    urd = _find_epfl_strecha_urd(scene)
    if urd is None:
        raise FileNotFoundError(f"no EPFL *_dense/urd directory found under {scene}")
    image_paths = sorted(p for p in urd.glob("*.png") if not p.name.startswith("._"))
    if not image_paths:
        raise FileNotFoundError(f"no EPFL images under {urd}")

    views = load_views(scene, down=1)
    c2w_all = views["c2w"].numpy().astype(np.float64)
    K_all = views["K"].numpy().astype(np.float64)

    sfm_path = scene / "sparse_sfm_points.txt"
    sfm_pts = (np.loadtxt(sfm_path)[:, :3].astype(np.float64)
               if sfm_path.is_file() else None)

    scan_dir = staging / scan_name
    (scan_dir / "images").mkdir(parents=True, exist_ok=True)
    (scan_dir / "cams").mkdir(parents=True, exist_ok=True)

    centers = []
    num_depth = 192
    for i, ip in enumerate(image_paths):
        c2w = c2w_all[i]
        cam_center = c2w[:3, 3]
        R = c2w[:3, :3].T
        t = -R @ cam_center
        centers.append(cam_center)
        rng = (_depth_range_from_sfm(R, t, sfm_pts, num_depth)
               if sfm_pts is not None else None)
        if rng is None:
            rng = _depth_range_from_cube(R, t, bound=1.5, num_depth=num_depth)
            src = "cube"
        else:
            src = "sfm"
        d_min, d_interval = rng
        print(f"  [stage epfl] view {i:02d}: depth range "
              f"{d_min:.3f}..{d_min + d_interval * num_depth:.3f} ({src})",
              flush=True)
        _write_cam(scan_dir / "cams" / f"{i:08d}_cam.txt", K_all[i], R, t,
                   d_min, d_interval, num_depth)
        Image.open(ip).convert("RGB").save(scan_dir / "images" / f"{i:08d}.jpg",
                                           quality=95)

    centers = np.stack(centers)
    (scan_dir / "pair.txt").write_text(_build_pair_txt(centers, n_src))
    (scan_dir / "view_names.txt").write_text("\n".join(p.name for p in image_paths) + "\n")
    print(f"[stage epfl] {len(image_paths)} views → {scan_dir}", flush=True)


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP quaternion (w,x,y,z) -> world-to-camera rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _read_colmap_points3d(path: Path) -> np.ndarray | None:
    """Parse COLMAP points3D.txt -> (N,3) XYZ. Returns None if absent/empty."""
    if not path.is_file():
        return None
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tok = line.split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            pts.append((float(tok[1]), float(tok[2]), float(tok[3])))
    return np.asarray(pts, dtype=np.float64) if pts else None


def _stage_eth3d(scene: Path, scan_name: str, staging: Path, n_src: int,
                 down: int = 1) -> None:
    """Stage an ETH3D high-res DSLR scene (COLMAP-undistorted) into MVSNet format.

    ETH3D ships a COLMAP text reconstruction under dslr_calibration_undistorted/
    (PINHOLE cameras, world-to-cam q/t) plus undistorted JPGs. Unlike DTU/TnT we
    keep the native *metric* COLMAP frame — that is the frame the ETH3D laser-scan
    GT lives in, so ACMMP's emitted cam-z depths fuse straight into world-metric
    points for the official multi-view-evaluation. Per-view depth ranges come from
    the sparse SfM cloud (origin-cube fallback is meaningless off-origin here).

    `down` integer-downsamples images and intrinsics (ETH3D DSLR is ~24 MP; ACMMP
    at full res is impractical, so the slurm job passes down=2).
    """
    calib = scene / "dslr_calibration_undistorted"
    img_root = scene / "images"   # NAME in images.txt is dslr_images_undistorted/<f>.JPG

    # cameras.txt: CAMERA_ID MODEL W H fx fy cx cy  (PINHOLE)
    cams: dict[int, np.ndarray] = {}
    for line in (calib / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cid, model = int(t[0]), t[1]
        if model != "PINHOLE":
            raise ValueError(f"{scene.name}: expected PINHOLE camera, got {model}")
        fx, fy, cx, cy = map(float, t[4:8])
        cams[cid] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)

    # images.txt: two lines per image; first has pose, second the 2D points.
    raw = (calib / "images.txt").read_text().splitlines()
    entries = []  # (name, cam_id, R, t)
    expect_pose = True
    for line in raw:
        if line.startswith("#") or not line.strip():
            continue
        if expect_pose:
            t = line.split()
            qw, qx, qy, qz = map(float, t[1:5])
            tx, ty, tz = map(float, t[5:8])
            cam_id, name = int(t[8]), t[9]
            R = _quat_to_rot(qw, qx, qy, qz)
            entries.append((name, cam_id, R, np.array([tx, ty, tz], dtype=np.float64)))
            expect_pose = False
        else:
            expect_pose = True  # skip the POINTS2D line
    entries.sort(key=lambda e: e[0])  # deterministic by image name

    sfm_pts = _read_colmap_points3d(calib / "points3D.txt")

    scan_dir = staging / scan_name
    (scan_dir / "images").mkdir(parents=True, exist_ok=True)
    (scan_dir / "cams").mkdir(parents=True, exist_ok=True)

    centers, view_names = [], []
    num_depth = 192
    for i, (name, cam_id, R, t) in enumerate(entries):
        ip = img_root / name
        if not ip.exists():
            print(f"  [stage eth3d] WARN missing image {ip}", flush=True)
            continue
        K = cams[cam_id].copy()
        if down > 1:
            K[0, :] /= down
            K[1, :] /= down
        cam_center = -R.T @ t
        centers.append(cam_center)

        rng = (_depth_range_from_sfm(R, t, sfm_pts, num_depth)
               if sfm_pts is not None else None)
        if rng is None:
            rng = _depth_range_from_cube(R, t, bound=1.5, num_depth=num_depth)
        d_min, d_interval = rng
        _write_cam(scan_dir / "cams" / f"{i:08d}_cam.txt", K, R, t,
                   d_min, d_interval, num_depth)

        im = Image.open(ip).convert("RGB")
        if down > 1:
            im = im.resize((im.width // down, im.height // down), Image.BILINEAR)
        im.save(scan_dir / "images" / f"{i:08d}.jpg", quality=95)
        view_names.append(name)

    if not centers:
        raise FileNotFoundError(f"no ETH3D images staged for {scene}")
    centers = np.stack(centers)
    (scan_dir / "pair.txt").write_text(_build_pair_txt(centers, n_src))
    (scan_dir / "view_names.txt").write_text("\n".join(view_names) + "\n")
    print(f"[stage eth3d] {len(view_names)} views (down={down}) → {scan_dir}", flush=True)


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
    ap.add_argument("--down", type=int, default=1,
                    help="integer image/intrinsic downsample factor (ETH3D staging "
                         "only; DSLR frames are ~24 MP so ACMMP uses down=2).")
    ap.add_argument("--stage-only", action="store_true",
                    help="only write the MVSNet-format staging (images/cams/pair.txt) "
                         "and exit, without running MVSFormer++. Lets non-learning MVS "
                         "(e.g. ACMMP) reuse the exact same camera staging.")
    args = ap.parse_args()

    scan_name = args.scan_name or args.scene.name
    staging = args.staging or (args.scene / "_mvsf_staging")
    out_dir = args.out_dir or (args.scene / f"mvsformer_depth_{args.max_w}x{args.max_h}")
    staging.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_scene(args.scene, scan_name, staging, args.n_src, blender=args.blender,
                down=args.down)

    if args.stage_only:
        print(f"[stage-only] MVSNet input staged at {staging / scan_name}", flush=True)
        return

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
