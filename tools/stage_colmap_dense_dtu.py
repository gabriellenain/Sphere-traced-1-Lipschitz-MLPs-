#!/usr/bin/env python3
"""Stage an IDR-format DTU scene for calibrated COLMAP dense MVS."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.linalg import rq


def _decompose_idr_camera(world_mat: np.ndarray, scale_mat: np.ndarray):
    """Return intrinsics, normalized-frame world-to-camera pose, and center."""
    P = world_mat[:3, :4].astype(np.float64)
    K, R = rq(P[:, :3])
    sign = np.sign(np.diag(K))
    sign[sign == 0] = 1.0
    K = K @ np.diag(sign)
    R = np.diag(sign) @ R
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1.0
        R[2, :] *= -1.0
    K /= K[2, 2]

    t_metric = np.linalg.solve(K, P[:, 3])
    center_metric = -R.T @ t_metric
    center = (np.linalg.inv(scale_mat.astype(np.float64))
              @ np.append(center_metric, 1.0))[:3]
    t = -R @ center
    return K, R, t, center


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to COLMAP's (qw, qx, qy, qz) convention."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s,
                         (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s,
                         (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                         (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s,
                     (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _image_paths(scene: Path) -> list[Path]:
    paths = sorted(p for p in (scene / "image").iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                   and not p.name.startswith("._"))
    if not paths:
        raise FileNotFoundError(f"no images under {scene / 'image'}")
    return paths


def _write_model(scene: Path, model: Path) -> tuple[list[str], np.ndarray]:
    paths = _image_paths(scene)
    cams = np.load(scene / "cameras.npz")
    width, height = Image.open(paths[0]).size
    model.mkdir(parents=True, exist_ok=True)

    camera_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                   "# POINTS2D[] as (X, Y, POINT3D_ID)"]
    centers = []
    for i, image_path in enumerate(paths):
        if Image.open(image_path).size != (width, height):
            raise ValueError(f"image size mismatch: {image_path}")
        K, R, t, center = _decompose_idr_camera(
            cams[f"world_mat_{i}"], cams[f"scale_mat_{i}"])
        q = _rot_to_quat(R)
        image_id = i + 1
        camera_lines.append(
            f"{image_id} PINHOLE {width} {height} "
            f"{K[0, 0]:.12g} {K[1, 1]:.12g} {K[0, 2]:.12g} {K[1, 2]:.12g}")
        image_lines.append(
            f"{image_id} {' '.join(f'{x:.16g}' for x in q)} "
            f"{' '.join(f'{x:.16g}' for x in t)} {image_id} {image_path.name}")
        image_lines.append("")
        centers.append(center)

    (model / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (model / "images.txt").write_text("\n".join(image_lines) + "\n")
    (model / "points3D.txt").write_text("")
    print(f"[model] wrote {model} ({len(paths)} registered images)")
    return [p.name for p in paths], np.stack(centers)


def _write_source_config(path: Path, names: list[str], centers: np.ndarray,
                         n_src: int) -> None:
    n_src = min(n_src, len(names) - 1)
    lines = []
    for i, name in enumerate(names):
        dist = np.linalg.norm(centers - centers[i], axis=1)
        dist[i] = np.inf
        sources = [names[j] for j in np.argsort(dist)[:n_src]]
        lines += [name, ", ".join(sources)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[config] wrote {path} ({n_src} nearest sources per image)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--n-src", type=int, default=10)
    args = parser.parse_args()

    names, centers = _write_model(args.scene, args.model)
    if args.source_config:
        _write_source_config(args.source_config, names, centers, args.n_src)


if __name__ == "__main__":
    main()
