"""Convert a Kinovis MVMannequin scene to NeuS/DTU layout (cameras.npz + image/ + mask/).

For each scene, this writes data/mvmannequin_neus/<scene>/:
  cameras.npz   world_mat_i = [P_i ; 0 0 0 1],  scale_mat_i = unit-sphere -> world (meters)
  image/NNN.png   symlink to MultiViewPreProcessed/<scene>/ImagesUndistorted/cam-<id>.png
  mask/NNN.png    symlink to MultiViewPreProcessed/<scene>/Masks/cam-<id>.png
  gt_mesh.ply     GT scan in normalized (unit-sphere) coordinates, for Chamfer eval.

The unit sphere is derived from the GT mesh in world coords (mesh_local -> transform_*.txt).
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def parse_calibration(xml_path: Path) -> list[tuple[int, np.ndarray, tuple[int, int]]]:
    """Return list of (cam_id, P[3,4], (W,H)) sorted by cam_id."""
    root = ET.parse(xml_path).getroot()
    cams = []
    for c in root.findall("Camera"):
        cid = int(c.get("id"))
        W = int(c.get("width")); H = int(c.get("height"))
        P = np.array(list(map(float, c.find("P").text.split())), dtype=np.float64).reshape(3, 4)
        cams.append((cid, P, (W, H)))
    cams.sort(key=lambda x: x[0])
    return cams


def scene_scale_from_mesh(ply_path: Path, transform_path: Path,
                          margin: float = 1.05) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (scale_mat 4x4 -- unit-sphere -> world, transform 4x4 mesh->world, radius_used)."""
    m = trimesh.load(ply_path, process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    T = np.loadtxt(transform_path).astype(np.float64).reshape(4, 4)
    vw = (T @ np.hstack([v, np.ones((len(v), 1))]).T).T[:, :3]
    center = 0.5 * (vw.min(0) + vw.max(0))
    r = float(np.linalg.norm(vw - center, axis=1).max()) * margin
    scale_mat = np.eye(4, dtype=np.float64)
    scale_mat[:3, :3] *= r
    scale_mat[:3, 3] = center
    return scale_mat, T, r


def convert_scene(scene: str, src_root: Path, out_root: Path) -> None:
    mv_dir = src_root / "MultiViewPreProcessed" / scene
    xml_path = mv_dir / "calibration_undistorted.xml"
    img_dir = mv_dir / "ImagesUndistorted"
    msk_dir = mv_dir / "Masks"
    eval_dir = mv_dir / "EvalMasks"
    ply_path = src_root / "Scans" / f"_{scene}.ply"
    tf_path = src_root / "Scans" / f"transform_{scene}.txt"

    for p in (xml_path, img_dir, msk_dir, ply_path, tf_path):
        if not p.exists():
            raise FileNotFoundError(p)

    cams = parse_calibration(xml_path)
    scale_mat, T_mesh2world, r = scene_scale_from_mesh(ply_path, tf_path)

    out_dir = out_root / scene
    (out_dir / "image").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_mask").mkdir(parents=True, exist_ok=True)

    npz: dict[str, np.ndarray] = {}
    for idx, (cid, P, (W, H)) in enumerate(cams):
        world_mat = np.eye(4, dtype=np.float64)
        world_mat[:3, :4] = P
        npz[f"world_mat_{idx}"] = world_mat
        npz[f"scale_mat_{idx}"] = scale_mat

        for sub_in, sub_out in [(img_dir, "image"), (msk_dir, "mask"), (eval_dir, "eval_mask")]:
            src = sub_in / f"cam-{cid}.png"
            if not src.exists():
                continue
            dst = out_dir / sub_out / f"{idx:03d}.png"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

    np.savez(out_dir / "cameras.npz", **npz)

    # Save normalized GT mesh for Chamfer eval: unit-sphere = scale_mat^{-1} @ world.
    m = trimesh.load(ply_path, process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    vw = (T_mesh2world @ np.hstack([v, np.ones((len(v), 1))]).T).T[:, :3]
    inv = np.linalg.inv(scale_mat)
    vn = (inv @ np.hstack([vw, np.ones((len(vw), 1))]).T).T[:, :3]
    trimesh.Trimesh(vertices=vn, faces=np.asarray(m.faces), process=False).export(out_dir / "gt_mesh.ply")

    print(f"[{scene}] n_cams={len(cams)}  W,H=({W},{H})  scene_radius={r:.4f} m"
          f"  center={scale_mat[:3, 3]}  -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["kinette-cos-hx", "kino-cos-hx"])
    ap.add_argument("--src", type=Path, default=Path("data/mvmannequin"))
    ap.add_argument("--out", type=Path, default=Path("data/mvmannequin_neus"))
    args = ap.parse_args()
    for s in args.scenes:
        convert_scene(s, args.src.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
