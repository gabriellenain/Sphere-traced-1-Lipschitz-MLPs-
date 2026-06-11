#!/usr/bin/env python3
"""Export the exact visual-hull SDF target used by hull initialization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lip_tracer.data import load_colmap_points
from lip_tracer.visual_hull import carve, occ_to_sdf, save_views


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Training run directory containing config.json.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-views", type=int, default=0,
                        help="Require each surviving voxel to project inside at least this many views.")
    return parser.parse_args()


def _sfm_roi(scene: Path, bound: float) -> tuple[np.ndarray, np.ndarray]:
    """Match fit_hull_init's padded sparse-SFM ROI exactly."""
    sfm_pts = load_colmap_points(scene).numpy()
    sfm_lo = sfm_pts.min(axis=0)
    sfm_hi = sfm_pts.max(axis=0)
    pad = np.maximum(0.15, 0.15 * (sfm_hi - sfm_lo))
    return (np.maximum(sfm_lo - pad, -bound),
            np.minimum(sfm_hi + pad, bound))


def main() -> None:
    args = _parse_args()
    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text())

    scene = Path(config["scene"])
    res = int(config["init"]["hull_res"])
    hull_sfm_roi = bool(config["init"].get("hull_sfm_roi", False))
    is_blender = (scene / "transforms_train.json").exists()
    bound_key = "bound_blender" if is_blender else "bound_dtu"
    bound = float(config["eval"][bound_key])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roi_bounds = _sfm_roi(scene, bound) if hull_sfm_roi else None

    print(f"Exporting hull target from {config_path}")
    print(f"  scene={scene}")
    print(f"  res={res} bound={bound} hull_sfm_roi={hull_sfm_roi}")
    occ = carve(scene=scene, res=res, bound=bound, roi_bounds=roi_bounds,
                min_views=args.min_views)
    if not occ.any():
        raise RuntimeError("Visual-hull carving produced an empty occupancy grid.")
    print(f"  occupied voxels: {occ.sum()} / {occ.size}")

    _, sdf = occ_to_sdf(occ, bound)
    sdf_grid = sdf.reshape(occ.shape)
    voxel_size = 2.0 * bound / max(res - 1, 1)
    verts_zyx, faces, _, _ = marching_cubes(
        sdf_grid, level=0.0, spacing=(voxel_size,) * 3
    )
    verts_xyz = verts_zyx[:, ::-1] - bound
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
    mesh.fix_normals()

    mesh_path = args.out_dir / "hull_target_sdf_norm.ply"
    grid_path = args.out_dir / "hull_target_sdf_grid.npz"
    views_path = args.out_dir / "hull_target_occ_projections.png"
    metadata_path = args.out_dir / "hull_target_metadata.json"
    mesh.export(mesh_path)
    np.savez_compressed(
        grid_path,
        occ=occ,
        sdf=sdf_grid.astype(np.float32),
        bound=np.float32(bound),
        res=np.int32(res),
    )
    save_views(occ, views_path)

    metadata = {
        "source_run_dir": str(args.run_dir.resolve()),
        "source_config": str(config_path.resolve()),
        "scene": str(scene),
        "res": res,
        "bound": bound,
        "hull_sfm_roi": hull_sfm_roi,
        "min_views": args.min_views,
        "roi_lo": roi_bounds[0].tolist() if roi_bounds is not None else None,
        "roi_hi": roi_bounds[1].tolist() if roi_bounds is not None else None,
        "occupied_voxels": int(occ.sum()),
        "total_voxels": int(occ.size),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "coordinate_frame": "normalized training coordinates",
        "target": "zero isosurface of occ_to_sdf(carve(...))",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved -> {mesh_path}")
    print(f"saved -> {grid_path}")
    print(f"saved -> {metadata_path}")
    print(f"  mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


if __name__ == "__main__":
    main()
