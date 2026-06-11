"""Export visual-hull surfaces (baseline / Fix1 / Fix1+SFM) as .ply for Blender.

Marching-cubes the zero level set of the hull's distance-transform SDF for each
carving variant on DTU scan24, so the border-bloat difference is directly visible.
"""
import numpy as np
import trimesh
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

from lip_tracer.visual_hull import carve, save_views
from lip_tracer.data import load_colmap_points

SCENE = Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan24")
OUT   = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_fix")
RES, BOUND = 128, 1.5
OUT.mkdir(parents=True, exist_ok=True)


def occ_to_sdf_grid(occ, bound):
    res = occ.shape[0]
    vox = 2 * bound / max(res - 1, 1)
    d_in  = distance_transform_edt(occ)
    d_out = distance_transform_edt(~occ)
    return (d_out - d_in) * vox, vox


def export(occ, name):
    sdf, vox = occ_to_sdf_grid(occ, BOUND)
    if not (sdf.min() < 0 < sdf.max()):
        print(f"  [{name}] empty/full occupancy — skipping mesh")
        return
    # Match tools/export_hull_target.py exactly: marching cubes in (z,y,x) index
    # space, then verts_xyz = verts_zyx[:, ::-1] - bound.
    verts_zyx, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(vox,) * 3)
    verts = verts_zyx[:, ::-1] - BOUND
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()
    p = OUT / f"hull_scan24_{name}.ply"
    mesh.export(p)
    save_views(occ, OUT / f"hull_scan24_{name}_proj.png")
    print(f"  [{name}] occ={occ.sum():>7d}  verts={len(verts):>6d}  → {p}")


sfm = load_colmap_points(SCENE).numpy()
print(f"SFM points: {len(sfm)}  bbox_min={sfm.min(0).round(2)}  bbox_max={sfm.max(0).round(2)}")

print("\n=== baseline (current carving) ===")
export(carve(SCENE, RES, BOUND), "baseline")

print("\n=== Fix 1 (border-aware) ===")
export(carve(SCENE, RES, BOUND, border_aware=True), "fix1_border")

print("\n=== Fix 1 + SFM gate ===")
export(carve(SCENE, RES, BOUND, border_aware=True,
             sfm_gate=True, sfm_pts=sfm, sfm_conf_views=3, sfm_margin_voxels=3.0),
       "fix1_sfm")

print(f"\nDone → {OUT}")
