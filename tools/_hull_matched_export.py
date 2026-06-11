"""Export the matched scan24 hull (res=256, SFM-ROI, border-aware) as a clean PLY."""
import numpy as np, trimesh
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
from lip_tracer.visual_hull import carve
from lip_tracer.data import load_colmap_points

SCENE = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24")
OUT = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_fix")
RES, BOUND = 256, 1.5
sfm = load_colmap_points(SCENE).numpy()
lo, hi = sfm.min(0), sfm.max(0); pad = np.maximum(0.15, 0.15 * (hi - lo))
roi = (np.maximum(lo - pad, -BOUND), np.minimum(hi + pad, BOUND))
occ = carve(SCENE, RES, BOUND, roi_bounds=roi, border_aware=True)
vox = 2 * BOUND / (RES - 1)
sdf = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * vox
v_zyx, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(vox,) * 3)   # full res
mesh = trimesh.Trimesh(vertices=v_zyx[:, ::-1] - BOUND, faces=faces, process=False)
mesh.fix_normals()
p = OUT / "hull_scan24_matched_res256_roi_border.ply"
mesh.export(p)
print(f"occ={occ.sum()}  verts={len(mesh.vertices)} faces={len(mesh.faces)}  → {p}")
