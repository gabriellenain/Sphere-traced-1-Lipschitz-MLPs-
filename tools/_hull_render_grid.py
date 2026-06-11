"""Render the (good) visual-hull SDF target from several viewpoints into a grid.

Carves scan24 exactly as the launched init will (res=256, SFM-ROI, border-aware),
meshes the zero level set of occ_to_sdf, and renders Lambertian-shaded views from
a turntable of azimuths so the 3D shape is visible.
"""
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

from lip_tracer.visual_hull import carve
from lip_tracer.data import load_colmap_points

SCENE = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24")
OUT   = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_fix")
RES, BOUND = 256, 1.5
OUT.mkdir(parents=True, exist_ok=True)

# SFM ROI exactly like fit_hull_init
sfm = load_colmap_points(SCENE).numpy()
lo, hi = sfm.min(0), sfm.max(0)
pad = np.maximum(0.15, 0.15 * (hi - lo))
roi = (np.maximum(lo - pad, -BOUND), np.minimum(hi + pad, BOUND))

occ = carve(SCENE, RES, BOUND, roi_bounds=roi, border_aware=True)
print(f"occ={occ.sum()} / {occ.size}")

vox = 2 * BOUND / (RES - 1)
d_in = distance_transform_edt(occ); d_out = distance_transform_edt(~occ)
sdf = (d_out - d_in) * vox
verts_zyx, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(vox,) * 3, step_size=2)
verts = verts_zyx[:, ::-1] - BOUND
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.fix_normals()
print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

V = mesh.vertices
F = mesh.faces
tris = V[F]                                   # (nf, 3, 3)
fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
light = np.array([0.4, 0.5, 0.8]); light /= np.linalg.norm(light)
ctr = V.mean(0); rad = np.linalg.norm(V - ctr, axis=1).max()

azims = [0, 60, 120, 180, 240, 300]
elev = 18
fig = plt.figure(figsize=(15, 10))
for i, az in enumerate(azims):
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    shade = np.clip(np.abs(fn @ light), 0.15, 1.0)       # two-sided Lambertian
    colors = np.stack([0.2 + 0.7 * shade] * 3 + [np.ones_like(shade)], axis=1)
    pc = Poly3DCollection(tris, facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    ax.view_init(elev=elev, azim=az)
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(f"azim {az}°", fontsize=10)

fig.suptitle("scan24 visual-hull SDF target  (res=256, SFM-ROI, border-aware)", fontsize=13)
fig.tight_layout()
p = OUT / "hull_scan24_render_grid.png"
fig.savefig(p, dpi=110, bbox_inches="tight")
print(f"saved → {p}")
