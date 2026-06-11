"""Visualise the *exact* SDF target that fit_colmap_init regresses against.

That target is NOT the smooth Poisson SDF — it's the distance-transform-derived
SDF from the binary 256^3 occupancy grid (see visual_hull.occ_to_sdf). This
script replicates the pipeline:

    poisson.ply  ->  normalise by bbox.txt  ->  raycasting -> sd  ->
    occ = (sd < 0)  ->  occ_to_sdf  ->  voxel-quantised SDF on the grid

and renders orthogonal slices of the result.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--mesh",   required=True, help="poisson.ply (un-normalised, NSVF-COLMAP frame)")
ap.add_argument("--bbox",   required=True, help="data/tnt/Barn/bbox.txt")
ap.add_argument("--out",    required=True)
ap.add_argument("--res",    type=int,   default=256, help="must match init_cfg.hull_res")
ap.add_argument("--bound",  type=float, default=1.5)
args = ap.parse_args()

# 1. Load mesh + normalise into unit cube (same as fit_colmap_init)
mesh = o3d.io.read_triangle_mesh(args.mesh)
bbox = np.loadtxt(args.bbox, dtype=np.float32)
center = 0.5 * (bbox[:3] + bbox[3:6])
bb_scale = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
verts = (np.asarray(mesh.vertices, dtype=np.float64) - center) / bb_scale
mesh.vertices = o3d.utility.Vector3dVector(verts)
print(f"[norm] center={center}  scale={bb_scale:.4f}", flush=True)

# 2. Carve occupancy via raycasting (same as fit_colmap_init)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
res, bound = args.res, args.bound
grid = np.linspace(-bound, bound, res, dtype=np.float32)
zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
sd_true = scene.compute_signed_distance(o3d.core.Tensor(pts)).numpy().reshape(res, res, res)
occ = (sd_true < 0)
print(f"[carve] mesh-only occupied {occ.sum():,}/{occ.size:,} "
      f"({100*occ.mean():.2f}%)", flush=True)

# Union with sibling fused.ply point cloud (matches fit_colmap_init exactly).
fused_path = Path(args.mesh).parent / "fused.ply"
if fused_path.exists():
    from scipy.spatial import cKDTree
    pcd = o3d.io.read_point_cloud(str(fused_path))
    pts_n = (np.asarray(pcd.points, dtype=np.float64) - center) / bb_scale
    voxel_size_n = 2.0 * bound / max(res - 1, 1)
    tree = cKDTree(pts_n)
    d, _ = tree.query(pts.astype(np.float64), k=1,
                      distance_upper_bound=voxel_size_n * 1.5)
    occ_pts = (d < voxel_size_n * 1.0).reshape(res, res, res)
    occ = occ | occ_pts
    print(f"[carve] +fused.ply ({len(pts_n):,} pts) -> "
          f"{occ.sum():,}/{occ.size:,} ({100*occ.mean():.2f}%)", flush=True)

# 3. occ_to_sdf — the actual regression target
from scipy.ndimage import distance_transform_edt
voxel_size = 2 * bound / max(res - 1, 1)
sd_target = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * voxel_size
print(f"[occ_to_sdf] voxel_size={voxel_size:.5f}  "
      f"target range = [{sd_target.min():.4f}, {sd_target.max():.4f}]", flush=True)

# 4. Plot four slices side-by-side: target SDF and (for comparison) true SDF
def axis_pair(axis):
    a, b = [i for i in range(3) if i != axis]
    return a, b, "xyz"[a] + "xyz"[b]

y_lo, y_hi = bbox[1], bbox[4]
mid_y = int(0.5 * res)                       # mid height
high_y = int(0.30 * res)                     # near roof (Y-down)
mid_z = int(0.5 * res)

panels = [
    ("target SDF — Y slice high (near roof)", sd_target, 1, high_y),
    ("target SDF — Y slice mid",              sd_target, 1, mid_y),
    ("target SDF — Z slice through centre",   sd_target, 2, mid_z),
    ("true Poisson SDF — Z slice (compare)",  sd_true,   2, mid_z),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 14))
vmax = 0.3  # ~ 30% of bound; covers most of the gradient field

for ax, (title, vol, axis, idx) in zip(axes.flat, panels):
    a, b, lbl = axis_pair(axis)
    if axis == 0:
        sl = vol[idx, :, :]
    elif axis == 1:
        sl = vol[:, idx, :]
    else:
        sl = vol[:, :, idx]
    # arrange so the in-plane axes match the slice's natural orientation
    # axes are (i, j, k) <-> (z, y, x). When axis=1 (slicing y), sl is (z, x).
    if axis == 1:
        x_axis = np.linspace(-bound, bound, res); y_axis = np.linspace(-bound, bound, res)
        im = ax.pcolormesh(x_axis, y_axis, sl, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("z")
    elif axis == 2:
        x_axis = np.linspace(-bound, bound, res); y_axis = np.linspace(-bound, bound, res)
        im = ax.pcolormesh(x_axis, y_axis, sl, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.invert_yaxis()
    ax.contour(x_axis, y_axis, sl, levels=[0.0], colors="k", linewidths=0.7)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04,
                 label="SDF (red=inside, blue=outside)")

fig.suptitle(f"fit_colmap_init regression target  (res={res}, bound={bound}, voxel={voxel_size:.4f})",
             fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=120, bbox_inches="tight")
print(f"[write] {args.out}", flush=True)
