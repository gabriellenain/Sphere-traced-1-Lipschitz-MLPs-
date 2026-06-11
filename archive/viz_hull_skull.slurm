#!/bin/bash
#SBATCH --job-name=1lip-viz-hull
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_hull_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_hull_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --exclude=gpu004,gpu011,gpu014
#SBATCH --time=0:30:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python -u - <<'EOF'
import numpy as np
from pathlib import Path
from skimage.measure import marching_cubes
from lip_tracer.visual_hull import carve, occ_to_sdf, save_views
from lip_tracer.config import SCENE, OUT_DIR

BOUND = 1.5
RES   = 256

print("carving …")
occ = carve(scene=SCENE, res=RES, bound=BOUND)
print(f"  occupied: {occ.sum()} / {occ.size}")

save_views(occ, OUT_DIR / "hull_occ_projections_skull.png")

print("distance transform → SDF …")
pts, sdf = occ_to_sdf(occ, BOUND)
sdf_grid = sdf.reshape(RES, RES, RES)
spacing  = 2 * BOUND / (RES - 1)

print("marching cubes …")
verts, faces, *_ = marching_cubes(sdf_grid, level=0.0, spacing=(spacing,)*3)
verts = verts - BOUND
print(f"  mesh: {len(verts)} verts  {len(faces)} faces")

out_obj = OUT_DIR / "hull_sdf_mesh_skull.obj"
with open(out_obj, "w") as f:
    for v in verts:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for face in faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print(f"saved → {out_obj}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mid = RES // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (sl, title) in zip(axes, [
    (sdf_grid[mid, :, :], "X slice"),
    (sdf_grid[:, mid, :], "Y slice"),
    (sdf_grid[:, :, mid], "Z slice"),
]):
    im = ax.imshow(sl, origin="lower", cmap="RdBu_r",
                   vmin=-0.3, vmax=0.3, extent=[-BOUND, BOUND, -BOUND, BOUND])
    ax.contour(np.linspace(-BOUND, BOUND, RES), np.linspace(-BOUND, BOUND, RES),
               sl, levels=[0.0], colors="k", linewidths=1.5)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)
fig.suptitle("Visual hull SDF skull (target for hull init)")
fig.tight_layout()
out_png = OUT_DIR / "hull_sdf_slices_skull.png"
fig.savefig(out_png, dpi=150)
print(f"saved → {out_png}")
EOF
