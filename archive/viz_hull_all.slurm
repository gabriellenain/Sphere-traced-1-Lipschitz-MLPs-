#!/bin/bash
#SBATCH --job-name=1lip-viz-hull-all
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_hull_all_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_hull_all_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --exclude=gpu004,gpu011,gpu014
#SBATCH --time=2:00:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python -u - <<'EOF'
import numpy as np
from pathlib import Path
from skimage.measure import marching_cubes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from lip_tracer.visual_hull import carve, occ_to_sdf
from lip_tracer.config import OUT_DIR

BOUND = 1.5
RES   = 256

DTU_ROOT     = Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu")
BLENDER_ROOT = Path("/scratch/_projets_/willow/1-lip-tracer/data/nerf_synthetic")

SCENES = [
    (DTU_ROOT / "scan65",       "skull",   "dtu"),
    (DTU_ROOT / "scan24",       "scan24",  "dtu"),
    (DTU_ROOT / "scan37",       "scan37",  "dtu"),
    (DTU_ROOT / "scan40",       "scan40",  "dtu"),
    (DTU_ROOT / "scan55",       "scan55",  "dtu"),
    (DTU_ROOT / "scan63",       "scan63",  "dtu"),
    (DTU_ROOT / "scan69",       "scan69",  "dtu"),
    (DTU_ROOT / "scan83",       "scan83",  "dtu"),
    (DTU_ROOT / "scan97",       "scan97",  "dtu"),
    (DTU_ROOT / "scan105",      "scan105", "dtu"),
    (DTU_ROOT / "scan106",      "scan106", "dtu"),
    (DTU_ROOT / "scan110",      "scan110", "dtu"),
    (DTU_ROOT / "scan114",      "scan114", "dtu"),
    (DTU_ROOT / "scan118",      "scan118", "dtu"),
    (DTU_ROOT / "scan122",      "scan122", "dtu"),
    (BLENDER_ROOT / "lego",     "lego",    "blender"),
    (BLENDER_ROOT / "ficus",    "ficus",   "blender"),
    (BLENDER_ROOT / "hotdog",   "hotdog",  "blender"),
]

# filter to scenes that actually exist on disk
SCENES = [(s, n, t) for s, n, t in SCENES if s.exists()]
print(f"Found {len(SCENES)} scenes: {[n for _, n, _ in SCENES]}")

slice_pngs = []

for scene, name, stype in SCENES:
    print(f"\n=== {name} ({stype}) ===")
    try:
        occ = carve(scene=scene, res=RES, bound=BOUND)
        print(f"  occupied: {occ.sum()} / {occ.size}")

        pts, sdf = occ_to_sdf(occ, BOUND)
        sdf_grid = sdf.reshape(RES, RES, RES)

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
        fig.suptitle(f"Visual hull SDF — {name} ({stype})")
        fig.tight_layout()
        out_png = OUT_DIR / f"hull_sdf_slices_{name}.png"
        fig.savefig(out_png, dpi=100)
        plt.close(fig)
        slice_pngs.append((name, out_png))
        print(f"  saved → {out_png}")
    except Exception as e:
        print(f"  FAILED: {e}")

# Assemble all per-scene PNGs into one vertical strip
print("\n=== assembling summary ===")
rows = []
for name, png_path in slice_pngs:
    img = np.array(Image.open(png_path))
    rows.append(img)

if rows:
    # pad widths to match
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], r.shape[2]), 255, dtype=r.dtype)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    summary = np.concatenate(padded, axis=0)
    out = OUT_DIR / "hull_sdf_slices_ALL.png"
    Image.fromarray(summary).save(out)
    print(f"saved → {out}")
EOF
