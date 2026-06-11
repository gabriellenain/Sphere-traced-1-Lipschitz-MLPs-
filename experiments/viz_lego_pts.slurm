#!/bin/bash
#SBATCH --job-name=viz-lego-pts
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_lego_pts_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:10:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/viz_colmap_pts_lego.py \
    --scene /scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego \
    --view 20 \
    --out outputs/lego_pts_overlay_v20.png
