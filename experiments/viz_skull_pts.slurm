#!/bin/bash
#SBATCH --job-name=viz-skull-pts
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_skull_pts_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:20:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/viz_sfm_pts_skull.py \
    --view 16 \
    --min-views 45 \
    --out outputs/skull_pts_overlay_v16_minv45.png
