#!/bin/bash
#SBATCH --job-name=viz-scan122-pts
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_scan122_pts_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:20:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/viz_sfm_pts_skull.py \
    --scene /scratch/_projets_/willow/1-lip-tracer/data/dtu/scan122 \
    --view 50 \
    --min-views 45 \
    --out outputs/scan122_pts_overlay_v50_minv45.png
# outputs:
#   outputs/scan122_pts_overlay_v50_minv2_ref_colmap.png  (reference + COLMAP overlay)
#   outputs/scan122_pts_overlay_v50_minv2_all4.png        (4 panels)
