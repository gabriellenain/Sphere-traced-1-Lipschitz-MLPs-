#!/bin/bash
#SBATCH --job-name=viz-depth-qual-122
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_depth_quality_scan122_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_depth_quality_scan122_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --exclude=gpu004,gpu011,gpu014
#SBATCH --time=0:15:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python -u archive/viz_mast3r_depth_quality.py \
    --scene /scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan122 \
    --depth-dir /scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan122/mast3r_depth_full_probe_64imgs_allpairs \
    --n-views 12
