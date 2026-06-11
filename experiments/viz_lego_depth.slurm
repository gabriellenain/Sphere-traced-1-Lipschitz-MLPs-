#!/bin/bash
#SBATCH --job-name=viz-lego-depth
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_lego_depth_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:10:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/viz_depth.py \
    --scene /scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego \
    --blender \
    --mvsformer-dir /scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego/mvsformer_depth_800x800_800x800 \
    --view 20 \
    --out outputs/viz_depth_lego_v0.png
