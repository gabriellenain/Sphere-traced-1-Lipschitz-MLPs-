#!/bin/bash
#SBATCH --job-name=viz-mvsf-skull
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_mvsformer_skull_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:15:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/viz_mvsformer_skull.py \
    --views 0,8,16,24,32,40 \
    --conf-thresh 0.5 \
    --out outputs/skull_mvsformer.png
