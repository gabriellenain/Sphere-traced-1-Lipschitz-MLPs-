#!/bin/bash
#SBATCH --job-name=viz-carving-skull
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_carving_skull_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu_homogen
#SBATCH --time=01:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/viz_carving_skull.py \
    --res 96 \
    --conf-thresh 0.7 \
    --out-thresh 0.7 \
    --out outputs/skull_carving.png
