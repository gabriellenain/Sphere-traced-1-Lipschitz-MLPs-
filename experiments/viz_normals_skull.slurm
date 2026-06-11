#!/bin/bash
#SBATCH --job-name=viz-normals-skull
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_normals_skull_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:10:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/viz_normals_skull.py \
    --views 0,8,16,24,32,40 \
    --out outputs/skull_normals.png
