#!/bin/bash
#SBATCH --job-name=dsine-normals
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dsine_normals_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu001,gpu002,gpu004,gpu011,gpu014

# Usage: sbatch precompute_normals_dsine.slurm <scan_number>
SCAN=${1:-122}
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/precompute_normals_dsine.py \
    --scene "$SCENE" \
    --dsine-dir /scratch/glenain/DSINE
