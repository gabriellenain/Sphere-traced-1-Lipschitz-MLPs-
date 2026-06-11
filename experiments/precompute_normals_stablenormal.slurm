#!/bin/bash
#SBATCH --job-name=stablenormal
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/stablenormal_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu001,gpu002,gpu004,gpu011,gpu014

# Usage: sbatch precompute_normals_stablenormal.slurm <scan_number>
SCAN=${1:-122}
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export HF_HUB_CACHE=/scratch/glenain/hf_cache

python tools/precompute_normals_stablenormal.py --scene "$SCENE"
