#!/bin/bash
#SBATCH --job-name=spectral-sdf
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/spectral_sdf_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/spectral_sdf_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python -u scripts/spectral_sdf_diagnostic.py \
    --run-dir outputs/buddha_sweep/none_W256_D16_10M \
    --mesh data/gt_meshes/happy_buddha_norm.ply \
    --grid-res 128 \
    --tau 0.03 \
    --n-bands 24
