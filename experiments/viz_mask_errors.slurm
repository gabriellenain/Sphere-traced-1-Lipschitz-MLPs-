#!/bin/bash
#SBATCH --job-name=mask-err-viz
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mask_err_viz_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mask_err_viz_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python tools/viz_mask_errors.py \
    --pt outputs/run_20260510_094943_scan65/checkpoint_final.pt \
    --views 16 0 24 32 40 \
    --down 2 \
    --batch 8192 \
    --out outputs/mask_error_viz
