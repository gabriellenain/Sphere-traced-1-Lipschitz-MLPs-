#!/bin/bash
#SBATCH --job-name=1lip-skull
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-skull_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-skull_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_skull_${SLURM_JOB_ID}.log
echo "Logging to $LOG"

python -u -m lip_tracer.train \
    --dataset skull \
    --init hull \
    --steps 300000 \
    --batch 4096 \
    --down 1 \
    --wandb 2>&1 | tee "$LOG"
