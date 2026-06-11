#!/bin/bash
#SBATCH --job-name=1lip-lego-new
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-lego-new_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-lego-new_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LEGO_NEW=/scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_lego_new_${SLURM_JOB_ID}.log
echo "Logging to $LOG"
echo "Using new venv + scene: $LEGO_NEW"

RESUME_ARGS=()
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    RESUME_ARGS=(--resume "$RESUME")
    echo "Resuming from $RESUME"
fi

python -u -m lip_tracer.train \
    --dataset lego \
    --scene "$LEGO_NEW" \
    --steps 200000 \
    --batch 4096 \
    --down 1 \
    --wandb 2>&1 | tee "$LOG"
