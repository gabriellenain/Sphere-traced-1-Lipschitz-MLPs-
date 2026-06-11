#!/bin/bash
#SBATCH --job-name=1lip-hotdog
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-hotdog_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-hotdog_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=24:00:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=/scratch/_projets_/willow/1-lip-tracer/data/nerf_synthetic/hotdog
LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_hotdog_${SLURM_JOB_ID}.log
echo "Logging to $LOG"
echo "Scene: $SCENE"

python -u -m lip_tracer.train \
    --blender \
    --scene "$SCENE" \
    --steps 20000 \
    --batch 2048 \
    --down 1 \
    --hidden 256 \
    --depth 12 \
    --init hull \
    --wandb 2>&1 | tee "$LOG"
