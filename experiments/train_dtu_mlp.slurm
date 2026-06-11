#!/bin/bash
#SBATCH --job-name=1lip-dtu-mlp
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-dtu-mlp_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-dtu-mlp_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

# Usage: sbatch train_dtu_mlp.slurm <scan_number>
#   e.g. sbatch train_dtu_mlp.slurm 122
SCAN=${1:-122}
SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan${SCAN}
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_dtu${SCAN}_mlp_${SLURM_JOB_ID}.log
echo "Logging to $LOG  (scene: $SCENE)"

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --architecture mlp \
    --hidden 256 \
    --depth 8 \
    --input-encoding neus \
    --multires 6 \
    --trace-iters 24 \
    --init hull \
    --steps 100000 \
    --batch 4096 \
    --down 1 \
    --dtu-eval-dir "$DTU_EVAL" \
    --wandb 2>&1 | tee "$LOG"
