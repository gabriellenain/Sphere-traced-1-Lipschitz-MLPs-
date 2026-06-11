#!/bin/bash
#SBATCH --job-name=1lip-ncc-dbg
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-ncc-dbg_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-ncc-dbg_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# NCC-ONLY baseline + thorough per-view hole diagnostics (views 16,32) for CVPR.
# Usage: sbatch train_dtu_ncc_debug.slurm <scan_number>   (default 65)
SCAN=${1:-65}
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS\ Data

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_nccdbg_dtu${SCAN}_${SLURM_JOB_ID}.log
echo "Logging to $LOG  (scene: $SCENE)  NCC-only + debug-views 16,32"
if [[ ! -d "$SCENE" ]]; then
    echo "ERROR: missing DTU scene: $SCENE"; exit 1
fi
if [[ ! -d "$DTU_EVAL" ]]; then
    echo "ERROR: missing DTU eval: $DTU_EVAL"; exit 1
fi

# Strictly NCC-only: every other geometry/photometric term explicitly zeroed
# so the holes are unambiguously attributable to multi-view NCC alone.
python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --init hull \
    --steps 300000 \
    --down 1 \
    --dtu-eval-dir "$DTU_EVAL" \
    --w-ncc 1.0 \
    --w-photo 0.0 \
    --w-idr-mask 0.0 \
    --w-sil 0.0 \
    --w-surf 0.0 \
    --w-eikonal 0.0 \
    --bsphere-radius 0.0 \
    --debug-views 16,32 \
    --debug-every 1000 \
    --debug-zoom-json debug_zoom_scan65.json \
    --wandb 2>&1 | tee "$LOG"
