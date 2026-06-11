#!/bin/bash
#SBATCH --job-name=1lip-dtu
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-dtu_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-dtu_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# Replicates run_20260601_143409_scan24_4933894's command + two changes:
#  - good init:  --hull-border-aware (border-aware carving + largest-component, w_sfm_free=0)
#  - Gipuma NCC: fixed large patch (11, half-pix 4) with bilateral adaptive-support
#                weights, γ annealed 1.0 → 0.05 (effective support large→small, coarse→fine)

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan24_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
LOG=$RUN_DIR/train.log
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
echo "Run dir: $RUN_DIR  (border-aware hull init)"

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --run-dir "$RUN_DIR" \
    --init hull \
    --steps 300000 \
    --batch 4096 \
    --trace-iters 64 \
    --down 1 \
    --dtu-eval-dir "$DTU_EVAL" \
    --dtu-official-freq 10000 \
    --dtu-official-res 384 \
    --sfm-min-views 30 \
    --feature-maps outputs/scan24_mast3r_features_idr.pt \
    --hull-sfm-roi \
    --hull-border-aware \
    --ncc-color gray \
    --ncc-patch 11 \
    --ncc-half-pix 4 \
    --ncc-bilateral-gamma 1.0 \
    --ncc-bilateral-gamma-end 0.05 \
    --wandb 2>&1 | tee "$LOG"
