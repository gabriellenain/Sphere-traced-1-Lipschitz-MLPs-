#!/bin/bash
#SBATCH --job-name=1lip-bmvs-clock
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-bmvs-clock_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-bmvs-clock_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# First BlendedMVS scene (NeuS-preprocessed). bmvs_clock: 143 views, 576x768,
# cameras r~3.0-3.8, object normalised to the unit sphere -> DTU trace params
# (t_far=5, bound=1.5, eps=1e-3) transfer directly.
#
# Stripped of DTU-only bits: no --dtu-eval-dir, no --feature-maps, no
# --hull-sfm-roi (bmvs ships no COLMAP sparse_sfm_points.txt). Hull init carves
# from masks + cameras only; the SFM/geo-sdf losses self-disable when absent.

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/bmvs/bmvs_clock
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_bmvs_clock_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
LOG=$RUN_DIR/train.log
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
echo "Run dir: $RUN_DIR  (bmvs_clock, mask-carve hull init)"

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --run-dir "$RUN_DIR" \
    --init hull \
    --hull-border-aware \
    --steps 300000 \
    --batch 4096 \
    --trace-iters 64 \
    --t-far 5.0 \
    --down 1 \
    --mc-res 512 \
    --ncc-color gray \
    --ncc-patch 11 \
    --ncc-half-pix 4 \
    --ncc-bilateral-gamma 1.0 \
    --ncc-bilateral-gamma-end 0.05 \
    --wandb 2>&1 | tee "$LOG"
