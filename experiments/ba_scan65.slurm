#!/bin/bash
#SBATCH --job-name=ba_scan65
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/ba_scan65_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/ba_scan65_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Photometric bundle adjustment (post-training pass) on the converged scan65 run.
# Refines per-camera extrinsics φ jointly with the SDF θ by block-coordinate
# descent, tracing the f_θ=0 surface with the live camera so pose gradients flow
# (idr_intersection). Starts from the run's final θ checkpoint and its EXACT
# config (model/trace/loss); only sdf_min_beta is forced 0 inside trace_idr (it
# feeds only the silhouette loss, which is off — geometry/E unchanged).
#
# JOINT (θ + φ) on the BARE photometric objective E(θ, φ): each cycle runs a
# φ-block (extrinsics, θ frozen) then a θ-block (SDF, φ frozen). No SDF
# regularisation — w_eikonal=0 (the run's setting), so the θ-block minimises
# pure E exactly like the φ-block. Camera 0 is gauge-locked (default);
# intrinsics fixed (DTU calibration trusted).
#
# Logs: stdout (this file) + per-step CSV (ba_metrics.csv) + wandb (1lip-tracer-ba).
RUN=outputs/run_20260525_223035_scan65_4911692

python -u -m lip_tracer.bundle_adjustment \
    --config "$RUN/config.json" \
    --ckpt   "$RUN/ckpt/checkpoint_final.pt" \
    --out    "$RUN/bundle_adjust" \
    --cycles      ${CYCLES:-40} \
    --block-phi   ${BLOCK_PHI:-200} \
    --block-theta ${BLOCK_THETA:-200} \
    --lr-phi      ${LR_PHI:-1e-5} \
    --lr-theta    ${LR_THETA:-1e-5} \
    --log-every   ${LOG_EVERY:-20} \
    --wandb
