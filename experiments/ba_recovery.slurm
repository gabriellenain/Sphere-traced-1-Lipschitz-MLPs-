#!/bin/bash
#SBATCH --job-name=ba_recovery
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/ba_recovery_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/ba_recovery_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=6:00:00
#SBATCH --exclude=gpu004

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Minimal synthetic pose-recovery test for photometric BA.
#
# Goal: confirm the pose-gradient path (idr_intersection → φ) actually works,
# in isolation, before trusting BA on real (un-perturbed) cameras.
#
# Design (the cleanest minimal test):
#   * perturb ONE camera only (PERTURB_CAM, default 1); every other camera —
#     including the gauge-locked cam 0 — stays at its calibrated GT pose. Those
#     fixed GT cameras + the locked gauge pin the scene and the SDF, so the
#     perturbed camera has a UNIQUE correct pose to recover (no gauge drift).
#   * θ FROZEN (BLOCK_THETA=0): pure φ optimisation. If θ also moved, the SDF
#     could deform to absorb the pose error and the recovery metric would lie.
#   * intrinsics OFF (default): R, t only — DTU calibration trusted.
#   * small perturbation (ROT/TRANS) so we start inside the convergence basin;
#     sweep magnitude across jobs to map the basin radius.
#
# Read the result from ba_metrics.csv columns rot_err_gt_deg / trans_err_gt_mm
# (NOT E): success = both decay toward ~0 over the φ steps.
RUN=${RUN:-outputs/run_20260602_220736_scan118_occ_4954645}
CKPT=${CKPT:-$RUN/ckpt/checkpoint_final.pt}
TAG=${TAG:-cam${PERTURB_CAM:-1}_r${ROT:-1.0}_t${TRANS:-2.0}}

python -u -m lip_tracer.bundle_adjustment \
    --config "$RUN/config.json" \
    --ckpt   "$CKPT" \
    --out    "$RUN/ba_recovery_$TAG" \
    --cycles          ${CYCLES:-1} \
    --block-phi       ${BLOCK_PHI:-2000} \
    --block-theta     ${BLOCK_THETA:-0} \
    --lr-phi          ${LR_PHI:-1e-3} \
    --log-every       ${LOG_EVERY:-20} \
    --perturb-cam     ${PERTURB_CAM:-1} \
    --perturb-rot-deg ${ROT:-1.0} \
    --perturb-trans-mm ${TRANS:-2.0} \
    --perturb-seed    ${SEED:-0}
