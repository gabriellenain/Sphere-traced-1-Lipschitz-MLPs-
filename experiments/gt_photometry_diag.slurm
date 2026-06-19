#!/bin/bash
#SBATCH --job-name=gt_photom
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/gt_photom_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/gt_photom_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTHONPATH=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-:${PYTHONPATH:-}

# GT-surface NCC photometry diagnostic (analysis/gt_photometry_diag.py), reusable
# across DTU scans. Reproduces the "dense GT-plane photometry overlays" CVPR grid
# (gray-NCC / RGB-NCC / gray-RGB on 10 auto-selected reference views, 300k dense
# GT points/view). All NCC scores come from the unchanged training pmvs_ncc_loss.
# SCAN selects the DTU scan; OUT overridable. Same data roots as scan24/65/122 runs.
SCAN=${SCAN:-37}
DATA=${DATA:-/scratch/_projets_/willow/1-lip-tracer-new/data}
OUT=${OUT:-artifacts/gt_photometry_scan${SCAN}_cvpr_grid_points}

python -u analysis/gt_photometry_diag.py \
    --scene "$DATA/dtu_idr/scan${SCAN}" \
    --dataset dtu \
    --dtu-eval-dir "$DATA/dtu_eval/SampleSet/MVS Data" \
    --scan-id "$SCAN" \
    --n-overlays "${N_OVERLAYS:-10}" \
    --dense-overlay-points-per-view "${POINTS:-300000}" \
    --out "$OUT" \
    ${EXTRA_ARGS:-}
