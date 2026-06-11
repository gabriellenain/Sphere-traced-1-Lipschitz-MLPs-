#!/bin/bash
#SBATCH --job-name=ncc-diag
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/ncc_diag_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/ncc_diag_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Test 0 — NCC-vs-depth profile on the scan24 run with the parasite walls.
# Loads checkpoint_step_150000 with its exact config, profiles the two views
# shown in the normals render (16 and 32). Outputs land in <run>/diag/.

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=outputs/run_20260601_212100_scan24_4938150
STEP=150000

for V in 16 32; do
    python -u analysis/diag_ncc_profile.py \
        --run-dir "$RUN" \
        --step "$STEP" \
        --view "$V" \
        --n-bump 4 --n-clean 4 \
        --n-depth 161 --band 0.04
done
