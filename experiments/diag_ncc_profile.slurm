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
export PYTHONPATH=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-:$PYTHONPATH   # so `import lip_tracer` resolves when running analysis/*.py
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Overridable: RUN=... STEP=... VIEWS="16 32" MODE=profile|bg_ablation sbatch ...
RUN=${RUN:-outputs/run_20260601_212100_scan24_4938150}
STEP=${STEP:-150000}
VIEWS=${VIEWS:-"16 32"}
MODE=${MODE:-profile}
DEVICE=${DEVICE:-cuda}     # set DEVICE=cpu to run on cpu_homogen (override --gres on the CLI)

for V in $VIEWS; do
    if [ "$MODE" = "roi_validpatch" ]; then
        # mask-free ROI variant of the valid-patch figure (COLMAP sparse -> ROI).
        # NPZ defaults to the view's saved ncc_bg_ablation npz in <run>/diag/.
        NPZ_V=${NPZ:-"$RUN/diag/ncc_bg_ablation_v${V}_s${STEP}.npz"}
        python -u analysis/diag_ncc_profile.py \
            --run-dir "$RUN" --step "$STEP" --view "$V" --device "$DEVICE" \
            --roi-validpatch --replot-from "$NPZ_V" \
            --roi-min-views "${ROI_MIN_VIEWS:-0}" --roi-dilate "${ROI_DILATE:--1}"
    elif [ "$MODE" = "grad_conflict" ]; then
        # ICLR (b): θ-gradient conflict between object-hit and background-hit NCC.
        python -u analysis/diag_ncc_profile.py \
            --run-dir "$RUN" --step "$STEP" --view "$V" --device "$DEVICE" --grad-conflict
    elif [ "$MODE" = "bg_ablation" ]; then
        # ICLR background ablation: one frozen ckpt, per-pixel ZNCC real-bg vs
        # mask-baked-black-bg (images differ ONLY outside the mask = pedestal).
        python -u analysis/diag_ncc_profile.py \
            --run-dir "$RUN" --step "$STEP" --view "$V" --device "$DEVICE" --bg-ablation \
            ${CKPT:+--ckpt "$CKPT"} ${OBJECT_MASK_DIR:+--object-mask-dir "$OBJECT_MASK_DIR"}
    else
        python -u analysis/diag_ncc_profile.py \
            --run-dir "$RUN" --step "$STEP" --view "$V" --device "$DEVICE" \
            --n-bump 4 --n-clean 4 --n-depth 161 --band 0.04
    fi
done
