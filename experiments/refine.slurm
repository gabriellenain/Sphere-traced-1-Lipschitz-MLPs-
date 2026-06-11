#!/bin/bash
#SBATCH --job-name=1lip-refine
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/refine_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/refine_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

PT=${PT:-outputs/checkpoint_best.pt}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan122}
OUT_DIR=${OUT_DIR:-outputs}
TAG=${TAG:-}
STEPS=${STEPS:-10000}
BATCH=${BATCH:-4096}
LR=${LR:-1e-4}
DELTA_MAX=${DELTA_MAX:-0.01}
W_REG=${W_REG:-1e-3}
W_PHOTO=${W_PHOTO:-0.0}
W_NCC=${W_NCC:-1.0}
NCC_PATCH=${NCC_PATCH:-5}
NCC_HALF_PIX=${NCC_HALF_PIX:-2.0}
NCC_MIN=${NCC_MIN:-0.0}

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "pt:        $PT"
echo "scene:     $SCENE"
echo "out_dir:   $OUT_DIR  tag: ${TAG:-auto}"
echo "steps:     $STEPS  batch: $BATCH  lr: $LR"
echo "delta_max: $DELTA_MAX  w_reg: $W_REG"
echo "w_photo:   $W_PHOTO  w_ncc: $W_NCC  ncc_patch: $NCC_PATCH  ncc_half_pix: $NCC_HALF_PIX  ncc_min: $NCC_MIN"

EXTRA=()
[ -n "$TAG" ] && EXTRA+=(--tag "$TAG")

python -u experiments/refine.py \
    --pt           "$PT"           \
    --scene        "$SCENE"        \
    --out-dir      "$OUT_DIR"      \
    --steps        "$STEPS"        \
    --batch        "$BATCH"        \
    --lr           "$LR"           \
    --delta-max    "$DELTA_MAX"    \
    --w-reg        "$W_REG"        \
    --w-photo      "$W_PHOTO"      \
    --w-ncc        "$W_NCC"        \
    --ncc-patch    "$NCC_PATCH"    \
    --ncc-half-pix "$NCC_HALF_PIX" \
    --ncc-min      "$NCC_MIN"      \
    "${EXTRA[@]}"
