#!/bin/bash
#SBATCH --job-name=acmmp-dtu
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/acmmp_dtu_%A_%a.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/acmmp_dtu_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004
#SBATCH --array=0-14
#
# Non-learning dense MVS init for DTU: ACMMP (Xu & Tao, multi-scale geometric-
# consistency + planar-prior PatchMatch). Unlike MVSFormer++, ACMMP is trained on
# nothing, so its depths do NOT leak DTU into the init.
#
# Per scan: (1) stage cameras/images into MVSNet format (same staging as the
# MVSFormer++ pipeline, IDR-normalised frame) via precompute_mvsformer_depths
# --stage-only; (2) run the ACMMP binary; (3) convert its .dmb depth/cost output
# into the depth_est/*.pfm + confidence/*.npy layout the carve loader reads.
set -euo pipefail

SCANS=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
SCAN="${SCANS[$SLURM_ARRAY_TASK_ID]}"
SCAN_NAME="scan${SCAN}"

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
DATA=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr
SCENE="${DATA}/${SCAN_NAME}"
ACMMP_DIR=/scratch/_projets_/willow/1-lip-tracer-new/ACMMP
ACMMP_ENV="${ACMMP_DIR}/.pixi/envs/default"
STAGING=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr_acmmp_staging
DENSE="${STAGING}/${SCAN_NAME}"
OUT_DIR="${SCENE}/acmmp_depth_1600x1200"

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export MPLCONFIGDIR=/tmp/mpl-acmmp-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "scan=$SCAN_NAME  scene=$SCENE"
echo "dense=$DENSE  out=$OUT_DIR"

if [[ ! -f "$ACMMP_DIR/build/ACMMP" ]]; then
    echo "ERROR: ACMMP binary not built at $ACMMP_DIR/build/ACMMP" >&2; exit 2
fi

# (1) stage MVSNet input (CPU) — reuse the MVSFormer++ staging verbatim
python -u tools/precompute_mvsformer_depths.py \
    --scene "$SCENE" --scan-name "$SCAN_NAME" \
    --staging "$STAGING" --stage-only

# (2) run ACMMP (GPU). LD_LIBRARY_PATH from the build env resolves opencv/cudart.
echo "[acmmp] running on $DENSE"
LD_LIBRARY_PATH="${ACMMP_ENV}/lib:${LD_LIBRARY_PATH:-}" \
    "$ACMMP_DIR/build/ACMMP" "$DENSE"

# (3) convert .dmb -> depth_est/*.pfm + confidence/*.npy
python -u tools/acmmp_dmb_to_depthdir.py \
    --dense "$DENSE" --out "$OUT_DIR" --scan "$SCAN_NAME"

echo "[done] $OUT_DIR/$SCAN_NAME"
