#!/bin/bash
#SBATCH --job-name=acmmp-sph-dtu
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/acmmp_sphere_dtu_%A_%a.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/acmmp_sphere_dtu_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu_homogen
#SBATCH --time=03:00:00
#SBATCH --array=0-14
#
# Sphere-carve the DTU inits at res 512 with ACMMP (non-learning) depths, using
# the SAME recipe as the MVSFormer++ scan24 sphere-carve
# (sphere_carve_auto_res256_nomask_clip2_close2): conf 0.5, votes 3, margin 3vox,
# sfm-clip 2vox, close 2, largest-component, auto sphere centre+radius from
# sparse_sfm_points.txt. Only res (->512) and the depth source (->ACMMP) change.
# Pure NumPy -> runs on the CPU partition (no GPU). Requires per scan the ACMMP
# depths (run_acmmp_dtu.slurm) AND sparse_sfm_points.txt.
set -euo pipefail

# Standard 15-scan DTU eval set.
SCANS=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
SCAN="${SCANS[$SLURM_ARRAY_TASK_ID]}"
SCAN_NAME="scan${SCAN}"

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
DATA=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr
SCENE="${DATA}/${SCAN_NAME}"
DEPTH_DIR="${SCENE}/acmmp_depth_1600x1200"
OUT_DIR="${ROOT}/_diagnostics/acmmp_${SCAN_NAME}_sphere/sphere_carve_auto_res512_nomask_clip2_close2"

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export MPLCONFIGDIR=/tmp/mpl-acmmp-sphere-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "scan=$SCAN_NAME  depth_dir=$DEPTH_DIR  out_dir=$OUT_DIR"

if [[ ! -d "$DEPTH_DIR/$SCAN_NAME/depth_est" ]]; then
    echo "ERROR: missing ACMMP depths: $DEPTH_DIR/$SCAN_NAME/depth_est" >&2; exit 2
fi
if [[ ! -f "$SCENE/sparse_sfm_points.txt" ]]; then
    echo "ERROR: missing $SCENE/sparse_sfm_points.txt" >&2; exit 2
fi
if [[ -f "$OUT_DIR/sphere_mvsformer_carved.ply" ]]; then
    echo "carve already exists; leaving $OUT_DIR untouched"; exit 0
fi

python -u tools/carve_sphere_with_mvsformer_truck.py \
    --scene "$SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$OUT_DIR" \
    --res 512 --bound 1.5 \
    --sphere-radius auto --sphere-radius-percentile 99.9 --sphere-radius-pad 0.02 \
    --center-from-sfm \
    --conf-thr 0.5 --votes-req 3 --margin-voxels 3.0 \
    --sfm-clip --clip-margin-voxels 2.0 \
    --close-iters 2 --largest-component \
    --depth-label ACMMP

echo "[done] $OUT_DIR/sphere_mvsformer_carve_compare.png"
