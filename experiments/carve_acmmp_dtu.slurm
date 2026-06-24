#!/bin/bash
#SBATCH --job-name=acmmp-dtu-carve
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/acmmp_dtu_carve_%A_%a.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/acmmp_dtu_carve_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu_homogen
#SBATCH --time=03:00:00
#SBATCH --array=0-14
#
# Carve the DTU visual hulls with ACMMP (non-learning) depths, using the SAME
# recipe as the MVSFormer++ DTU carve (carve_mvsformer_tntckpt_dtu.slurm:
# conf 0.5, votes 3, margin 3vox, clip-margin 6vox) so the only thing that
# changes vs the MVSFormer++ figure is the depth source. Pure NumPy -> CPU
# partition (no GPU). RES selects the voxel grid (default 256; pass RES=512 for
# the high-res variant, which writes to hull_carve_res512 so the 256 result is
# left intact). Requires per scan the ACMMP depths (run_acmmp_dtu.slurm).
set -euo pipefail

# Standard 15-scan DTU eval set (index 5=scan65, index 13=scan118).
SCANS=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
SCAN="${SCANS[$SLURM_ARRAY_TASK_ID]}"
SCAN_NAME="scan${SCAN}"

RES=${RES:-256}
[ "$RES" = "256" ] && RES_SUFFIX="" || RES_SUFFIX="_res${RES}"

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
DATA=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr
SCENE="${DATA}/${SCAN_NAME}"
DEPTH_DIR="${SCENE}/acmmp_depth_1600x1200"
OUT_DIR="${ROOT}/_diagnostics/acmmp_${SCAN_NAME}_idr/hull_carve${RES_SUFFIX}"

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export MPLCONFIGDIR=/tmp/mpl-acmmp-carve-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "scan=$SCAN_NAME  res=$RES  depth_dir=$DEPTH_DIR  out_dir=$OUT_DIR"

if [[ ! -d "$DEPTH_DIR/$SCAN_NAME/depth_est" ]]; then
    echo "ERROR: missing ACMMP depths: $DEPTH_DIR/$SCAN_NAME/depth_est" >&2; exit 2
fi
if [[ -f "$OUT_DIR/vh_mvsformer_carved.ply" ]]; then
    echo "carve already exists; leaving $OUT_DIR untouched"; exit 0
fi

python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
    --scene "$SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$OUT_DIR" \
    --res "$RES" --bound 1.5 \
    --conf-thr 0.5 --votes-req 3 \
    --margin-voxels 3.0 --clip-margin-voxels 6.0 \
    --depth-label ACMMP

echo "[done] $OUT_DIR/vh_mvsformer_carve_compare.png"
