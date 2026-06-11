#!/bin/bash
#SBATCH --job-name=mvsf-carve-ignatius
#SBATCH --output=outputs/mvsf_carve_ignatius_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004

# VH-vs-MVSFormer++-carved comparison for TnT Ignatius, using the gsam
# statue+fountain masks (undistorted into the COLMAP pinhole frame to match
# the cameras). TnT path: visibility-percentile visual hull, no SFM ROI/clip.

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-mvsformer-ignatius
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=$PWD/data/tnt/Ignatius
BASE=$PWD/_diagnostics/mvsformer_ignatius
DEPTH_DIR=$BASE/depths_1152x640    # MVSFormer++ appends resolution to --out-dir
mkdir -p "$BASE"

# --- 1. MVSFormer++ depths (TnT staging; skip if present) ---
if [ ! -d "$DEPTH_DIR/Ignatius/depth_est" ]; then
  python -u tools/precompute_mvsformer_depths.py \
    --scene "$SCENE" \
    --scan-name Ignatius \
    --out-dir "$BASE/depths" \
    --staging "$BASE/staging" \
    --max-h 640 --max-w 1152 \
    --num-view 5 --n-src 10 \
    --numdepth 128 \
    --keep-staging
fi

# --- 2. carve + render comparison PNG ---
python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
  --scene "$SCENE" \
  --depth-dir "$DEPTH_DIR" \
  --out-dir "$BASE/hull_carve" \
  --res 256 --bound 1.5 \
  --conf-thr 0.5 --votes-req 3

echo "DONE -> $BASE/hull_carve/vh_mvsformer_carve_compare.png"
