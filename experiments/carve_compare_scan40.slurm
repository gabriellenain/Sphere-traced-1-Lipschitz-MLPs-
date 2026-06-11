#!/bin/bash
#SBATCH --job-name=mvsf-carve-scan40
#SBATCH --output=outputs/mvsf_carve_scan40_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

# Produce the VH-vs-MVSFormer++-carved comparison PNG for DTU scan40.
# scan40 has sparse_sfm_points.txt, so the carve uses the full DTU path
# (edge-aware VH + SFM ROI + AABB clip) — no code changes.

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-mvsformer-scan40
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=$PWD/data/dtu_idr/scan40
BASE=$PWD/_diagnostics/mvsformer_scan40_idr
DEPTH_DIR=$BASE/depths_1152x864   # MVSFormer++ appends the resolution to --out-dir
mkdir -p "$BASE"

# --- 1. MVSFormer++ depths (skip if already present) ---
if [ ! -d "$DEPTH_DIR/scan40/depth_est" ]; then
  python -u tools/precompute_mvsformer_depths.py \
    --scene "$SCENE" \
    --scan-name scan40 \
    --out-dir "$BASE/depths" \
    --staging "$BASE/staging" \
    --max-h 864 --max-w 1152 \
    --num-view 5 --n-src 10 \
    --numdepth 128 \
    --keep-staging
fi

# --- 2. carve + render the comparison PNG ---
python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
  --scene "$SCENE" \
  --depth-dir "$DEPTH_DIR" \
  --out-dir "$BASE/hull_carve" \
  --res 256 --bound 1.5 \
  --conf-thr 0.5 --votes-req 3

echo "DONE -> $BASE/hull_carve/vh_mvsformer_carve_compare.png"
