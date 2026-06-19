#!/bin/bash
#SBATCH --job-name=mvsf-depth-barn
#SBATCH --output=outputs/mvsformer_depth_barn_%j.log
#SBATCH --error=outputs/mvsformer_depth_barn_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu004

set -euo pipefail

# Usage: sbatch precompute_mvsformer_depths_tnt_barn.slurm [SceneName]   # default Barn
SCENE_NAME=${1:-Barn}
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
SCENE=$ROOT/data/tnt/${SCENE_NAME}
OUT=$SCENE/mvsformer_depth_1152x640
STAGING=$SCENE/_mvsf_staging

cd "$ROOT"
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate

export MPLCONFIGDIR=/tmp/matplotlib-mvsformer-${SCENE_NAME,,}
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u tools/precompute_mvsformer_depths.py \
  --scene "$SCENE" \
  --scan-name "$SCENE_NAME" \
  --out-dir "$OUT" \
  --staging "$STAGING" \
  --max-h 640 --max-w 1152 \
  --num-view 5 --n-src 10 \
  --numdepth 128 \
  --keep-staging
