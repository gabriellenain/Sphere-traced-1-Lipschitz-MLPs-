#!/bin/bash
#SBATCH --job-name=mvsf-depth-bmvs-clock
#SBATCH --output=outputs/mvsformer_depth_bmvs_clock_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate

# Scene-parameterized (default keeps the original bmvs_clock behaviour):
#   BMVS=bmvs_sculpture sbatch experiments/precompute_mvsformer_depths_bmvs_clock.slurm
# bmvs_clock and bmvs_sculpture are both 768x576 native (/64); set MAX_H/MAX_W
# for other bmvs scenes whose native resolution differs.
BMVS=${BMVS:-bmvs_clock}
MAX_H=${MAX_H:-576}
MAX_W=${MAX_W:-768}
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
DIAG=$ROOT/_diagnostics/mvsformer_${BMVS}
mkdir -p "$DIAG"

export MPLCONFIGDIR=/tmp/matplotlib-mvsformer-${BMVS}
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "scene=$BMVS  max=${MAX_W}x${MAX_H}  out=$DIAG/depths"
python -u tools/precompute_mvsformer_depths.py \
  --scene "$ROOT/data/bmvs/$BMVS" \
  --scan-name "$BMVS" \
  --out-dir "$DIAG/depths" \
  --staging "$DIAG/staging" \
  --max-h "$MAX_H" --max-w "$MAX_W" \
  --num-view 5 --n-src 10 \
  --numdepth 128 \
  --keep-staging
