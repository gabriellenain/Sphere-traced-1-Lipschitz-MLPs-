#!/bin/bash
#SBATCH --job-name=mvsf-depth-scan24
#SBATCH --output=outputs/mvsformer_depth_scan24_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
mkdir -p _diagnostics/mvsformer_scan24_idr

export MPLCONFIGDIR=/tmp/matplotlib-mvsformer-scan24
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u tools/precompute_mvsformer_depths.py \
  --scene /home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/dtu_idr/scan24 \
  --scan-name scan24 \
  --out-dir /home/glenain/Sphere-traced-1-Lipschitz-MLPs-/_diagnostics/mvsformer_scan24_idr/depths \
  --staging /home/glenain/Sphere-traced-1-Lipschitz-MLPs-/_diagnostics/mvsformer_scan24_idr/staging \
  --max-h 864 --max-w 1152 \
  --num-view 5 --n-src 10 \
  --numdepth 128 \
  --keep-staging
