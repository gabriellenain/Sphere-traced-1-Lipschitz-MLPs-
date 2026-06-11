#!/bin/bash
#SBATCH --job-name=mvsf-depth-scan65
#SBATCH --output=outputs/mvsformer_depth_scan65_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
mkdir -p outputs
python -u tools/precompute_mvsformer_depths.py \
  --scene /scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65 \
  --max-h 1152 --max-w 1536 \
  --num-view 5 --n-src 10
