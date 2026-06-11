#!/bin/bash
#SBATCH --job-name=gsam-ig-full
#SBATCH --output=outputs/gsam_ig_full_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclude=gpu004
set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export HF_HOME=/scratch/_projets_/willow/1-lip-tracer-new/hf_cache
export MPLCONFIGDIR=/tmp/mpl-ig-full
# winning recipe: statue+book+pedestal, low thresholds, largest-CC + fill-holes
python -u tools/grounded_sam_segment.py \
  --image-dir data/tnt/Ignatius/rgb \
  --out-dir   data/tnt/Ignatius/mask_statue \
  --prompt "statue." --prompt "book." --prompt "pedestal." \
  --box-threshold 0.28 --text-threshold 0.18 \
  --keep-top 1 --largest-cc --fill-holes \
  --overview-views 0 52 105 158 210 262
echo "DONE -> data/tnt/Ignatius/mask_statue"
