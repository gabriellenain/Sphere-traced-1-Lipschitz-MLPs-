#!/bin/bash
#SBATCH --job-name=gsam-caterpillar-full
#SBATCH --output=outputs/gsam_caterpillar_full_%j.log
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
export MPLCONFIGDIR=/tmp/mpl-gsam-caterpillar-full
OUT=data/tnt/Caterpillar/mask_caterpillar
# TnT "Caterpillar" is a yellow wheel loader / bulldozer (construction machine).
python -u tools/grounded_sam_segment.py \
  --image-dir data/tnt/Caterpillar/rgb \
  --out-dir "$OUT" \
  --prompt "bulldozer." --prompt "wheel loader." --prompt "construction vehicle." \
  --box-threshold 0.28 --text-threshold 0.18 \
  --keep-top 1 --largest-cc --fill-holes \
  --overview-views 0 40 80 120 160 200 240 280 320 380
echo "DONE -> $OUT/_overview.png"
