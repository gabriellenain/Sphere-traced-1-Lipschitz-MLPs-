#!/bin/bash
#SBATCH --job-name=gsam-truck-full
#SBATCH --output=outputs/gsam_truck_full_%j.log
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
export MPLCONFIGDIR=/tmp/mpl-gsam-truck-full

OUT=data/tnt/Truck/mask_truck
python -u tools/grounded_sam_segment.py \
  --image-dir data/tnt/Truck/rgb \
  --out-dir "$OUT" \
  --prompt "truck." \
  --keep-top 1 \
  --largest-cc \
  --fill-holes \
  --overview-views 0 25 50 75 100 125 150 175 200 217

echo "DONE -> $OUT/_overview.png"
