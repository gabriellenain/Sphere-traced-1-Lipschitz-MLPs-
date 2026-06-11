#!/bin/bash
#SBATCH --job-name=gsam-ignatius
#SBATCH --output=outputs/gsam_ignatius_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu004

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export HF_HOME=/scratch/_projets_/willow/1-lip-tracer-new/hf_cache
export MPLCONFIGDIR=/tmp/mpl-gsam-ignatius

OUT=_diagnostics/ignatius_gsam_statue
python -u tools/grounded_sam_segment.py \
  --image-dir "$OUT/input" \
  --out-dir   "$OUT" \
  --prompt "statue." \
  --keep-top 1 \
  --largest-cc \
  --fill-holes \
  --overview-views 0 1 2 3 4

echo "DONE -> $OUT/_overview.png"
