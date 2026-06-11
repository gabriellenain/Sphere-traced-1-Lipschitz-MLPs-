#!/bin/bash
#SBATCH --job-name=gsam-ig-undist
#SBATCH --output=outputs/gsam_ig_undist_%j.log
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
export MPLCONFIGDIR=/tmp/mpl-gsam-undist
python -u tools/grounded_sam_segment.py \
  --image-dir _diagnostics/ignatius_gsam_undist/input \
  --out-dir   _diagnostics/ignatius_gsam_undist \
  --prompt "statue." --keep-top 1 --largest-cc --fill-holes \
  --overview-views 0 1 2 3 4
echo DONE
