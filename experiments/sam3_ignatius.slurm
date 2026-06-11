#!/bin/bash
#SBATCH --job-name=sam3-ignatius
#SBATCH --output=outputs/sam3_ignatius_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu004
set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/sam3_env/sam3venv/bin/activate
export HF_HOME=/scratch/_projets_/willow/1-lip-tracer-new/hf_cache
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/mpl-sam3
python -u tools/sam3_segment.py \
  --image-dir _diagnostics/ignatius_gsam_undist/input \
  --out-dir   _diagnostics/ignatius_sam3_multi \
  --prompt "statue" --prompt "pedestal" --prompt "book" \
  --score-thresh 0.3 --largest-cc --fill-holes \
  --overview 0 1 2 3 4
echo DONE
