#!/bin/bash
#SBATCH --job-name=sam3v-ignatius
#SBATCH --output=outputs/sam3v_ignatius_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=gpu004
set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/sam3_env/sam3venv/bin/activate
export HF_HOME=/scratch/_projets_/willow/1-lip-tracer-new/hf_cache
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/mpl-sam3v
export CPATH=/scratch/_projets_/willow/1-lip-tracer-new/py39_dev_include/python3.9:${CPATH:-}
python -u tools/sam3_video_segment.py \
  --rgb-dir data/tnt/Ignatius/rgb \
  --out-dir _diagnostics/ignatius_sam3_video_multi \
  --prompt "statue" --prompt "book" --prompt "pedestal" --seed-frame 0
echo DONE
