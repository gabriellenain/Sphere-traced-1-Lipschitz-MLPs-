#!/bin/bash
#SBATCH --job-name=gtsam-ignatius
#SBATCH --output=outputs/gtsam_ignatius_%j.log
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
export MPLCONFIGDIR=/tmp/mpl-gtsam
python -u tools/gt_sam_masks.py \
  --scene data/tnt/Ignatius --gt-dir data/tnt_gt/Ignatius --gt-name Ignatius \
  --out-dir _diagnostics/ignatius_gt_sam \
  --stems 0_0000_000001 0_0064_000065 0_0129_000130 0_0194_000195 0_0259_000260
echo DONE
