#!/bin/bash
#SBATCH --job-name=mvsf-smoke
#SBATCH --output=tea_mvsf_smoke.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=gpu004

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export PYTHONPATH="$HOME/scratch/MVSFormerPlusPlus:${PYTHONPATH:-}"
python -u archive/test_mvsformer.py
