#!/bin/bash
#SBATCH --job-name=truck-hull-sweep
#SBATCH --output=outputs/truck_hull_sweep_%j.log
#SBATCH --error=outputs/truck_hull_sweep_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu004

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export MPLCONFIGDIR=/tmp/mpl-truck-hull-sweep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u analysis/_truck_hull_sweep.py \
  --scene data/tnt/Truck \
  --res 256 --bound 1.5 \
  --percentiles 0.85 0.90 0.95 0.98 0.99 \
  --min-views 5 \
  --out _diagnostics/mvsformer_truck_gsam_clean/hull_sweep

echo "DONE -> _diagnostics/mvsformer_truck_gsam_clean/hull_sweep/hull_sweep.png"
