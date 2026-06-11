#!/bin/bash
#SBATCH --job-name=truck-carve-gentle
#SBATCH --output=outputs/truck_carve_gentle_%j.log
#SBATCH --error=outputs/truck_carve_gentle_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu004
set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export MPLCONFIGDIR=/tmp/mpl-truck-carve-gentle
OUT=_diagnostics/mvsformer_truck_gsam_clean/hull_carve_p085_gentle
mkdir -p "$OUT"
python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
  --scene data/tnt/Truck \
  --depth-dir _diagnostics/mvsformer_truck/depths_1152x640 \
  --out-dir "$OUT" \
  --res 256 --bound 1.5 \
  --conf-thr 0.5 --votes-req 6 --margin-voxels 5 \
  --vh-percentile 0.85 --vh-min-views 5 \
  --sfm-clip
echo "DONE -> $OUT"
