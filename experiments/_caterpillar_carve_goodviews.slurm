#!/bin/bash
#SBATCH --job-name=cat-carve-gv
#SBATCH --output=outputs/caterpillar_carve_goodviews_%j.log
#SBATCH --error=outputs/caterpillar_carve_goodviews_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu004
set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export MPLCONFIGDIR=/tmp/mpl-cat-carve-gv
OUT=_diagnostics/mvsformer_caterpillar_gsam_clean/hull_carve_goodviews
GOOD=_diagnostics/mvsformer_caterpillar_gsam_clean/good_views_p90.txt
mkdir -p "$OUT"
# Conservative carve, restricted to the 361 clean-mask views (drops the 22 views
# whose gsam masks leak onto ground/background). Cleaner percentile hull + carve;
# init contains 100% of the object.
python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
  --scene data/tnt/Caterpillar \
  --depth-dir _diagnostics/mvsformer_caterpillar/depths_1152x640 \
  --out-dir "$OUT" \
  --res 256 --bound 1.5 \
  --conf-thr 0.8 --votes-req 12 --margin-voxels 10 \
  --vh-percentile 0.85 --vh-min-views 5 \
  --sfm-clip \
  --good-views "$GOOD"
echo "DONE -> $OUT"
