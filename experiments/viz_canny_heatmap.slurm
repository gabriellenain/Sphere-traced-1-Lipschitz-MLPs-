#!/bin/bash
#SBATCH --job-name=canny-heat
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/canny_heatmap_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/canny_heatmap_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

DTU=/scratch/_projets_/willow/1-lip-tracer/data/dtu
# scan65 has 49 views (0..48); scan122 has more, so v50 only valid there.
PAIRS=${PAIRS:-"scan65:16 scan122:16 scan122:50"}
METHODS=${METHODS:-"canny harris sobel"}
DOWN=${DOWN:-1}; ALPHA=${ALPHA:-0.8}; LO=${LO:-0.06}; HI=${HI:-0.15}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for pv in $PAIRS; do
  scene=${pv%%:*}; view=${pv##*:}
  for m in $METHODS; do
    echo "=== [$m] $scene view $view ==="
    MPLCONFIGDIR=/tmp/mpl python -u tools/viz_canny_heatmap.py \
      "$DTU/$scene" "$view" "$DOWN" "$ALPHA" "$LO" "$HI" "$m"
  done
done
