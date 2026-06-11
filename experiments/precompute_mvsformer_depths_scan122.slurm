#!/bin/bash
#SBATCH --job-name=mvsf-depth-scan122
#SBATCH --output=outputs/mvsformer_depth_scan122_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu011,gpu014

set -euo pipefail
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
mkdir -p outputs
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan122}
MAX_H=${MAX_H:-1152}
MAX_W=${MAX_W:-1536}
NUM_VIEW=${NUM_VIEW:-5}
N_SRC=${N_SRC:-10}
python -u tools/precompute_mvsformer_depths.py \
  --scene "$SCENE" \
  --max-h "$MAX_H" --max-w "$MAX_W" \
  --num-view "$NUM_VIEW" --n-src "$N_SRC"
