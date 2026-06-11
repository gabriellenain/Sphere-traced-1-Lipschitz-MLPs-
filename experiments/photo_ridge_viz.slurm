#!/bin/bash
#SBATCH --job-name=photo-ridge
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/photo_ridge_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/photo_ridge_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65}
DATASET=${DATASET:-dtu}
OUT=${OUT:-outputs/photo_ridge_scan65_${SLURM_JOB_ID}}
RES=${RES:-64}
BOUND=${BOUND:-1.5}
VIEWS=${VIEWS:-0:49:2}
PATCH_RADIUS=${PATCH_RADIUS:-3}
MIN_VIEWS=${MIN_VIEWS:-3}
MAX_VIEWS=${MAX_VIEWS:-8}
SIGMA_VOX=${SIGMA_VOX:-1.0}
P_QUANTILE=${P_QUANTILE:-0.70}
G_QUANTILE=${G_QUANTILE:-0.15}
CHUNK=${CHUNK:-2048}
DEVICE=${DEVICE:-cuda}
PNG_SCALE=${PNG_SCALE:-6}

python -u tools/photo_ridge_viz.py \
  --scene "$SCENE" \
  --dataset "$DATASET" \
  --out_dir "$OUT" \
  --res "$RES" \
  --bound "$BOUND" \
  --views "$VIEWS" \
  --patch-radius "$PATCH_RADIUS" \
  --min-views "$MIN_VIEWS" \
  --max-views "$MAX_VIEWS" \
  --sigma-vox "$SIGMA_VOX" \
  --p-quantile "$P_QUANTILE" \
  --g-quantile "$G_QUANTILE" \
  --png-scale "$PNG_SCALE" \
  --chunk "$CHUNK" \
  --device "$DEVICE"
