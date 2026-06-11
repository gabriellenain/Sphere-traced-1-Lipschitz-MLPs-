#!/bin/bash
#SBATCH --job-name=photo-diag
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/photo_diag_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/photo_diag_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

CKPT=${CKPT:-outputs/run_20260510_094943_scan65/checkpoint_final.pt}
VIEW=${VIEW:-0}
DOWN=${DOWN:-4}
N_ALT=${N_ALT:-6}
COS_THRESH=${COS_THRESH:-0.1}
T_FAR=${T_FAR:-10.0}
T_NEAR=${T_NEAR:-0.1}
CHUNK=${CHUNK:-32768}
N_COST_RAYS=${N_COST_RAYS:-8}
COST_L=${COST_L:-64}
OUT=${OUT:-outputs/photo_diag_${SLURM_JOB_ID}}

echo "ckpt:  $CKPT"
echo "view:  $VIEW  down: $DOWN  n_alt: $N_ALT"
echo "out:   $OUT"

python -u tools/photo_diag.py \
  --ckpt        "$CKPT" \
  --view        "$VIEW" \
  --down        "$DOWN" \
  --n_alt       "$N_ALT" \
  --cos_thresh  "$COS_THRESH" \
  --t_far       "$T_FAR" \
  --t_near      "$T_NEAR" \
  --chunk       "$CHUNK" \
  --n_cost_rays "$N_COST_RAYS" \
  --cost_L      "$COST_L" \
  --out_dir     "$OUT"
