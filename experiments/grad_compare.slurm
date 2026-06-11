#!/bin/bash
#SBATCH --job-name=grad-compare
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grad_compare_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grad_compare_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

# ---- checkpoints (edit) ----------------------------------------------------
MINE=${MINE:-outputs/run_20260430_171910_dtu/checkpoint_best_photo.pt}
NEUS=${NEUS:-baselines/NeuS/exp/scan65/wmask/checkpoints/ckpt_300000.pth}
GEONEUS=${GEONEUS:-baselines/Geo-Neus/exp/scan65/wmask/checkpoints/ckpt_300000.pth}

# ---- baseline configs ------------------------------------------------------
NEUS_CONF=${NEUS_CONF:-baselines/NeuS/confs/wmask.conf}
GEONEUS_CONF=${GEONEUS_CONF:-baselines/Geo-Neus/confs/wmask.conf}

# ---- slice / viz -----------------------------------------------------------
AXIS=${AXIS:-xz}
OFFSET=${OFFSET:-0.0}
BOUND=${BOUND:-1.0}
RES=${RES:-512}
VMAX=${VMAX:-}        # empty → auto (99th percentile)
CMAP=${CMAP:-viridis}
OUT=${OUT:-outputs/grad_compare_${SLURM_JOB_ID}.png}

echo "mine     : $MINE"
echo "neus     : $NEUS"
echo "geo-neus : $GEONEUS"
echo "axis     : $AXIS   offset=$OFFSET   bound=$BOUND   res=$RES"
echo "out      : $OUT"

EXTRA=""
if [ -n "$VMAX" ]; then EXTRA="$EXTRA --vmax $VMAX"; fi

MPLCONFIGDIR=/tmp/mpl python -u analysis/grad_compare.py \
    --mine         "$MINE" \
    --neus         "$NEUS" \
    --geoneus      "$GEONEUS" \
    --neus_conf    "$NEUS_CONF" \
    --geoneus_conf "$GEONEUS_CONF" \
    --axis         "$AXIS" \
    --offset       "$OFFSET" \
    --bound        "$BOUND" \
    --res          "$RES" \
    --cmap         "$CMAP" \
    --out          "$OUT" \
    $EXTRA
