#!/bin/bash
#SBATCH --job-name=grad-iclr
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grad_compare_iclr_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grad_compare_iclr_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

# ---- scan122 checkpoints ---------------------------------------------------
MINE=outputs/run_20260521_014634_scan122_4907463/ckpt/checkpoint_best_photo.pt
NEUS=outputs/neus_neus_4907171/checkpoints/ckpt_300000.pth
GEONEUS=outputs/geoneus_geoneus_4907459/checkpoints/ckpt_300000.pth

# ---- use the actual run configs (contain model.sdf_network) ----------------
NEUS_CONF=outputs/neus_neus_4907171/run.conf
GEONEUS_CONF=outputs/geoneus_geoneus_4907459/run.conf

# ---- slice / viz -----------------------------------------------------------
AXIS=${AXIS:-xz}
OFFSET=${OFFSET:-0.0}
BOUND=${BOUND:-1.0}
RES=${RES:-512}
CMAP=${CMAP:-coolwarm}
DIVERGING=${DIVERGING:-1}        # 1 → diverging colormap centred on |grad f|=1
OUT=${OUT:-outputs/grad_compare_iclr_diverging.png}

EXTRA=""
if [ -n "$VMAX" ]; then EXTRA="$EXTRA --vmax $VMAX"; fi
if [ -n "$VMIN" ]; then EXTRA="$EXTRA --vmin $VMIN"; fi
if [ "$DIVERGING" = "1" ]; then EXTRA="$EXTRA --diverging"; fi

echo "mine     : $MINE"
echo "neus     : $NEUS"
echo "geo-neus : $GEONEUS"
echo "axis=$AXIS offset=$OFFSET bound=$BOUND res=$RES out=$OUT"

MPLCONFIGDIR=/tmp/mpl python -u analysis/grad_compare.py \
    --mine         "$MINE" \
    --neus         "$NEUS" \
    --geoneus      "$GEONEUS" \
    --neus_conf    "$NEUS_CONF" \
    --geoneus_conf "$GEONEUS_CONF" \
    --axis "$AXIS" --offset "$OFFSET" --bound "$BOUND" --res "$RES" \
    --cmap "$CMAP" --out "$OUT" $EXTRA
