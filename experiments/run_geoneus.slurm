#!/bin/bash
#SBATCH --job-name=geoneus
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/geoneus_%x_%j/slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/geoneus_%x_%j/slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu004,gpu012

# ---------- params (edit these) ----------
# Positional arg: `sbatch run_geoneus.slurm 122` → SCAN=scan122.
[ -n "$1" ] && SCAN="scan$1"
SCAN=${SCAN:-scan65}
CONF=${CONF:-confs/wmask.conf}               # wmask.conf (object-only) | womask.conf
MODE=${MODE:-train}
IS_CONTINUE=${IS_CONTINUE:-0}
EVAL_EVERY=${EVAL_EVERY:-10000}
MESH_RES_PERIODIC=${MESH_RES_PERIODIC:-384}
MESH_RES_FINAL=${MESH_RES_FINAL:-512}
WANDB_PROJECT=${WANDB_PROJECT:-geoneus-dtu}
WANDB_MODE=${WANDB_MODE:-online}

DTU_ROOT=${DTU_ROOT:-/scratch/_projets_/willow/1-lip-tracer/data/dtu}
DTU_EVAL_DIR=${DTU_EVAL_DIR:-/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data}
REPO=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
GEONEUS_DIR=$REPO/baselines/Geo-Neus
OUT_ROOT=$REPO/outputs
# -----------------------------------------

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export WANDB_MODE

# Shared per-scan data dir under Geo-Neus's public_data/ (created once, reused across runs).
# Geo-Neus expects: image/, mask/, cameras.npz, sfm_pts/{points,view_id}.npy, pairs.txt
SCAN_DIR=$GEONEUS_DIR/public_data/$SCAN
if [ ! -e "$SCAN_DIR/cameras.npz" ]; then
    mkdir -p "$SCAN_DIR"
    ln -sfn "$DTU_ROOT/$SCAN/image"       "$SCAN_DIR/image"
    ln -sfn "$DTU_ROOT/$SCAN/mask"        "$SCAN_DIR/mask"
    ln -sfn "$DTU_ROOT/$SCAN/cameras.npz" "$SCAN_DIR/cameras.npz"
    ln -sfn "$DTU_ROOT/$SCAN/sfm_pts"     "$SCAN_DIR/sfm_pts"
    ln -sfn "$DTU_ROOT/$SCAN/pairs.txt"   "$SCAN_DIR/pairs.txt"
fi

# Per-run output directory.
RUN_DIR=$OUT_ROOT/geoneus_${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# Per-run conf: rewrite base_exp_dir + data_dir, force val_mesh_freq + save_freq.
RUN_CONF=$RUN_DIR/run.conf
sed -e "s|./exp/CASE_NAME/[^[:space:]]*|$RUN_DIR|g" \
    -e "s|./public_data/CASE_NAME/|$SCAN_DIR/|g" \
    -e "s|^\s*val_mesh_freq\s*=.*|    val_mesh_freq = $EVAL_EVERY|g" \
    -e "s|^\s*save_freq\s*=.*|    save_freq = $EVAL_EVERY|g" \
    "$GEONEUS_DIR/$CONF" > "$RUN_CONF"

EXTRA=""
[ "$IS_CONTINUE" = "1" ] && EXTRA="--is_continue"

python -u "$REPO/scripts/run_geoneus_with_eval.py" \
    --conf "$RUN_CONF" \
    --case "$SCAN" \
    --mode "$MODE" \
    --run_dir "$RUN_DIR" \
    --dtu_eval_dir "$DTU_EVAL_DIR" \
    --eval_every "$EVAL_EVERY" \
    --mesh_res_periodic "$MESH_RES_PERIODIC" \
    --mesh_res_final "$MESH_RES_FINAL" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SCAN}" \
    $EXTRA
