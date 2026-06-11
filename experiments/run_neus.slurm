#!/bin/bash
#SBATCH --job-name=neus
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/neus_%x_%j/slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/neus_%x_%j/slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu004,gpu012

# ---------- params (edit these) ----------
# Positional arg: `sbatch run_neus.slurm 122` → SCAN=scan122.
[ -n "$1" ] && SCAN="scan$1"
SCAN=${SCAN:-scan65}                           # DTU scan
CONF=${CONF:-confs/wmask.conf}                 # wmask.conf (object-only) | womask.conf
MODE=${MODE:-train}                            # train | validate_mesh | interpolate_<i>_<j>
IS_CONTINUE=${IS_CONTINUE:-0}                  # 1 to resume / required for validate_mesh
EVAL_EVERY=${EVAL_EVERY:-10000}                # run DTU chamfer every N steps
MESH_RES_PERIODIC=${MESH_RES_PERIODIC:-384}    # MC res for every-EVAL_EVERY chamfer (matches your dtu_official_res)
MESH_RES_FINAL=${MESH_RES_FINAL:-512}          # MC res for final mesh (matches analysis/eval_dtu_official.py --res)
WANDB_PROJECT=${WANDB_PROJECT:-neus-dtu}
WANDB_MODE=${WANDB_MODE:-online}               # online | offline | disabled

DTU_ROOT=${DTU_ROOT:-/scratch/_projets_/willow/1-lip-tracer/data/dtu}
DTU_EVAL_DIR=${DTU_EVAL_DIR:-/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data}
REPO=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
NEUS_DIR=$REPO/baselines/NeuS
OUT_ROOT=$REPO/outputs
# -----------------------------------------

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export WANDB_MODE

# Shared per-scan data dir (created once, reused across runs).
SCAN_DIR=$NEUS_DIR/public_data/$SCAN
if [ ! -e "$SCAN_DIR/cameras_sphere.npz" ]; then
    mkdir -p "$SCAN_DIR"
    ln -sfn "$DTU_ROOT/$SCAN/image"       "$SCAN_DIR/image"
    ln -sfn "$DTU_ROOT/$SCAN/mask"        "$SCAN_DIR/mask"
    ln -sfn "$DTU_ROOT/$SCAN/cameras.npz" "$SCAN_DIR/cameras_sphere.npz"
fi

# Per-run output directory: logs, ckpts, meshes, conf snapshot, dtu_eval/, slurm.log.
RUN_DIR=$OUT_ROOT/neus_${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# Per-run conf: rewrite base_exp_dir to RUN_DIR and data_dir to the shared SCAN_DIR,
# force val_mesh_freq + save_freq = EVAL_EVERY.
RUN_CONF=$RUN_DIR/run.conf
sed -e "s|./exp/CASE_NAME/[^[:space:]]*|$RUN_DIR|g" \
    -e "s|./public_data/CASE_NAME/|$SCAN_DIR/|g" \
    -e "s|^\s*val_mesh_freq\s*=.*|    val_mesh_freq = $EVAL_EVERY|g" \
    -e "s|^\s*save_freq\s*=.*|    save_freq = $EVAL_EVERY|g" \
    "$NEUS_DIR/$CONF" > "$RUN_CONF"

EXTRA=""
[ "$IS_CONTINUE" = "1" ] && EXTRA="--is_continue"

python -u "$REPO/scripts/run_neus_with_eval.py" \
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
