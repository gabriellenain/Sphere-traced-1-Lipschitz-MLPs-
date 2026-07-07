#!/bin/bash
#SBATCH --job-name=neus
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/neus_%x_%j/slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/neus_%x_%j/slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu,willow
#SBATCH --time=48:00:00
# match the 1-Lip runs: steer onto the good H200/A100 nodes (gpu015-018,
# parq-gpu001), off the small/old GPUs. NeuS stacks all images in RAM, so
# 128G is needed too.
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

# ---------- params (edit these) ----------
# DTU:  sbatch run_neus.slurm 122            → SCAN=scan122 (DTU chamfer eval)
# TnT:  DATASET=tnt SCENE=Barn sbatch run_neus.slurm
#                                            → NeuS womask + official TnT F-score
DATASET=${DATASET:-dtu}                         # dtu | tnt
[ -n "$1" ] && SCAN="scan$1"
SCAN=${SCAN:-scan65}                            # DTU scan (dataset=dtu)
SCENE=${SCENE:-Barn}                            # TnT scene  (dataset=tnt)
MODE=${MODE:-train}                             # train | validate_mesh | interpolate_<i>_<j>
IS_CONTINUE=${IS_CONTINUE:-0}                   # 1 to resume / required for validate_mesh
EVAL_EVERY=${EVAL_EVERY:-10000}                 # run chamfer (DTU) / F-score (TnT) every N steps
MESH_RES_PERIODIC=${MESH_RES_PERIODIC:-384}     # MC res for every-EVAL_EVERY eval
MESH_RES_FINAL=${MESH_RES_FINAL:-512}           # MC res for final mesh
WANDB_MODE=${WANDB_MODE:-online}                # online | offline | disabled

# dataset-specific defaults
if [ "$DATASET" = "tnt" ]; then
    CONF=${CONF:-confs/womask.conf}             # no masks for TnT
    WANDB_PROJECT=${WANDB_PROJECT:-neus-tnt}
    CASE=$SCENE
    TNT_SCENE_DIR=${TNT_SCENE_DIR:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt/$SCENE}
    TNT_GT_DIR=${TNT_GT_DIR:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt_gt/$SCENE}
else
    CONF=${CONF:-confs/wmask.conf}              # wmask.conf (object-only) | womask.conf
    WANDB_PROJECT=${WANDB_PROJECT:-neus-dtu}
    CASE=$SCAN
    DTU_ROOT=${DTU_ROOT:-/scratch/_projets_/willow/1-lip-tracer/data/dtu}
    DTU_EVAL_DIR=${DTU_EVAL_DIR:-/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data}
fi

REPO=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
NEUS_DIR=$REPO/baselines/NeuS
OUT_ROOT=$REPO/outputs
# -----------------------------------------

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export WANDB_MODE

# Shared per-case data dir (created once, reused across runs).
CASE_DIR=$NEUS_DIR/public_data/$CASE
if [ "$DATASET" = "tnt" ]; then
    # Convert the own-COLMAP (Neuralangelo path) pose/+rgb/ scene to NeuS
    # cameras_sphere.npz; --force not passed so this is a no-op once built.
    python -u "$REPO/tools/nsvf_to_neus.py" \
        --scene-dir "$TNT_SCENE_DIR" \
        --out-dir   "$CASE_DIR"
else
    if [ ! -e "$CASE_DIR/cameras_sphere.npz" ]; then
        mkdir -p "$CASE_DIR"
        ln -sfn "$DTU_ROOT/$SCAN/image"       "$CASE_DIR/image"
        ln -sfn "$DTU_ROOT/$SCAN/mask"        "$CASE_DIR/mask"
        ln -sfn "$DTU_ROOT/$SCAN/cameras.npz" "$CASE_DIR/cameras_sphere.npz"
    fi
fi

# Per-run output directory: logs, ckpts, meshes, conf snapshot, eval/, slurm.log.
RUN_DIR=$OUT_ROOT/neus_${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# Per-run conf: rewrite base_exp_dir to RUN_DIR and data_dir to the shared CASE_DIR,
# force val_mesh_freq + save_freq = EVAL_EVERY.
RUN_CONF=$RUN_DIR/run.conf
sed -e "s|./exp/CASE_NAME/[^[:space:]]*|$RUN_DIR|g" \
    -e "s|./public_data/CASE_NAME/|$CASE_DIR/|g" \
    -e "s|^\s*val_mesh_freq\s*=.*|    val_mesh_freq = $EVAL_EVERY|g" \
    -e "s|^\s*save_freq\s*=.*|    save_freq = $EVAL_EVERY|g" \
    "$NEUS_DIR/$CONF" > "$RUN_CONF"

EXTRA=""
[ "$IS_CONTINUE" = "1" ] && EXTRA="--is_continue"
if [ "$DATASET" = "tnt" ]; then
    EXTRA="$EXTRA --dataset tnt --tnt_scene $SCENE --tnt_scene_dir $TNT_SCENE_DIR --tnt_gt_dir $TNT_GT_DIR"
else
    EXTRA="$EXTRA --dataset dtu --dtu_eval_dir $DTU_EVAL_DIR"
fi

python -u "$REPO/scripts/run_neus_with_eval.py" \
    --conf "$RUN_CONF" \
    --case "$CASE" \
    --mode "$MODE" \
    --run_dir "$RUN_DIR" \
    --eval_every "$EVAL_EVERY" \
    --mesh_res_periodic "$MESH_RES_PERIODIC" \
    --mesh_res_final "$MESH_RES_FINAL" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${CASE}" \
    $EXTRA
