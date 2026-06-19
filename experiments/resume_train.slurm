#!/bin/bash
#SBATCH --job-name=1lip-resume
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-resume_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-resume_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

# Generic continuation launcher: resume an existing run's weights from a checkpoint
# and extend training to a larger horizon. Defaults extend the Ignatius sphere-nomask
# run (step 299500 -> 600000) with a fresh-LR cosine warm restart and the official
# TnT F-score logged every TNT_OFFICIAL_FREQ steps. Override the env vars to retarget.
#   RESUME_CKPT      checkpoint to resume weights from (ckpt/checkpoint_latest.pt)
#   STEPS            new absolute end step (training runs [resume_step, STEPS))
#   LR_WARM_RESTART  1 = reset optimiser + cosine LR over the remaining segment
#   TNT_SCENE_NAME   official TnT scene name to enable periodic F-score (empty = off)

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-

RESUME_CKPT=${RESUME_CKPT:-$ROOT/outputs/run_20260606_211917_ignatius_sphere_nomask_res384_init_4992251/ckpt/checkpoint_latest.pt}
STEPS=${STEPS:-600000}
LR_WARM_RESTART=${LR_WARM_RESTART:-1}

# Trace / mask params must match the original run (the --config path overrides occ_*
# from these args unconditionally, so omitting them would silently change the tracer).
OCC_ITERS=${OCC_ITERS:-32}
OCC_NEWTON_STEPS=${OCC_NEWTON_STEPS:-0}
OCC_EPS=${OCC_EPS:-1e-3}
USE_MASKS=${USE_MASKS:-0}              # 0 -> --no-use-masks
COMPILE=${COMPILE:-1}                  # 0 -> --no-compile (eager; avoids inductor recompile/CUDA wedges)

# Official DTU Chamfer (DTUeval-python), reuses the run's periodic eval cadence.
# Leave DTU_EVAL_DIR empty to keep DTU eval off (the train.py resume path resets
# dtu_official_freq to its CLI default of 0 unless we re-pass these flags).
DTU_EVAL_DIR=${DTU_EVAL_DIR:-}
DTU_OFFICIAL_FREQ=${DTU_OFFICIAL_FREQ:-10000}
DTU_OFFICIAL_RES=${DTU_OFFICIAL_RES:-384}
DTU_OFFICIAL_BOUND=${DTU_OFFICIAL_BOUND:-1.0}

# Official TnT F-score, same good-alignment recipe as eval_tnt_official.py.
# Use `-` (not `:-`) so an explicitly-empty TNT_SCENE_NAME disables TnT eval
# (e.g. when resuming a DTU run); only an UNSET var falls back to Ignatius.
TNT_SCENE_NAME=${TNT_SCENE_NAME-Ignatius}
TNT_EVAL=${TNT_EVAL:-$ROOT/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}

if [ ! -f "$RESUME_CKPT" ]; then
  echo "ERROR: missing resume checkpoint: $RESUME_CKPT" >&2
  exit 1
fi

# Resume reuses the original run dir (run_dir = ckpt/../..). Write a fresh log so the
# original train.log (the completed 0..300k history) is preserved.
RUN_DIR=$(dirname "$(dirname "$RESUME_CKPT")")
LOG=$RUN_DIR/train_resume_${STEPS}.log

# train.py rewrites config.json with the new (extended) config on resume; preserve the
# original record once so the pre-resume config is not lost.
[ -f "$RUN_DIR/config.json" ] && cp -n "$RUN_DIR/config.json" "$RUN_DIR/config_pre_resume.json"

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-1lip-resume
# Use in-process dladdr symbolization instead of spawning addr2line, which parses
# hundreds of MB of DWARF from libtorch_*.so off NFS on first use and stalls the
# training thread for minutes whenever PyTorch builds a C++ traceback.
export TORCH_SYMBOLIZE_MODE=dladdr

MASK_ARG=(--no-use-masks)
[ "$USE_MASKS" = "1" ] && MASK_ARG=(--use-masks)

COMPILE_ARG=()
[ "$COMPILE" = "0" ] && COMPILE_ARG=(--no-compile)

WARM_ARG=()
[ "$LR_WARM_RESTART" = "1" ] && WARM_ARG=(--lr-warm-restart)

TNT_ARGS=()
if [ -n "$TNT_SCENE_NAME" ]; then
  TNT_ARGS=(--tnt-eval-dir "$TNT_EVAL" --tnt-official-scene "$TNT_SCENE_NAME" \
            --tnt-official-frame "$TNT_OFFICIAL_FRAME" --tnt-official-freq "$TNT_OFFICIAL_FREQ" \
            --tnt-official-res "$TNT_OFFICIAL_RES" --tnt-official-n-samples "$TNT_OFFICIAL_N_SAMPLES")
fi

DTU_ARGS=()
if [ -n "$DTU_EVAL_DIR" ]; then
  DTU_ARGS=(--dtu-eval-dir "$DTU_EVAL_DIR" --dtu-official-freq "$DTU_OFFICIAL_FREQ" \
            --dtu-official-res "$DTU_OFFICIAL_RES" --dtu-official-bound "$DTU_OFFICIAL_BOUND")
fi

echo "Resume ckpt:       $RESUME_CKPT"
echo "Run dir (reused):  $RUN_DIR"
echo "Steps (new end):   $STEPS"
echo "LR warm restart:   $LR_WARM_RESTART"
echo "Trace:             occ-iters=$OCC_ITERS occ-newton=$OCC_NEWTON_STEPS occ-eps=$OCC_EPS  masks=${MASK_ARG[*]}"
echo "TnT eval:          ${TNT_ARGS[*]:-<off>}"
echo "DTU eval:          ${DTU_ARGS[*]:-<off>}"
echo "Log:               $LOG"

python -u -m lip_tracer.train \
    --resume "$RESUME_CKPT" \
    --steps "$STEPS" \
    --occ-iters "$OCC_ITERS" \
    --occ-newton-steps "$OCC_NEWTON_STEPS" \
    --occ-eps "$OCC_EPS" \
    "${MASK_ARG[@]}" \
    ${COMPILE_ARG[@]+"${COMPILE_ARG[@]}"} \
    ${WARM_ARG[@]+"${WARM_ARG[@]}"} \
    ${TNT_ARGS[@]+"${TNT_ARGS[@]}"} \
    ${DTU_ARGS[@]+"${DTU_ARGS[@]}"} \
    --wandb 2>&1 | tee "$LOG"
