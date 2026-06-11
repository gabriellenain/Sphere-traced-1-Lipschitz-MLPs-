#!/bin/bash
#SBATCH --job-name=1lip-tnt-hull
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-tnt-hull_%j.slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-tnt-hull_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# Usage: sbatch train_tnt_hullinit.slurm [SceneName]   # default: Barn
# Pipeline: visibility-aware percentile visual hull (p=0.99) -> fit_to_hull -> train.

SCENE_NAME=${1:-Barn}
SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt/${SCENE_NAME}
TNT_EVAL=${TNT_EVAL:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUTS=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_tnt_${SCENE_NAME}_hullp099_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$OUTPUTS/1lip-tnt-hull_${SLURM_JOB_ID}.rundir"
LOG=$RUN_DIR/train.log
echo "Logging to $LOG  (scene: $SCENE)"
echo "Official TnT eval: every ${TNT_OFFICIAL_FREQ} steps  res=${TNT_OFFICIAL_RES}  n=${TNT_OFFICIAL_N_SAMPLES}  frame=${TNT_OFFICIAL_FRAME}"
if [[ ! -f "$TNT_EVAL/${SCENE_NAME}/${SCENE_NAME}.ply" ]]; then
    echo "ERROR: missing official TnT GT assets under $TNT_EVAL/${SCENE_NAME}"
    exit 1
fi

# ---- Step 1: build visibility-aware percentile hull init at p=0.99 ----
INIT_DIR=$RUN_DIR/hull_init
mkdir -p "$INIT_DIR"
python -u tools/fit_hull_barn_pct.py \
    --scene "$SCENE" \
    --out-dir "$INIT_DIR" \
    --res 192 --bound 1.5 --down 2 --fit-steps 2000 \
    --thresholds 0.99 --best-thresh 0.99 \
    --min-views 8 2>&1 | tee "$INIT_DIR/log.txt"

if [[ ! -f "$INIT_DIR/init.pt" ]]; then
    echo "ERROR: init.pt not produced; aborting"; exit 1
fi
echo "init: $INIT_DIR/init.pt"

# ---- Step 2: train with --resume from the hull init ----
python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --run-dir "$RUN_DIR" \
    --resume "$INIT_DIR/init.pt" \
    --steps 300000 \
    --down 1 \
    --tnt-eval-dir "$TNT_EVAL" \
    --tnt-official-scene "$SCENE_NAME" \
    --tnt-official-frame "$TNT_OFFICIAL_FRAME" \
    --tnt-official-freq "$TNT_OFFICIAL_FREQ" \
    --tnt-official-res "$TNT_OFFICIAL_RES" \
    --tnt-official-n-samples "$TNT_OFFICIAL_N_SAMPLES" \
    --wandb 2>&1 | tee "$LOG"
