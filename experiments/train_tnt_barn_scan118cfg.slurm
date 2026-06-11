#!/bin/bash
#SBATCH --job-name=1lip-barn-s118
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-barn-s118_%j.slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-barn-s118_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
#
# T&T Barn with the EXACT scan118 config (run_20260602_134230_scan118_4951319):
# DTU-style in-loop visual-hull init (--init hull --hull-border-aware), NCC-only
# objective, identical model/trace/train hyperparams. The only scan118 flags
# dropped are DTU-specific: --dtu-eval-dir / --dtu-official-* (no DTU GT for
# Barn) and --feature-maps (w_feature=0 there, so it was a no-op). Foreground
# = the existing not-sky masks in data/tnt/Barn/mask (model-generated), so the
# hull + fg-sampling target is "barn + ground", matching the TnT crop/GT.

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
RUN_DIR=$OUTPUTS/run_${TS}_tnt_${SCENE_NAME}_scan118cfg_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$OUTPUTS/1lip-barn-s118_${SLURM_JOB_ID}.rundir"
LOG=$RUN_DIR/train.log
echo "Logging to $LOG  (scene: $SCENE)"
echo "Official TnT eval: every ${TNT_OFFICIAL_FREQ} steps  res=${TNT_OFFICIAL_RES}  n=${TNT_OFFICIAL_N_SAMPLES}  frame=${TNT_OFFICIAL_FRAME}"

if [[ ! -d "$SCENE" ]]; then
    echo "ERROR: missing T&T scene: $SCENE"; exit 1
fi
if [[ ! -f "$TNT_EVAL/${SCENE_NAME}/${SCENE_NAME}.ply" ]]; then
    echo "ERROR: missing official TnT GT assets under $TNT_EVAL/${SCENE_NAME}"
    exit 1
fi

# Flags mirror scan118's command exactly, minus the DTU-only ones.
python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --run-dir "$RUN_DIR" \
    --init hull \
    --steps 300000 \
    --batch 4096 \
    --trace-iters 64 \
    --down 1 \
    --hull-border-aware \
    --ncc-color gray \
    --sfm-min-views 30 \
    --w-sfm 0 \
    --w-geo-sdf 0 \
    --tnt-eval-dir "$TNT_EVAL" \
    --tnt-official-scene "$SCENE_NAME" \
    --tnt-official-frame "$TNT_OFFICIAL_FRAME" \
    --tnt-official-freq "$TNT_OFFICIAL_FREQ" \
    --tnt-official-res "$TNT_OFFICIAL_RES" \
    --tnt-official-n-samples "$TNT_OFFICIAL_N_SAMPLES" \
    --wandb 2>&1 | tee "$LOG"
