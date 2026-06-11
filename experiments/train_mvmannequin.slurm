#!/bin/bash
#SBATCH --job-name=1lip-mvm
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-mvm_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-mvm_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# Usage: sbatch train_mvmannequin.slurm <scene>
#   e.g. sbatch train_mvmannequin.slurm kinette-cos-hx
SCENE_NAME=${1:-kinette-cos-hx}
SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/mvmannequin_neus/${SCENE_NAME}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs
TS=$(date +%Y%m%d_%H%M%S)
TAG=mvm_${SCENE_NAME}
RUN_DIR=$OUTPUTS/run_${TS}_${TAG}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/train.log
ln -sfn "$RUN_DIR" "$OUTPUTS/1lip-mvm_${SLURM_JOB_ID}.rundir"
mkdir -p "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/1lip-mvm_${SLURM_JOB_ID}.rundir"
echo "Logging to $LOG  (scene: $SCENE)"
echo "Run dir: $RUN_DIR"
if [[ ! -d "$SCENE" ]]; then
    echo "ERROR: missing scene directory: $SCENE"
    echo "Run: python tools/prepare_mvmannequin.py --scenes ${SCENE_NAME}"
    exit 1
fi

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then RESUME_ARGS=(--resume "$RESUME"); fi

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --run-dir "$RUN_DIR" \
    --init hull \
    --steps ${STEPS:-300000} \
    --batch ${BATCH:-16384} \
    --trace-iters ${TRACE_ITERS:-64} \
    --eps ${EPS:-1e-3} \
    --t-far ${T_FAR:-10.0} \
    --hull-res ${HULL_RES:-256} \
    --down ${DOWN:-1} \
    --dtu-official-freq ${OFFICIAL_FREQ:-10000} \
    --dtu-official-res ${OFFICIAL_RES:-384} \
    --dtu-official-bound ${OFFICIAL_BOUND:-1.0} \
    "${RESUME_ARGS[@]}" \
    --wandb 2>&1 | tee "$LOG"
