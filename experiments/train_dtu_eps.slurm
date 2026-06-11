#!/bin/bash
#SBATCH --job-name=1lip-dtu-eps
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-dtu-eps_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-dtu-eps_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# Usage: sbatch train_dtu.slurm <scan_number>
#   e.g. sbatch train_dtu.slurm 122
SCAN=${1:-122}
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}
OLD_SCENE=${OLD_SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/DTU/scan${SCAN}}
if [[ ! -f "$OLD_SCENE/sparse_sfm_points.txt" ]]; then
    OLD_SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan${SCAN}
fi
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS\ Data
SFM_MIN_VIEWS=${SFM_MIN_VIEWS:-30}
DTU_OFFICIAL_FREQ=${DTU_OFFICIAL_FREQ:-10000}
DTU_OFFICIAL_RES=${DTU_OFFICIAL_RES:-384}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Compute run_dir up front so the train log lives inside it (matches the
# layout train.py would otherwise auto-generate: run_<ts>_<scene-name>).
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs
TS=$(date +%Y%m%d_%H%M%S)
TAG=$(basename "$SCENE")
RUN_DIR=$OUTPUTS/run_${TS}_${TAG}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/train.log
ln -sfn "$RUN_DIR" "$OUTPUTS/1lip-dtu_${SLURM_JOB_ID}.rundir"
# Mirror the run dir into the repo's /home/.../outputs/ as a symlink so it
# shows up in the IDE alongside older runs (data still lives on /scratch).
mkdir -p "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/1lip-dtu_${SLURM_JOB_ID}.rundir"
echo "Logging to $LOG  (scene: $SCENE)"
echo "Run dir: $RUN_DIR"
echo "Using new venv + DTU eval: $DTU_EVAL"
echo "Architecture: $ARCHITECTURE"
echo "COLMAP supervision: W_SFM=$W_SFM  SFM_MIN_VIEWS=$SFM_MIN_VIEWS"
echo "Stability: LR=$LR  W_EIKONAL=$W_EIKONAL  W_SIL=$W_SIL"
echo "Official DTU eval: every ${DTU_OFFICIAL_FREQ} steps  res=${DTU_OFFICIAL_RES}"
if [[ ! -d "$SCENE" ]]; then
    echo "ERROR: missing healthy DTU scene directory: $SCENE"
    echo "Run: bash download_dtu_idr.sh $SCAN"
    exit 1
fi
if [[ ! -d "$DTU_EVAL" ]]; then
    echo "ERROR: missing healthy DTU eval directory: $DTU_EVAL"
    echo "Run: bash download_dtu_eval.sh"
    exit 1
fi
if [[ "$W_SFM" != "0" && "$W_SFM" != "0.0" && ! -f "$SCENE/sparse_sfm_points.txt" ]]; then
    echo "COLMAP files missing in $SCENE; trying to precompute from $OLD_SCENE"
    if [[ -f "$OLD_SCENE/sparse_sfm_points.txt" ]]; then
        python tools/precompute_sfm_pairs_idr.py \
            --scene "$SCENE" \
            --old-scene "$OLD_SCENE"
    else
        echo "ERROR: requested W_SFM=$W_SFM, but no COLMAP points found:"
        echo "  missing: $SCENE/sparse_sfm_points.txt"
        echo "  missing: $OLD_SCENE/sparse_sfm_points.txt"
        echo "Generate/copy sparse_sfm_points.txt + sfm_pairs.json first, or launch with W_SFM=0."
        exit 1
    fi
fi

FEATURE_MAPS=outputs/scan${SCAN}_mast3r_features_idr.pt
FEATURE_ARGS=()
if [[ "$W_FEATURE" != "0" && "$W_FEATURE" != "0.0" ]]; then
    FEATURE_ARGS=(--feature-maps "$FEATURE_MAPS")
fi

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then RESUME_ARGS=(--resume "$RESUME"); fi

HULL_ARGS=()
if [[ "$HULL_SFM_ROI" == "1" ]]; then HULL_ARGS=(--hull-sfm-roi); fi

DEBUG_REGIONS_ARGS=()

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --run-dir "$RUN_DIR" \
    --init hull \
    --steps ${STEPS:-300000} \
    --batch ${BATCH:-16384} \
    --trace-iters ${TRACE_ITERS:-64} \
    --eps ${EPS:-0.001} \
    --down 1 \
    --dtu-eval-dir "$DTU_EVAL" \
    --dtu-official-freq "$DTU_OFFICIAL_FREQ" \
    --dtu-official-res "$DTU_OFFICIAL_RES" \
    --sfm-min-views "$SFM_MIN_VIEWS" \
    "${FEATURE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${HULL_ARGS[@]}" \
    "${DEBUG_REGIONS_ARGS[@]}" \
    --wandb 2>&1 | tee "$LOG"
