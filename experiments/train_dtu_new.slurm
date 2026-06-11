#!/bin/bash
#SBATCH --job-name=1lip-dtu-new
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-dtu-new_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-dtu-new_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

# Usage: sbatch train_dtu_new.slurm <scan_number>
#   e.g. sbatch train_dtu_new.slurm 122
SCAN=${1:-122}
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS\ Data

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_dtu${SCAN}_new_${SLURM_JOB_ID}.log
echo "Logging to $LOG  (scene: $SCENE)"
echo "Using new venv + DTU eval: $DTU_EVAL"
if [[ ! -d "$SCENE" ]]; then
    echo "ERROR: missing healthy DTU scene directory: $SCENE"
    echo "Stage/copy/download DTU scan${SCAN} onto healthy OSTs before submitting this job."
    exit 1
fi
if [[ ! -d "$DTU_EVAL" ]]; then
    echo "ERROR: missing healthy DTU eval directory: $DTU_EVAL"
    echo "Stage/copy/download dtu_eval onto healthy OSTs before submitting this job."
    exit 1
fi

FEATURE_MAPS=outputs/scan${SCAN}_mast3r_features_idr.pt
FEATURE_ARGS=()
if [[ "$W_FEATURE" != "0" && "$W_FEATURE" != "0.0" ]]; then
    FEATURE_ARGS=(--feature-maps "$FEATURE_MAPS")
fi

RESUME_ARGS=()
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    RESUME_ARGS=(--resume "$RESUME")
    echo "Resuming from $RESUME (explicit)"
else
    LAST_RUN=$(ls -td outputs/run_*_scan${SCAN} 2>/dev/null | head -1)
    if [[ -n "$LAST_RUN" ]]; then
        for name in checkpoint.pt checkpoint_best_photo.pt checkpoint_best_geo.pt; do
            if [[ -f "$LAST_RUN/$name" ]]; then
                RESUME_ARGS=(--resume "$LAST_RUN/$name")
                echo "Resuming from $LAST_RUN/$name"
                break
            fi
        done
    fi
fi

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --input-encoding pe \
    --multires 6 \
    --trace-iters 24 \
    --init hull \
    --steps 100000 \
    --batch 4096 \
    --down 1 \
    --dtu-eval-dir "$DTU_EVAL" \
    "${FEATURE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --wandb 2>&1 | tee "$LOG"
