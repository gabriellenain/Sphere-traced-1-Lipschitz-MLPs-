#!/bin/bash
#SBATCH --job-name=1lip-skull-new
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-skull-new_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-skull-new_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --nodelist=gpu007,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SKULL_NEW=/scratch/_projets_/willow/1-lip-tracer-new/data/skull/scan65

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_skull_new_${SLURM_JOB_ID}.log
echo "Logging to $LOG"
echo "Using new venv + scene: $SKULL_NEW"
if [[ ! -d "$SKULL_NEW" ]]; then
    echo "ERROR: missing healthy skull scene directory: $SKULL_NEW"
    echo "Stage/copy/download skull data onto healthy OSTs before submitting this job."
    exit 1
fi

RESUME_ARGS=()
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    RESUME_ARGS=(--resume "$RESUME")
    echo "Resuming from $RESUME (explicit)"
else
    LAST_RUN=$(ls -td outputs/run_*_scan65 2>/dev/null | head -1)
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
    --dataset skull \
    --scene "$SKULL_NEW" \
    --init hull \
    --steps 100000 \
    --batch 4096 \
    --down 1 \
    "${RESUME_ARGS[@]}" \
    --wandb 2>&1 | tee "$LOG"
