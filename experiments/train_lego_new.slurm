#!/bin/bash
#SBATCH --job-name=1lip-lego-new
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-lego-new_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-lego-new_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=6:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LEGO_NEW=/scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_lego_new_${SLURM_JOB_ID}.log
echo "Logging to $LOG"
echo "Using new venv + scene: $LEGO_NEW"

RESUME_ARGS=()
CONFIG_ARGS=()
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
    RESUME_ARGS=(--resume "$RESUME")
    echo "Resuming from $RESUME (explicit)"
    RESUME_CONFIG="$(dirname "$RESUME")/config.json"
    if [[ -f "$RESUME_CONFIG" ]]; then
        CONFIG_ARGS=(--config "$RESUME_CONFIG")
        echo "Using resume config $RESUME_CONFIG"
    else
        echo "ERROR: no config.json next to $RESUME"
        exit 2
    fi
else
    LAST_RUN=$(ls -td outputs/run_*_lego 2>/dev/null | head -1)
    if [[ -n "$LAST_RUN" ]]; then
        for name in checkpoint.pt checkpoint_best_photo.pt checkpoint_best_geo.pt; do
            if [[ -f "$LAST_RUN/$name" ]]; then
                RESUME_ARGS=(--resume "$LAST_RUN/$name")
                echo "Resuming from $LAST_RUN/$name"
                if [[ -f "$LAST_RUN/config.json" ]]; then
                    CONFIG_ARGS=(--config "$LAST_RUN/config.json")
                    echo "Using resume config $LAST_RUN/config.json"
                else
                    echo "ERROR: no config.json in $LAST_RUN"
                    exit 2
                fi
                break
            fi
        done
    fi
fi

if [[ ${#CONFIG_ARGS[@]} -gt 0 ]]; then
    TRAIN_ARGS=("${CONFIG_ARGS[@]}" --scene "$LEGO_NEW" "${RESUME_ARGS[@]}")
else
    TRAIN_ARGS=(
        --dataset lego
        --scene "$LEGO_NEW"
        --steps 100000
        --batch 4096
        --down 1
        --w-eikonal 0.1
        --w-photo 1.0
        --w-mask-fg 0.1
        --w-mask-bg 0.1
        --sil-s 50
        --sil-s-interval 6000
        --sil-s-max-mults 5
        "${RESUME_ARGS[@]}"
    )
fi

python -u -m lip_tracer.train "${TRAIN_ARGS[@]}" --wandb 2>&1 | tee "$LOG"
