#!/bin/bash
#SBATCH --job-name=1lip-scan122-normal
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_scan122_normal_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_scan122_normal_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu004,gpu011,gpu014,gpu008

# Uses dtu/scan122 (SDFStudio format with meta_data.json) so that
# per-view normal .npy files are loaded for w_normal supervision.
# dtu_idr/scan122 has no meta_data.json so normals can't be loaded there.

SCAN=122
SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan${SCAN}
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_scan122_normal_${SLURM_JOB_ID}.log
echo "Logging to $LOG"
echo "scene: $SCENE"

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --input-encoding pe \
    --multires 6 \
    --trace-iters 24 \
    --init hull \
    --steps 100000 \
    --batch 4096 \
    --down 1 \
    --w-eikonal 0.1 \
    --w-normal 0.05 \
    --dtu-eval-dir "$DTU_EVAL" \
    --wandb 2>&1 | tee "$LOG"
