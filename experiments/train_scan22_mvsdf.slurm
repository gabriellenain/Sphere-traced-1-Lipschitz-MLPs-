#!/bin/bash
#SBATCH --job-name=1lip-scan122-mvsdf
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_scan122_mvsdf_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_scan122_mvsdf_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu004,gpu011,gpu014,gpu008

# Run precompute_mast3r_depths_scan22.slurm first to populate the depth dir.
# Usage: sbatch train_scan22_mvsdf.slurm

SCAN=122
SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan${SCAN}
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data

MVS_DEPTH_DIR=${SCENE}/mast3r_depth_1600x1200

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/train_scan122_mvsdf_${SLURM_JOB_ID}.log
echo "Logging to $LOG"
echo "scene:         $SCENE"
echo "mvs_depth_dir: $MVS_DEPTH_DIR"

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --input-encoding identity \
    --multires 3 \
    --trace-iters 24 \
    --init hull \
    --steps 100000 \
    --batch 4096 \
    --down 1 \
    --mvs-depth-dir "$MVS_DEPTH_DIR" \
    --mvsdf-schedule \
    --w-eikonal 0.1 \
    --dtu-eval-dir "$DTU_EVAL" \
    --wandb 2>&1 | tee "$LOG"
