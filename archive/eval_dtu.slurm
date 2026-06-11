#!/bin/bash
#SBATCH --job-name=1lip-eval
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/eval_dtu_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/eval_dtu_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

PT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260424_142902_dtu/checkpoint_best_photo.pt
SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m lip_tracer.train \
    --no-train \
    --pt       "$PT" \
    --scene    "$SCENE" \
    --dtu-eval-dir "$DTU_EVAL" \
    --mc-res 512
