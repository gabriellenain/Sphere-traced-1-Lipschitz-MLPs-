#!/bin/bash
#SBATCH --job-name=1lip-render
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

PT=${1:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260430_105248_blender/checkpoint_best_photo.pt}

echo "Rendering checkpoint: $PT"

python -u -m lip_tracer.train \
    --no-train \
    --pt "$PT" \
    --blender
