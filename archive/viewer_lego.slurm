#!/bin/bash
#SBATCH --job-name=1lip-viewer
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viewer_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viewer_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --exclude=gpu004
#SBATCH --time=2:00:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

PT=${1:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260430_105248_blender/checkpoint_best_photo.pt}

echo "Loading checkpoint: $PT"
echo "Viser will be available via port forwarding: ssh -L 8080:NODE:8080 cluster"
echo "Node: $(hostname)"

python -u -m lip_tracer.train \
    --no-train \
    --pt "$PT" \
    --blender \
    --viewer \
    --viewer-res 512 \
    --mc-res 512 \
    --render-down 1
