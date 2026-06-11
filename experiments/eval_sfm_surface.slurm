#!/bin/bash
#SBATCH --job-name=sfm-surf
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sfm_surface_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sfm_surface_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --exclude=gpu001,gpu002,gpu004,gpu011,gpu014

# Usage:
#   sbatch eval_sfm_surface.slurm [checkpoint.pt] [scan] [res] [bound] [architecture]
#
# Example:
#   sbatch eval_sfm_surface.slurm outputs/run_20260504_195013_scan122/checkpoint_best_photo.pt 122 256 1.5 cpl

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

PT=${1:-outputs/run_20260504_195013_scan122/checkpoint_best_photo.pt}
SCAN=${2:-122}
RES=${3:-256}
BOUND=${4:-1.5}
ARCH=${5:-cpl}
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}
LOG=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sfm_surface_${SLURM_JOB_ID}.log

echo "Logging to $LOG"
echo "Checkpoint: $PT"
echo "Scene:      $SCENE"
echo "Arch:       $ARCH"
echo "MC res:     $RES"
echo "Bound:      $BOUND"
echo "Node:       $(hostname)"

if [[ ! -f "$PT" ]]; then
    echo "ERROR: missing checkpoint: $PT"
    exit 1
fi
if [[ ! -d "$SCENE" ]]; then
    echo "ERROR: missing DTU scene directory: $SCENE"
    exit 1
fi

python -u analysis/eval_sfm_surface.py \
    --pt "$PT" \
    --scene "$SCENE" \
    --architecture "$ARCH" \
    --bound "$BOUND" \
    --res "$RES" \
    --device cuda \
    2>&1 | tee "$LOG"
