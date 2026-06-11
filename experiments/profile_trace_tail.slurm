#!/bin/bash
#SBATCH --job-name=trace-tail
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/profile_trace_tail_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/profile_trace_tail_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu004

PT=${1:?pt path required}
SCENE=${2:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan65}
VIEWS=${3:-16,32}
ITERS=${4:-256}
DOWN=${5:-2}
OUT=${6:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/trace_tail_${SLURM_JOB_ID}.png}

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u analysis/profile_trace_tail.py \
    --pt    "$PT" \
    --scene "$SCENE" \
    --views "$VIEWS" \
    --iters "$ITERS" \
    --down  "$DOWN" \
    --out   "$OUT"
