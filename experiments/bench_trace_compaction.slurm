#!/bin/bash
#SBATCH --job-name=bench-compact
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
# Pin to A100/H100/H200 only — that's what the paper's training uses, so the
# bench needs to report wall-clock on the same class of card.
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

PT=${1:?pt path required}
SCENE=${2:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan65}
VIEWS=${3:-16,32}
DOWN=${4:-2}
KS=${5:-24,64,128,256}
OUT=${6:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_${SLURM_JOB_ID}.png}

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u analysis/bench_trace_compaction.py \
    --pt    "$PT" \
    --scene "$SCENE" \
    --views "$VIEWS" \
    --down  "$DOWN" \
    --Ks    "$KS" \
    --out   "$OUT"
