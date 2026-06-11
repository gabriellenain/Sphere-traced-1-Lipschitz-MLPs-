#!/bin/bash
#SBATCH --job-name=bench-compact-B
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_B_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_B_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
# Pin to A100/H100/H200 only — match the original bench (4910896, H100 NVL).
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

PT=${1:?pt path required}
SCENE=${2:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan65}
VIEWS=${3:-16,32}
DOWN=${4:-2}
BS=${5:-8192,16384,32768,65536,131072,262144,524288}
KS=${6:-24,64,128,256}
OUT=${7:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_B_${SLURM_JOB_ID}.png}

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u analysis/bench_compact_B_sweep.py \
    --pt    "$PT" \
    --scene "$SCENE" \
    --views "$VIEWS" \
    --down  "$DOWN" \
    --Bs    "$BS" \
    --Ks    "$KS" \
    --out   "$OUT"
