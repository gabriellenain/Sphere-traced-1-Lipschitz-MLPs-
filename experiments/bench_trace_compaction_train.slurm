#!/bin/bash
#SBATCH --job-name=bench-compact-train
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_train_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_train_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
# Same GPU class as training (A100/H100/H200) so the wall-clock numbers transfer.
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

PT=${1:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260522_171536_scan65_4908959/ckpt/checkpoint_step_290000.pt}
SCENE=${2:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan65}
VIEWS=${3:-0,16,32,48}
BS=${BS:-4096,16384}
KS=${KS:-64,128,256}
OUT=${OUT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bench_compact_train_${SLURM_JOB_ID}.png}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u analysis/bench_trace_compaction_train.py \
    --pt    "$PT" \
    --scene "$SCENE" \
    --views "$VIEWS" \
    --Bs    "$BS" \
    --Ks    "$KS" \
    --out   "$OUT"
