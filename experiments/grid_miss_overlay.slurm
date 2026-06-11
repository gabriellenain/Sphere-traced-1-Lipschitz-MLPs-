#!/bin/bash
#SBATCH --job-name=grid-miss
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grid_miss_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/grid_miss_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00

# Grid of the miss-overlay panel across views, using the run's REAL tracer
# (bracketing on, train-budget iters from config) — i.e. exactly how the run
# was trained. Usage:
#   sbatch grid_miss_overlay.slurm <run_dir> [ckpt] [n_views] [down] [max_iters] [out_dir]
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${1:?usage: sbatch grid_miss_overlay.slurm <run_dir> [ckpt] [n_views] [down] [max_iters] [out_dir]}"
CKPT="${2:-checkpoint_latest.pt}"
NVIEWS="${3:-16}"
DOWN="${4:-4}"
MAXIT="${5:-500}"
OUT="${6:-figs/sphere_trace_steps}"

[ -d "$RUN_DIR" ] || { echo "[error] run_dir not found: $RUN_DIR"; exit 1; }
mkdir -p "$OUT"
echo "run_dir: $RUN_DIR  ckpt: $CKPT  n_views: $NVIEWS  down: $DOWN  max: $MAXIT"

python -u analysis/grid_miss_overlay.py "$RUN_DIR" \
    --ckpt "$CKPT" --n-views "$NVIEWS" --down "$DOWN" --max-iters "$MAXIT" \
    --out-dir "$OUT"
