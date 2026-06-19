#!/bin/bash
#SBATCH --job-name=st-diag-gpu
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_diag_gpu_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_diag_gpu_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate

RUN_DIR="${1:?usage: sbatch st_diag_gpu.slurm <run_dir> [view] [ckpt]}"
VIEW="${2:-50}"
CKPT="${3:-checkpoint_final.pt}"
ALIAS_SS="${ALIAS_SS:-1,8}"
MAX_ITERS="${MAX_ITERS:-256}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT_DIR="$RUN_DIR/st_buffers_view${VIEW}_it${MAX_ITERS}_gpu"
mkdir -p "$OUT_DIR"

echo "run_dir: $RUN_DIR"
echo "view=$VIEW  ckpt=$CKPT  alias_ss=$ALIAS_SS  iters=$MAX_ITERS  threads=$OMP_NUM_THREADS"
echo "gpu: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ALIAS GPU sweep ss=$ALIAS_SS iters=$MAX_ITERS -> $OUT_DIR/view${VIEW}_*"

python analysis/sphere_traced_screen_mesh.py \
    "$RUN_DIR" \
    --ckpt "$CKPT" \
    --view "$VIEW" --alias-diag --alias-ss "$ALIAS_SS" --max-iters "$MAX_ITERS" \
    --out "$OUT_DIR/view${VIEW}.ply" \
    --device cuda
