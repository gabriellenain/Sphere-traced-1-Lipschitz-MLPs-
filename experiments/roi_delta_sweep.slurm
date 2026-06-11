#!/bin/bash
#SBATCH --job-name=roi-sweep
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/roi_sweep_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/roi_sweep_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

RUN=${RUN:-outputs/run_20260517_114302_scan122}
OUT=${OUT:-artifacts/roi_delta_sweep_scan122_v16}

python analysis/roi_delta_sweep.py \
    --run "$RUN" \
    --view 16 \
    --x0 600 --x1 950 --y0 200 --y1 430 \
    --stride 1 --chunk 4096 \
    --out "$OUT"
