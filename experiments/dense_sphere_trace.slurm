#!/bin/bash
#SBATCH --job-name=dense-st
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dense_sphere_trace_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dense_sphere_trace_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

RUN_DIR="${1:?usage: sbatch dense_sphere_trace.slurm <run_dir> [n_views] [res]}"
N_VIEWS="${2:-120}"
RES="${3:-384}"
OUT="${RUN_DIR}/ckpt/sphere_traced_dense.ply"

python analysis/dense_sphere_trace.py "$RUN_DIR" \
    --n-views "$N_VIEWS" \
    --res "$RES" \
    --voxel-size 0.005 \
    --grazing-cos 0.1 \
    --out "$OUT"
