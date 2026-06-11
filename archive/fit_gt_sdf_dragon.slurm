#!/bin/bash
#SBATCH --job-name=dragon-gt-sdf
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dragon_gt_sdf_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dragon_gt_sdf_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=08:00:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python -u fit_gt_sdf.py \
  --mesh data/gt_meshes/dragon_watertight_norm.ply \
  --out-dir outputs/gt_sdf_fit_dragon \
  --steps 10000 \
  --batch 8192 \
  --bound 1.5 \
  --near-std 0.02 \
  --n-near 200000 \
  --n-vol 200000 \
  --mc-res 256
