#!/bin/bash
#SBATCH --job-name=dragon-sdf-quick
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dragon_sdf_quick_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dragon_sdf_quick_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python -u fit_gt_sdf.py \
  --mesh data/gt_meshes/dragon_watertight_norm.ply \
  --out-dir outputs/gt_sdf_fit_dragon_quick \
  --steps 1000 \
  --batch 4096 \
  --bound 1.5 \
  --near-std 0.02 \
  --n-near 20000 \
  --n-vol 20000 \
  --sdf-chunk 5000 \
  --mc-res 128
