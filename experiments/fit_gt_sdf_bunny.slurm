#!/bin/bash
#SBATCH --job-name=bunny-neus-m6
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bunny_neus_m6_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/bunny_neus_m6_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python -u fit_gt_sdf.py \
  --mesh data/gt_meshes/bunny2_faces.ply \
  --out-dir outputs/gt_sdf_fit_bunny_neus_m6 \
  --architecture neus \
  --input-encoding pe \
  --multires 6 \
  --hidden 256 \
  --depth 8 \
  --steps 50000 \
  --batch 8192 \
  --bound 0.9 \
  --near-std 0.01 \
  --n-near 300000 \
  --n-vol 100000 \
  --sdf-chunk 20000 \
  --mc-res 512
