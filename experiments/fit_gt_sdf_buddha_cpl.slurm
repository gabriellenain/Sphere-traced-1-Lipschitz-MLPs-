#!/bin/bash
#SBATCH --job-name=buddha-cpl
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/buddha_cpl_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/buddha_cpl_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

# Normalise the raw Stanford scan: centre, scale to [-1,1], swap Y↔Z (Z-up)
python -c "
import numpy as np, trimesh
raw = trimesh.load('artifacts/happy_recon.ply', process=False)
v = np.asarray(raw.vertices, np.float32)
v -= v.mean(0); v /= np.abs(v).max()
v = v[:, [0, 2, 1]]   # Stanford Y-up → Z-up
trimesh.Trimesh(vertices=v, faces=raw.faces, process=True).export('data/gt_meshes/happy_buddha_norm.ply')
print('saved data/gt_meshes/happy_buddha_norm.ply')
"

python -u fit_gt_sdf.py \
  --mesh data/gt_meshes/happy_buddha_norm.ply \
  --out-dir outputs/gt_sdf_fit_buddha_cpl \
  --architecture cpl \
  --input-encoding pe \
  --multires 0 \
  --hidden 256 \
  --depth 8 \
  --steps 50000 \
  --batch 8192 \
  --bound 1.1 \
  --near-std 0.01 \
  --n-near 300000 \
  --n-vol 100000 \
  --sdf-chunk 20000 \
  --mc-res 512
