#!/bin/bash
#SBATCH --job-name=render-gt-match
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_gt_match_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_gt_match_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --exclude=gpu004

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

/home/glenain/blender/blender -b -P tools/render_neus_match_gt.py -- \
  --mesh scan122_paper_like_clean_largest_512.ply \
  --cameras baselines/NeuS/public_data/scan122/cameras_sphere.npz \
  --view 50 --width 1600 --height 1200 --samples 64 \
  --out neus_scan122_v50_render.png
