#!/bin/bash
#SBATCH --job-name=phong-grid
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/phong_grid_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/phong_grid_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-outputs/run_20260602_150702_scan40_4951716/ckpt/checkpoint_step_020000.pt}
python -u analysis/_phong_grid_all_views.py --ckpt "$CKPT" --down 2 --ss 2
