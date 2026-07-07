#!/bin/bash
#SBATCH --job-name=zncc-viz
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/zncc_viz_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/zncc_viz_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python tools/viz_zncc_surface.py \
    --pt outputs/run_20260614_115440_truck_gsam_mvsformer_carved_init_nomask_abstau0_geosdf1.0_5018512/ckpt/checkpoint_latest.pt \
    --scene /home/glenain/Sphere-traced-1-Lipschitz-MLPs-/_diagnostics/truck_gsam_scene \
    --views 0 50 100 150 200 242 \
    --down 1 \
    --batch 8192 \
    --out outputs/zncc_viz_truck
