#!/bin/bash
#SBATCH --job-name=happy-buddha
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/happy_buddha_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/happy_buddha_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python analysis/happy_buddha_mc.py
