#!/bin/bash
#SBATCH --job-name=buddha-sdf-vis
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/buddha_sdf_vis_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/buddha_sdf_vis_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:20:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python -u analysis/visualize_buddha_sdf.py
