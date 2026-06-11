#!/bin/bash
#SBATCH --job-name=viz-sfm-lego
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_sfm_pts_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/viz_sfm_pts_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=00:10:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOB_ID

python -u tools/viz_sfm_pts.py
