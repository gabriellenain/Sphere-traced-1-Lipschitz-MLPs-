#!/bin/bash
#SBATCH --job-name=sfm-pairs-idr
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sfm_pairs_idr_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu_homogen
#SBATCH --time=00:10:00

# Usage: sbatch precompute_sfm_pairs_idr.slurm <scan_number>
SCAN=${1:-122}
IDR_SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}
OLD_SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan${SCAN}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python tools/precompute_sfm_pairs_idr.py \
    --scene "$IDR_SCENE" \
    --old-scene "$OLD_SCENE"
