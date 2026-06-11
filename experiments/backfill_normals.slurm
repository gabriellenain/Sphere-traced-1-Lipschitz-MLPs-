#!/bin/bash
#SBATCH --job-name=backfill-norm
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/backfill_normals_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/backfill_normals_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

RUN=${RUN:-outputs/run_20260516_214430_scan65}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan65}

python -u analysis/_backfill_normals.py "$RUN" "$SCENE"
