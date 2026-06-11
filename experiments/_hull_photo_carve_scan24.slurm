#!/bin/bash
#SBATCH --job-name=s24-photocarve
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_photo_carve_scan24/slurm_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_photo_carve_scan24/slurm_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

set -euo pipefail
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
mkdir -p "$ROOT/outputs/hull_photo_carve_scan24"
cd "$ROOT"
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
python -u tools/_hull_photo_carve_scan24.py
