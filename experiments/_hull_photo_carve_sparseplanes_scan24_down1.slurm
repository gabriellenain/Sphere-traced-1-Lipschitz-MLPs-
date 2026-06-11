#!/bin/bash
#SBATCH --job-name=s24-spar-d1
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_photo_carve_sparseplanes_scan24_down1/slurm_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_photo_carve_sparseplanes_scan24_down1/slurm_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

set -euo pipefail
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export DOWN=1
export PATCH=5
export OUT_DIR=$ROOT/outputs/hull_photo_carve_sparseplanes_scan24_down1
mkdir -p "$OUT_DIR"
cd "$ROOT"
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
python -u tools/_hull_photo_carve_sparseplanes_scan24.py
