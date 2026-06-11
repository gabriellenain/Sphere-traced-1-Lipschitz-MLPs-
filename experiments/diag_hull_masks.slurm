#!/bin/bash
#SBATCH --job-name=hull-diag
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull-diag_%j.slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull-diag_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt/Barn
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

OUT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_diag_Barn_${SLURM_JOB_ID}
mkdir -p "$OUT"

# Force regenerate sky masks (Grounded-SAM had overwritten them).
rm -f "$SCENE/mask"/*.png "$SCENE/mask/_overview.png"
python -u tools/sky_segment.py --scene "$SCENE" 2>&1 | tee "$OUT/log.txt"

# Run the diagnostic
python -u tools/diag_hull_masks.py \
    --scene "$SCENE" \
    --out-dir "$OUT" \
    --res 160 --bound 1.5 --down 2 --fit-steps 1500 --n-mask-samples 8 2>&1 | tee -a "$OUT/log.txt"
