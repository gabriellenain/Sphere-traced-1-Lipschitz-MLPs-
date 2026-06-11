#!/bin/bash
#SBATCH --job-name=hull-gsam
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull-gsam_%j.slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull-gsam_%j.slurm.log
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

OUT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/fit_hull_gsam_Barn_${SLURM_JOB_ID}
mkdir -p "$OUT"
LOG="$OUT/log.txt"

# overwrite previous (sky-only) masks with Grounded-SAM "a barn." masks
python -u tools/grounded_sam_segment.py \
    --scene "$SCENE" --prompt "the main barn building." 2>&1 | tee -a "$LOG"

# carve hull + fit_to_hull + Phong render
python -u tools/fit_hull_barn.py \
    --scene "$SCENE" \
    --out-dir "$OUT" \
    --res 192 --bound 1.5 --down 2 --steps 2000 2>&1 | tee -a "$LOG"
