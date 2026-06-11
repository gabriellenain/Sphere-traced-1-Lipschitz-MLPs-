#!/bin/bash
#SBATCH --job-name=hull-barn
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull-barn_%j.slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull-barn_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# Usage: sbatch visual_hull_barn.slurm

SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt/Barn
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

OUT_DIR=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_Barn_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/log.txt"

if [[ ! -f "$SCENE/mask/_overview.png" ]]; then
    python -u tools/sky_segment.py --scene "$SCENE" 2>&1 | tee -a "$LOG"
else
    echo "[seg] reusing existing masks in $SCENE/mask" | tee -a "$LOG"
fi

python -u tools/visualize_visual_hull.py \
    --scene "$SCENE" \
    --out   "$OUT_DIR/hull.png" \
    --res 192 --bound 1.0 --down 2 --view 0 2>&1 | tee -a "$LOG"
