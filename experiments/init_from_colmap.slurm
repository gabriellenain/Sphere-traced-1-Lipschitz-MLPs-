#!/bin/bash
#SBATCH --job-name=colmap-init
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/colmap-init_%j.slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/colmap-init_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# Usage: sbatch init_from_colmap.slurm <SceneName>

SCENE_NAME=${1:-Barn}
SCENE=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt/${SCENE_NAME}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

OUT_DIR=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/colmap_init_${SCENE_NAME}_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"

export PATH=/home/glenain/.pixi/bin:$PATH

# NO_IGR=1 → only triangulate + write <scene>/sparse_sfm_points.txt (the SfM ROI
# cloud the MVSFormer sphere-carve consumes); skip the IGR pretrain + PNG.
IGR_ARGS=(--ckpt "$OUT_DIR/init.pt" --steps 5000)
[ "${NO_IGR:-0}" = "1" ] && IGR_ARGS=(--no-igr)

python -u tools/init_from_colmap.py \
    --scene "$SCENE" \
    --work  /home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt/${SCENE_NAME}/colmap \
    --out-png "$OUT_DIR/init.png" \
    "${IGR_ARGS[@]}" 2>&1 | tee "$OUT_DIR/log.txt"
