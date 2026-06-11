#!/bin/bash
#SBATCH --job-name=dragon-mesh
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dragon_mesh_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dragon_mesh_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

INPUT=${INPUT:-data/gt_meshes/dragon_recon/dragon_vrip_res2.ply}
OUT=${OUT:-data/gt_meshes/dragon_watertight_norm.ply}
DEPTH=${DEPTH:-8}
SCALE=${SCALE:-1.05}
DENSITY_QUANTILE=${DENSITY_QUANTILE:-0.01}
BBOX_MARGIN=${BBOX_MARGIN:-0.03}
TARGET_SIZE=${TARGET_SIZE:-1.2}

echo "input: $INPUT"
echo "out:   $OUT"
echo "depth: $DEPTH"
echo "cpus:  $SLURM_CPUS_PER_TASK"

NO_CROP=${NO_CROP:-0}
NO_CROP_FLAG=""
[ "$NO_CROP" = "1" ] && NO_CROP_FLAG="--no-crop"

python -u tools/make_dragon_watertight.py \
    --input "$INPUT" \
    --out "$OUT" \
    --depth "$DEPTH" \
    --scale "$SCALE" \
    --density-quantile "$DENSITY_QUANTILE" \
    --bbox-margin "$BBOX_MARGIN" \
    --target-size "$TARGET_SIZE" \
    $NO_CROP_FLAG
