#!/bin/bash
#SBATCH --job-name=tnt-sanity
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/tnt_sanity_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/tnt_sanity_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

# IMPORTANT: this venv has open3d (0.19.0); the 1-lip-tracer-new venv does NOT.
source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCENE=${SCENE:-Barn}
SCENE_DIR=${SCENE_DIR:-data/tnt/${SCENE}}
GT_DIR=${GT_DIR:-data/tnt_gt/${SCENE}}
SANITY_PLY=${SANITY_PLY:-outputs/sanity_tnt_gt_in_colmap_frame.ply}
OUT=${OUT:-outputs/tnt_sanity_${SLURM_JOB_ID}}

echo "scene:   $SCENE"
echo "gt_dir:  $GT_DIR"
echo "out:     $OUT"

mkdir -p "$OUT"

# 1. GT -> COLMAP-SfM frame (inv of _trans.txt)
python -u analysis/tnt_sanity_gt.py \
    --gt-dir "$GT_DIR" \
    --scene  "$SCENE" \
    --out    "$SANITY_PLY"

# 2. eval with --frame colmap-sfm re-applies _trans.txt -> GT lands on GT -> F~1
python -u analysis/eval_tnt_official.py \
    --mesh      "$SANITY_PLY" \
    --scene     "$SCENE" \
    --scene-dir "$SCENE_DIR" \
    --gt-dir    "$GT_DIR" \
    --frame     colmap-sfm \
    --out       "$OUT"

LOGFILE="/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/tnt_sanity_${SLURM_JOB_ID}.log"
[ -f "$LOGFILE" ] && cp "$LOGFILE" "$OUT/job.log"
