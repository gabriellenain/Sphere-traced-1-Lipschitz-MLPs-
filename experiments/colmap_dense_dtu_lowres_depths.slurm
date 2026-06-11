#!/bin/bash
#SBATCH --job-name=colmap-dtu-lowres-depth
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_dtu_lowres_depth_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_dtu_lowres_depth_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Calibrated low-resolution COLMAP geometric depth maps for IDR-format DTU scans.
# Depth diagnostic only: stops after PatchMatch; no fusion, no Poisson meshing.
# Usage: MAX_IMAGE_SIZE=800 N_SRC=5 sbatch colmap_dense_dtu_lowres_depths.slurm 40

set -euo pipefail

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCAN=${1:?usage: sbatch colmap_dense_dtu_lowres_depths.slurm <scan_number>}
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}}
MAX_IMAGE_SIZE=${MAX_IMAGE_SIZE:-800}
N_SRC=${N_SRC:-5}
DEPTH_MIN=${DEPTH_MIN:-0.5}
DEPTH_MAX=${DEPTH_MAX:-3.5}
WS=${WS:-$ROOT/outputs/colmap_dtu_scan${SCAN}_lowres${MAX_IMAGE_SIZE}_${SLURM_JOB_ID}}
MODEL=$WS/input_model

mkdir -p "$WS" "$ROOT/outputs/logs"
test -f "$SCENE/cameras.npz"
test -d "$SCENE/image"

echo "[$(date)] scan=$SCAN scene=$SCENE workspace=$WS max_image_size=$MAX_IMAGE_SIZE n_src=$N_SRC depth=$DEPTH_MIN..$DEPTH_MAX"
echo "[$(date)] colmap version:"
colmap help 2>&1 | head -1

echo "[$(date)] stage calibrated input model"
python tools/stage_colmap_dense_dtu.py \
    --scene "$SCENE" \
    --model "$MODEL"

echo "[$(date)] image_undistorter low-res"
colmap image_undistorter \
    --image_path  "$SCENE/image" \
    --input_path  "$MODEL" \
    --output_path "$WS" \
    --output_type COLMAP \
    --max_image_size "$MAX_IMAGE_SIZE"

echo "[$(date)] write explicit PatchMatch source views"
python tools/stage_colmap_dense_dtu.py \
    --scene "$SCENE" \
    --model "$MODEL" \
    --source-config "$WS/stereo/patch-match.cfg" \
    --n-src "$N_SRC"

echo "[$(date)] patch_match_stereo geometric"
colmap patch_match_stereo \
    --workspace_path "$WS" \
    --workspace_format COLMAP \
    --PatchMatchStereo.cache_size 24 \
    --PatchMatchStereo.depth_min "$DEPTH_MIN" \
    --PatchMatchStereo.depth_max "$DEPTH_MAX" \
    --PatchMatchStereo.geom_consistency true

echo "[$(date)] done depth maps"
find "$WS/stereo/depth_maps" -maxdepth 1 -name '*.geometric.bin' | wc -l
