#!/bin/bash
#SBATCH --job-name=colmap-dtu
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_dtu_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_dtu_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Calibrated COLMAP dense MVS for an IDR-format DTU scan.
# Usage: sbatch colmap_dense_dtu.slurm <scan_number>

set -euo pipefail

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCAN=${1:?usage: sbatch colmap_dense_dtu.slurm <scan_number>}
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}}
TRIM=${TRIM:-0}
WS=${WS:-$ROOT/outputs/colmap_dtu_scan${SCAN}_trim${TRIM}_${SLURM_JOB_ID}}
N_SRC=${N_SRC:-10}
DEPTH_MIN=${DEPTH_MIN:-0.5}
DEPTH_MAX=${DEPTH_MAX:-3.5}
MODEL=$WS/input_model

mkdir -p "$WS" "$ROOT/outputs/logs"
test -f "$SCENE/cameras.npz"
test -d "$SCENE/image"

echo "[$(date)] scan=$SCAN scene=$SCENE workspace=$WS trim=$TRIM n_src=$N_SRC depth=$DEPTH_MIN..$DEPTH_MAX"
echo "[$(date)] colmap version:"
colmap help 2>&1 | head -1

echo "[$(date)] stage calibrated input model"
python tools/stage_colmap_dense_dtu.py \
    --scene "$SCENE" \
    --model "$MODEL"

echo "[$(date)] image_undistorter"
colmap image_undistorter \
    --image_path  "$SCENE/image" \
    --input_path  "$MODEL" \
    --output_path "$WS" \
    --output_type COLMAP

echo "[$(date)] write explicit PatchMatch source views"
python tools/stage_colmap_dense_dtu.py \
    --scene "$SCENE" \
    --model "$MODEL" \
    --source-config "$WS/stereo/patch-match.cfg" \
    --n-src "$N_SRC"

echo "[$(date)] patch_match_stereo"
colmap patch_match_stereo \
    --workspace_path "$WS" \
    --workspace_format COLMAP \
    --PatchMatchStereo.cache_size 32 \
    --PatchMatchStereo.depth_min "$DEPTH_MIN" \
    --PatchMatchStereo.depth_max "$DEPTH_MAX" \
    --PatchMatchStereo.geom_consistency true

echo "[$(date)] stereo_fusion"
colmap stereo_fusion \
    --workspace_path "$WS" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$WS/fused.ply"
if grep -q '^element vertex 0$' "$WS/fused.ply"; then
    echo "ERROR: stereo_fusion produced an empty point cloud" >&2
    exit 1
fi

echo "[$(date)] poisson_mesher trim=$TRIM"
colmap poisson_mesher \
    --input_path "$WS/fused.ply" \
    --output_path "$WS/poisson_trim${TRIM}.ply" \
    --PoissonMeshing.trim "$TRIM"

echo "[$(date)] done"
ls -lh "$WS/fused.ply" "$WS/poisson_trim${TRIM}.ply"
