#!/bin/bash
#SBATCH --job-name=colmap-barn
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_barn_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_barn_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Canonical COLMAP dense MVS pipeline for TnT Barn (the recipe the official
# Tanks-and-Temples leaderboard / Geo-Neus / NeuS papers report as the COLMAP
# baseline). Inputs:
#   images : data/tnt/Barn/rgb              (384 frames, 1920x1080, PINHOLE)
#   sparse : data/tnt/Barn/colmap/sparse/0  (precomputed SfM, binary)
# Stages: image_undistorter -> patch_match_stereo (geometric consistency)
#         -> stereo_fusion -> poisson_mesher.
# Outputs land in outputs/colmap_barn/ : fused.ply  poisson.ply
# Expect ~1-3h on a single modern GPU at full 1920x1080.

set -euo pipefail

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
IMG=$ROOT/data/tnt/Barn/rgb
SPARSE=$ROOT/data/tnt/Barn/colmap/sparse/0
WS=$ROOT/outputs/colmap_barn
mkdir -p $WS $ROOT/outputs/logs

echo "[$(date)] colmap version:"; colmap help 2>&1 | head -1

# --- 1. Undistort images and lay out the COLMAP MVS workspace -------------
# (Cameras are already PINHOLE, so this is essentially a copy + workspace
#  scaffold for patch_match_stereo.)
echo "[$(date)] image_undistorter"
colmap image_undistorter \
    --image_path   $IMG \
    --input_path   $SPARSE \
    --output_path  $WS \
    --output_type  COLMAP

# --- 2. PatchMatch stereo (GPU) ------------------------------------------
# Full-resolution canonical-TnT settings (geom_consistency=true). Drops the
# half-res / fewer-samples speed knobs so we can match the published ~0.55
# COLMAP F-score on Barn. Expect ~5-6h on a single GPU.
echo "[$(date)] patch_match_stereo"
colmap patch_match_stereo \
    --workspace_path   $WS \
    --workspace_format COLMAP \
    --PatchMatchStereo.cache_size     32 \
    --PatchMatchStereo.geom_consistency true

# --- 3. Stereo fusion into a single point cloud --------------------------
echo "[$(date)] stereo_fusion"
colmap stereo_fusion \
    --workspace_path   $WS \
    --workspace_format COLMAP \
    --input_type       geometric \
    --output_path      $WS/fused.ply

# --- 4. Poisson surface reconstruction -----------------------------------
echo "[$(date)] poisson_mesher"
colmap poisson_mesher \
    --input_path   $WS/fused.ply \
    --output_path  $WS/poisson.ply

echo "[$(date)] done. artifacts:"
ls -lh $WS/fused.ply $WS/poisson.ply
