#!/bin/bash
#SBATCH --job-name=colmap-sfm
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_sfm_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/colmap_sfm_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# From-scratch COLMAP SfM on RAW (unmasked) Tanks & Temples frames -- the camera
# pipeline Neuralangelo / Geo-NeuS use on TnT. Distortion is modelled during SfM
# (OPENCV), then image_undistorter bakes it out to a clean PINHOLE camera, which
# is what lip_tracer's loader needs. Finally export TXT so tools/colmap_to_nsvf.py
# can write an NSVF-format scene.
#
# Usage:  sbatch colmap_sfm_raw.slurm <SceneName> [matcher]
#   <SceneName>  reads frames from   data/tnt/<SceneName>_raw/images/
#                writes workspace to data/tnt/<SceneName>_raw/colmap/
#   [matcher]    sequential (default, TnT frames are video-ordered) | exhaustive

set -euo pipefail

SCENE=${1:?usage: sbatch colmap_sfm_raw.slurm <SceneName> [sequential|exhaustive]}
MATCHER=${2:-sequential}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
cd "$ROOT"
export PATH=/home/glenain/.pixi/bin:$PATH

IMG=$ROOT/data/tnt/${SCENE}_raw/images
WS=$ROOT/data/tnt/${SCENE}_raw/colmap
DB=$WS/database.db
mkdir -p "$WS" "$ROOT/outputs/logs"

if [ ! -d "$IMG" ] || [ -z "$(ls -A "$IMG" 2>/dev/null)" ]; then
    echo "ERROR: no raw frames at $IMG" >&2
    echo "  Download the scene image set from https://www.tanksandtemples.org/download/" >&2
    echo "  and unpack the frames there (any *.jpg/*.png)." >&2
    exit 1
fi

echo "[$(date)] colmap: $(colmap help 2>&1 | head -1)  | scene=$SCENE matcher=$MATCHER frames=$(ls "$IMG" | wc -l)"

# --- 1. Features (one physical camera, distortion modelled) ----------------
colmap feature_extractor \
    --database_path "$DB" \
    --image_path    "$IMG" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model  OPENCV \
    --FeatureExtraction.use_gpu 1

# --- 2. Match --------------------------------------------------------------
if [ "$MATCHER" = "exhaustive" ]; then
    colmap exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 1
else
    colmap sequential_matcher --database_path "$DB" --FeatureMatching.use_gpu 1
fi

# --- 3. Sparse reconstruction ---------------------------------------------
mkdir -p "$WS/sparse"
colmap mapper \
    --database_path "$DB" \
    --image_path    "$IMG" \
    --output_path   "$WS/sparse"

# --- 4. Undistort -> clean PINHOLE images + model -------------------------
# image_undistorter rewrites the model into $WS/{images,sparse} as PINHOLE.
colmap image_undistorter \
    --image_path  "$IMG" \
    --input_path  "$WS/sparse/0" \
    --output_path "$WS" \
    --output_type COLMAP

# --- 5. Export TXT for the python converter --------------------------------
mkdir -p "$WS/sparse_txt"
colmap model_converter \
    --input_path  "$WS/sparse" \
    --output_path "$WS/sparse_txt" \
    --output_type TXT

echo "[$(date)] done. Next:"
echo "  python tools/colmap_to_nsvf.py --colmap $WS --out $ROOT/data/tnt/${SCENE}_colmap"
echo "  (inspect printed bbox extent; then train on scene ${SCENE}_colmap)"
