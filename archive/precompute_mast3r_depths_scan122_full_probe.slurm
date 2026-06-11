#!/bin/bash
#SBATCH --job-name=mast3r-full-probe-122
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_full_probe_scan122_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_full_probe_scan122_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu001,gpu004,gpu011,gpu014,gpu008,gpu002

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCAN=122
SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan${SCAN}
DEVICE=cuda
MODEL=naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

IMAGE_SIZE=512
TARGET_WIDTH=1600
TARGET_HEIGHT=1200
SCENE_GRAPH=logwin-4
MAX_IMAGES=${MAX_IMAGES:-64}
MAX_PAIRS=0
PAIR_TAG=all
OUT_DIR=${SCENE}/mast3r_depth_knowncams512_${MAX_IMAGES}imgs_${PAIR_TAG}pairs

echo "scan:        $SCAN"
echo "scene:       $SCENE"
echo "out_dir:     $OUT_DIR"
echo "model:       $MODEL"
echo "image_size:  $IMAGE_SIZE"
echo "target:      ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "scene_graph: $SCENE_GRAPH"
echo "max_images:  $MAX_IMAGES"
echo "max_pairs:   $MAX_PAIRS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

python -u archive/precompute_mast3r_depths.py \
    --scene "$SCENE" \
    --out-dir "$OUT_DIR" \
    --weights "$MODEL" \
    --device "$DEVICE" \
    --image-size "$IMAGE_SIZE" \
    --target-width "$TARGET_WIDTH" \
    --target-height "$TARGET_HEIGHT" \
    --scene-graph "$SCENE_GRAPH" \
    --batch 1 \
    --niter 300 \
    --min-conf-thr 1.0 \
    --flush-cache \
    --max-images "$MAX_IMAGES" \
    --max-pairs "$MAX_PAIRS" \
    --use-idr-mask \
    --use-known-cameras
