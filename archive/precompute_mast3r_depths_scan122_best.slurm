#!/bin/bash
#SBATCH --job-name=mast3r-best-122
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_best_scan122_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_best_scan122_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu001,gpu004,gpu011,gpu014,gpu008,gpu002

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCAN=122
SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan${SCAN}
DEVICE=cuda
MODEL=naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

# Highest resolution that previously made it deep into inference on 22 GB
# GPUs. Full 1600x1200 has its own probe job first.
IMAGE_SIZE=${IMAGE_SIZE:-1024}
TARGET_WIDTH=1600
TARGET_HEIGHT=1200
SCENE_GRAPH=${SCENE_GRAPH:-swin-2}
OUT_DIR=${OUT_DIR:-${SCENE}/mast3r_depth_best_input${IMAGE_SIZE}_${TARGET_WIDTH}x${TARGET_HEIGHT}}

echo "scan:        $SCAN"
echo "scene:       $SCENE"
echo "out_dir:     $OUT_DIR"
echo "model:       $MODEL"
echo "image_size:  $IMAGE_SIZE"
echo "target:      ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "scene_graph: $SCENE_GRAPH"

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
    --no-symmetrize \
    --flush-cache \
    --use-idr-mask
