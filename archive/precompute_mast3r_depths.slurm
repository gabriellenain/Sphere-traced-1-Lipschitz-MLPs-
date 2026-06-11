#!/bin/bash
#SBATCH --job-name=mast3r-depth
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_depth_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_depth_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu001,gpu004,gpu011,gpu014,gpu008,gpu002

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCAN=${SCAN:-122}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan${SCAN}}
DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
TARGET_WIDTH=${TARGET_WIDTH:-1600}
TARGET_HEIGHT=${TARGET_HEIGHT:-1200}
OUT_DIR=${OUT_DIR:-${SCENE}/mast3r_depth_input${IMAGE_SIZE}_${TARGET_WIDTH}x${TARGET_HEIGHT}}
SCENE_GRAPH=${SCENE_GRAPH:-logwin-4}
BATCH=${BATCH:-1}
NITER=${NITER:-300}
MIN_CONF_THR=${MIN_CONF_THR:-3.0}

echo "scan:        $SCAN"
echo "scene:       $SCENE"
echo "out_dir:     $OUT_DIR"
echo "model:       $MODEL"
echo "image_size:  $IMAGE_SIZE"
echo "target:      ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "scene_graph: $SCENE_GRAPH"
echo "batch:       $BATCH"
echo "niter:       $NITER"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u archive/precompute_mast3r_depths.py \
    --scene "$SCENE" \
    --out-dir "$OUT_DIR" \
    --weights "$MODEL" \
    --device "$DEVICE" \
    --image-size "$IMAGE_SIZE" \
    --target-width "$TARGET_WIDTH" \
    --target-height "$TARGET_HEIGHT" \
    --scene-graph "$SCENE_GRAPH" \
    --batch "$BATCH" \
    --niter "$NITER" \
    --min-conf-thr "$MIN_CONF_THR" \
    --clean-depth
