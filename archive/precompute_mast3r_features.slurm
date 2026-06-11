#!/bin/bash
#SBATCH --job-name=mast3r-feats
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_feats_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mast3r_feats_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014,gpu001

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCAN=${SCAN:-65}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan${SCAN}}
OUT=${OUT:-outputs/scan${SCAN}_mast3r_features_1024.pt}
BATCH=${BATCH:-2}
SIZE=${SIZE:-1024}
DATASET=${DATASET:-skull}
DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric}
MIN_GPU_MEM_GB=${MIN_GPU_MEM_GB:-20}

echo "dataset: $DATASET"
echo "scene: $SCENE"
echo "out:   $OUT"
echo "model: $MODEL"
echo "batch: $BATCH"
echo "size:  $SIZE"
echo "device: $DEVICE"
echo "min_gpu_mem_gb: $MIN_GPU_MEM_GB"

if [[ "$DEVICE" == cuda* ]]; then
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    MIN_GPU_MEM_MB=$((MIN_GPU_MEM_GB * 1024))
    echo "gpu: $GPU_NAME (${GPU_MEM_MB} MiB)"
    if (( GPU_MEM_MB < MIN_GPU_MEM_MB )); then
        echo "ERROR: GPU has ${GPU_MEM_MB} MiB, need at least ${MIN_GPU_MEM_MB} MiB. Resubmit on a larger GPU."
        exit 42
    fi
fi

python -u archive/precompute_mast3r_features.py \
    --dataset "$DATASET" \
    --scene "$SCENE" \
    --model "$MODEL" \
    --out "$OUT" \
    --batch "$BATCH" \
    --size "$SIZE" \
    --device "$DEVICE"
