#!/bin/bash
#SBATCH --job-name=1lip-batch16k-test
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/test_batch16k_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/test_batch16k_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu001,gpu002,gpu004,gpu008,gpu011,gpu014

SCAN=122
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --input-encoding pe \
    --multires 6 \
    --trace-iters 24 \
    --init hull \
    --init-steps 2000 \
    --steps 50 \
    --batch 16384 \
    --down 1 \
    --architecture cpl 2>&1 | tee outputs/test_batch16k_${SLURM_JOB_ID}.log

python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"GPU total:       {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
EOF
