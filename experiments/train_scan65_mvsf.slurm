#!/bin/bash
#SBATCH --job-name=1lip-scan65-mvsf
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-scan65-mvsf_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/1lip-scan65-mvsf_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCENE=/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65
MVSF_DIR=${MVSF_DIR:-$SCENE/mvsformer_depth_1536x1152_1536x1152}
FEAT_PT=${FEAT_PT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/scan65_mvsformer_feats.pt}
DTU_EVAL=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS\ Data

python -u -m lip_tracer.train \
    --scene "$SCENE" \
    --input-encoding pe \
    --multires 6 \
    --trace-iters 24 \
    --init hull \
    --steps 40000 \
    --batch 4096 \
    --down 1 \
    --mvsformer-depth-dir "$MVSF_DIR" \
    --mvsformer-conf-thr 0.7 \
    --w-mvs-sdf 1.0 \
    --mvs-sdf-out-thresh 0.7 \
    --mvs-sdf-near-att 0.1 \
    --mvs-sdf-far-thresh 0.5 \
    --mvs-sdf-far-att 0.1 \
    --w-photo 0.5 \
    --w-feature 0.1 \
    --feature-maps "$FEAT_PT" \
    --w-eikonal 0.1 \
    --dtu-eval-dir "$DTU_EVAL" \
    --wandb 2>&1 | tee outputs/train_scan65_mvsf_${SLURM_JOB_ID}.log
