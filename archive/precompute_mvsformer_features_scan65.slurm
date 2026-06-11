#!/bin/bash
#SBATCH --job-name=mvsf-feats-scan65
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvsf_feats_scan65_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvsf_feats_scan65_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

python -u tools/precompute_mvsformer_features.py \
    --scene /scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65 \
    --out outputs/scan65_mvsformer_feats.pt \
    --stage stage4
