#!/bin/bash
#SBATCH --job-name=compare-geo
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/compare_geo_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/compare_geo_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

A_CKPT=${A_CKPT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260430_105248_blender/checkpoint_best_photo.pt}
GT_MESH=${GT_MESH:-data/gt_meshes/lego_depth_gt_mesh.ply}

if [ ! -f "${GT_MESH}" ]; then
    python -u archive/make_lego_gt_mesh.py \
        --out "${GT_MESH}" \
        --stride 4 \
        --max-edge 0.08
fi

python -u archive/compare_geometry.py \
    --a-ckpt "${A_CKPT}" \
    --b-mesh "${GT_MESH}" \
    --res 256 --bound 1.5 --n-pts 100000 \
    --thresholds 0.005 0.010 0.025 \
    --save-meshes \
    --out outputs/compare_${SLURM_JOB_ID}
