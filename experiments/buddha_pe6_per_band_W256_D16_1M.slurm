#!/bin/bash
#SBATCH --job-name=buddha-pe6pb-W256D16-1M
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_pe6_per_band_W256_D16_1M_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_pe6_per_band_W256_D16_1M_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Same config as buddha_none_W256_D16_1M but with PE multires=6 and
# lipschitz_mode=per_band: each band k divided by 2^k, then output by
# sqrt(L+1). Gradient budget split evenly across passthrough and all bands.

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

python -u fit_gt_sdf.py \
    --mesh data/gt_meshes/happy_buddha_norm.ply \
    --out-dir outputs/buddha_sweep/pe6_per_band_W256_D16_1M \
    --architecture cpl \
    --input-encoding pe \
    --multires 6 \
    --lipschitz-mode per_band \
    --hidden 256 \
    --depth 16 \
    --group-size 2 \
    --steps 1000000 \
    --batch 8192 \
    --lr 0.001 \
    --bound 1.1 \
    --near-std 0.01 \
    --n-near 300000 \
    --n-vol 100000 \
    --near-frac 0.7 \
    --sdf-chunk 20000 \
    --mc-res 512 \
    --seed 0 \
    --results-csv outputs/buddha_sweep/results.csv \
    --sweep
