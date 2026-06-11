#!/bin/bash
#SBATCH --job-name=buddha-n256d16-10M
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/logs/buddha_none_W256_D16_10M_%j.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/logs/buddha_none_W256_D16_10M_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
# Exclude pre-Ampere nodes (no bf16 tensor cores → 10-20× slower on this workload):
#   gpu001       = V100      (Volta)
#   gpu002-003   = RTX 6000  (Turing)
#   gpu004       = GTX 1080ti / down
#   gpu006-009   = RTX 8000  (Turing)
#   gpu011,014   = RTX 2080ti (Turing)
# Leaves: gpu012/013 (A100), gpu015-017 (H100), gpu018 (H200), parq-gpu001 (A100).
# gpu013 is kept eligible because --compile is no longer used (the Python.h
# Triton-shim issue that knocked out 4910759 only matters under --compile).
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

# Same config as buddha_none_W256_D16_1M but trained for 10M steps.
# bf16 autocast + torch.compile + periodic checkpointing so a 48h wall hit
# can be picked up by simply re-running this script (--resume reads
# checkpoint_gt_sdf.pt from --out-dir).

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

# Mirror the slurm log back into /home/.../outputs/logs/ as a symlink so it
# shows up in the IDE (data still lives on /scratch).
mkdir -p /home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs
ln -sfn /scratch/_projets_/willow/1-lip-tracer-new/outputs/logs/buddha_none_W256_D16_10M_${SLURM_JOB_ID}.log \
        /home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_none_W256_D16_10M_${SLURM_JOB_ID}.log

python -u fit_gt_sdf.py \
    --mesh data/gt_meshes/happy_buddha_norm.ply \
    --out-dir /scratch/_projets_/willow/1-lip-tracer-new/outputs/buddha_sweep/none_W256_D16_10M \
    --architecture cpl \
    --input-encoding identity \
    --multires 0 \
    --hidden 256 \
    --depth 16 \
    --group-size 2 \
    --steps 10000000 \
    --batch 32768 \
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
    --resume \
    --ckpt-every 50000 \
    --sweep
