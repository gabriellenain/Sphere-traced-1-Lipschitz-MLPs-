#!/bin/bash
#SBATCH --job-name=buddha-n256d16-1M
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_none_W256_D16_1M_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_none_W256_D16_1M_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Same config as outputs/buddha_sweep/none_W256_D16 (identity encoding, CPL,
# hidden=256, depth=16, groupsort) but trained for 1M steps instead of 50K.

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

# Overridable probes (defaults reproduce the original 1M identity-CPL baseline):
#   TARGET_SCALE>1  : fit c*SDF_gt (1-Lip net forced to c-Lip -> infeasible).
#   ENC/MULTIRES/LIP_MODE/OUTPUT_DIV : the (model(gamma)/lambda) PE construction,
#     e.g. ENC=pe MULTIRES=6 LIP_MODE=none OUTPUT_DIV=36.959 fits d with raw PE.
OUT=${OUT:-outputs/buddha_sweep/none_W256_D16_1M}
STEPS=${STEPS:-1000000}
TARGET_SCALE=${TARGET_SCALE:-1.0}
ENC=${ENC:-identity}
MULTIRES=${MULTIRES:-0}
LIP_MODE=${LIP_MODE:-none}
OUTPUT_DIV=${OUTPUT_DIV:-1.0}

python -u fit_gt_sdf.py \
    --mesh data/gt_meshes/happy_buddha_norm.ply \
    --out-dir "$OUT" \
    --target-scale "$TARGET_SCALE" \
    --output-div "$OUTPUT_DIV" \
    --architecture cpl \
    --input-encoding "$ENC" \
    --multires "$MULTIRES" \
    --lipschitz-mode "$LIP_MODE" \
    --hidden 256 \
    --depth 16 \
    --group-size 2 \
    --steps "$STEPS" \
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
