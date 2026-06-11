#!/bin/bash
#SBATCH --job-name=buddha-pe-sweep-W256D16-50k
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_pe_sweep_W256_D16_50k_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_pe_sweep_W256_D16_50k_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Buddha SDF-regression PE-bands sweep.
# Fixed: W=256, D=16, cpl + groupsort-2, lipschitz_mode=none, 50k steps.
# Vary:  multires in {1,2,3,4,5,6,7,8,9,10}.
# Each run trains, marching-cubes the zero level set, evaluates Chamfer,
# and appends one row to outputs/buddha_sweep/results.csv. Resumable.

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

MESH=data/gt_meshes/happy_buddha_norm.ply
SWEEP_DIR=outputs/buddha_sweep
RESULTS_CSV=$SWEEP_DIR/results.csv
STEPS=${STEPS:-50000}
MC_RES=${MC_RES:-512}

mkdir -p "$SWEEP_DIR"

MULTIRES_LIST=(1 2 3 4 5 6 7 8 9 10)

run=0
total=${#MULTIRES_LIST[@]}
for L in "${MULTIRES_LIST[@]}"; do
    run=$((run + 1))
    OUT="$SWEEP_DIR/pe${L}_none_W256_D16_50k"
    echo "=============================================================="
    echo "[$run/$total] multires=$L -> $OUT"
    echo "=============================================================="
    if [ -f "$OUT/checkpoint_gt_sdf.pt" ]; then
        echo "already done ($OUT/checkpoint_gt_sdf.pt exists) -- skipping"
        continue
    fi
    python -u fit_gt_sdf.py \
        --mesh "$MESH" \
        --out-dir "$OUT" \
        --architecture cpl \
        --input-encoding pe \
        --multires "$L" \
        --lipschitz-mode none \
        --hidden 256 \
        --depth 16 \
        --group-size 2 \
        --steps "$STEPS" \
        --batch 8192 \
        --bound 1.1 \
        --near-std 0.01 \
        --n-near 300000 \
        --n-vol 100000 \
        --sdf-chunk 20000 \
        --mc-res "$MC_RES" \
        --seed 0 \
        --results-csv "$RESULTS_CSV" \
        --sweep \
        || echo "[$run/$total] FAILED: multires=$L"
done

echo "sweep complete -- results in $RESULTS_CSV"
