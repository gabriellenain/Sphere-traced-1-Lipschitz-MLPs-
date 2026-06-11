#!/bin/bash
#SBATCH --job-name=buddha-sweep
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_sweep_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/buddha_sweep_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

# Sequential Buddha SDF-regression architecture sweep:
#   widths    = 64, 128, 256, 512
#   depths    = 4, 8, 16
#   encodings = none (identity), pe (Fourier PE, L=6)
# 4 x 3 x 2 = 24 runs, architecture fixed to cpl (1-Lipschitz).
# Each run trains, saves a checkpoint, marching-cubes the zero level set,
# evaluates Chamfer, and appends one row to outputs/buddha_sweep/results.csv.
# Re-running skips configs whose checkpoint already exists (resumable).

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

MESH=data/gt_meshes/happy_buddha_norm.ply
SWEEP_DIR=outputs/buddha_sweep
RESULTS_CSV=$SWEEP_DIR/results.csv
STEPS=${STEPS:-50000}
MC_RES=${MC_RES:-512}

# Normalise the raw Stanford scan once: centre, scale to [-1,1], Y-up -> Z-up.
python -c "
from pathlib import Path
if not Path('$MESH').exists():
    import numpy as np, trimesh
    raw = trimesh.load('artifacts/happy_recon.ply', process=False)
    v = np.asarray(raw.vertices, np.float32)
    v -= v.mean(0); v /= np.abs(v).max()
    v = v[:, [0, 2, 1]]
    Path('data/gt_meshes').mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v, faces=raw.faces, process=True).export('$MESH')
    print('saved $MESH')
else:
    print('$MESH already exists')
"

mkdir -p "$SWEEP_DIR"

WIDTHS=(64 128 256 512)
DEPTHS=(4 8 16)
ENCODINGS=(none pe)

run=0
for enc in "${ENCODINGS[@]}"; do
    if [ "$enc" = "pe" ]; then
        ENC_ARGS=(--input-encoding pe --multires 6)
    else
        ENC_ARGS=(--input-encoding identity --multires 0)
    fi
    for width in "${WIDTHS[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            run=$((run + 1))
            OUT="$SWEEP_DIR/${enc}_W${width}_D${depth}"
            echo "=============================================================="
            echo "[$run/24] enc=$enc width=$width depth=$depth -> $OUT"
            echo "=============================================================="
            if [ -f "$OUT/checkpoint_gt_sdf.pt" ]; then
                echo "already done ($OUT/checkpoint_gt_sdf.pt exists) -- skipping"
                continue
            fi
            python -u fit_gt_sdf.py \
                --mesh "$MESH" \
                --out-dir "$OUT" \
                --architecture cpl \
                "${ENC_ARGS[@]}" \
                --hidden "$width" \
                --depth "$depth" \
                --group-size 2 \
                --steps "$STEPS" \
                --batch 8192 \
                --bound 1.1 \
                --near-std 0.01 \
                --n-near 300000 \
                --n-vol 100000 \
                --sdf-chunk 20000 \
                --mc-res "$MC_RES" \
                --results-csv "$RESULTS_CSV" \
                --sweep \
                || echo "[$run/24] FAILED: enc=$enc width=$width depth=$depth"
        done
    done
done

echo "sweep complete -- results in $RESULTS_CSV"
