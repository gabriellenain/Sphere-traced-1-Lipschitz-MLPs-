#!/bin/bash
#SBATCH --job-name=conv-map
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/conv_map_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/conv_map_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00

# Per-view miss-convergence diagnostic (analysis/plot_miss_convergence_budget.py):
# trace each view at the run's TRAINING budget, take the foreground misses, then
# re-trace them to --max-iters and classify into
#   recovered (converge with more iters) / stalled (root exists) / no-root (escape).
# Overlaid on the view + the |∇f|/grazing scatter panels — i.e. WHERE & WHY rays miss.
# short-iters is read from the run config so it matches the ckpt's own trace.
# Usage:
#   sbatch conv_map.slurm <run_dir> [ckpt] [views] [down] [max_iters] [out_dir]
# Defaults: ckpt=checkpoint_latest.pt views=16 down=4 max_iters=500 out_dir=figs/sphere_trace_steps
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${1:?usage: sbatch conv_map.slurm <run_dir> [ckpt] [views] [down] [max_iters] [out_dir]}"
CKPT="${2:-checkpoint_latest.pt}"
VIEWS="${3:-16}"
DOWN="${4:-4}"
MAXIT="${5:-500}"
OUT="${6:-figs/sphere_trace_steps}"
# BRACKET=1 → define misses with the real run tracer (bracketing on), so the map
# shows only the run's TRUE residual (no-root/tangent), not the un-bracketed march.
BRACKET_ARG=""; [ -n "$BRACKET" ] && BRACKET_ARG="--bracket"

[ -d "$RUN_DIR" ] || { echo "[error] run_dir not found: $RUN_DIR"; exit 1; }
mkdir -p "$OUT"
# Training budget (sphere-trace iters) of the run that produced this ckpt.
SHORT=$(python -c "import json;print(json.load(open('$RUN_DIR/config.json'))['trace']['iters'])")
echo "run_dir: $RUN_DIR"
echo "ckpt:    $CKPT"
echo "views:   $VIEWS  down=$DOWN"
echo "budget:  short-iters=$SHORT (train budget from cfg)  max-iters=$MAXIT"
echo "out:     $OUT"

for V in ${VIEWS//,/ }; do
    echo "=================== view $V ==================="
    python -u analysis/plot_miss_convergence_budget.py "$RUN_DIR" \
        --ckpt "$CKPT" --view "$V" --down "$DOWN" \
        --short-iters "$SHORT" --max-iters "$MAXIT" $BRACKET_ARG \
        --out-dir "$OUT" || echo "[warn] view $V failed"
done
