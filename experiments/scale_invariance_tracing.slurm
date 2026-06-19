#!/bin/bash
#SBATCH --job-name=scale_inv
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/scale_inv_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/scale_inv_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu_homogen
#SBATCH --requeue
#SBATCH --time=01:00:00

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export PYTHONPATH=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Scale-(non)invariance of finite-budget sphere tracing (analysis/scale_invariance_tracing.py).
# Demonstrates f_c = f/c shares the zero set but changes the K-step traced depth.
# CPU run (GPU queue full): lighter heatmap downsample / budget keep it ~minutes.
RUN=${RUN:-outputs/run_20260613_134200_scan69_sphere_nomask_init_5014496}
VIEW=${VIEW:-16}
# GAMMA optional: unset -> script derives the raw-PE Lipschitz gain sqrt((4^L+2)/3).
GAMMA_ARG=""
[ -n "${GAMMA:-}" ] && GAMMA_ARG="--gamma $GAMMA"

python -u analysis/scale_invariance_tracing.py \
    --run "$RUN" \
    --view "$VIEW" \
    $GAMMA_ARG \
    --n-rays "${N_RAYS:-6000}" \
    ${EXTRA_ARGS:-}
