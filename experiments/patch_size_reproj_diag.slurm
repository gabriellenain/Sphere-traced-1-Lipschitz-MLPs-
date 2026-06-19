#!/bin/bash
#SBATCH --job-name=patch-size-diag
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/patch_size_diag_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/patch_size_diag_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate

# Resolve preempted runs by job id: pass the ORIGINAL-timestamp dir that holds
# config.json + ckpt/ (not the latest log-only requeue dir).
RUN_DIR="${1:?usage: sbatch patch_size_reproj_diag.slurm <run_dir> [views] [n_rays] [ckpt]}"
VIEWS="${2:-}"                       # empty -> 8 spread across all views
N_RAYS="${3:-400}"
CKPT="${4:-}"                        # empty -> ckpt/checkpoint_step_000000.pt (init)
PATCHES="${PATCHES:-5,2;7,3;11,5}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "run_dir: $RUN_DIR"
echo "views=${VIEWS:-<auto>}  n_rays=$N_RAYS  patches=$PATCHES  ckpt=${CKPT:-<init>}"
echo "gpu: ${CUDA_VISIBLE_DEVICES:-unset}"

ARGS=(--run-dir "$RUN_DIR" --n-rays "$N_RAYS" --patches "$PATCHES" --device cuda)
[ -n "$VIEWS" ] && ARGS+=(--views "$VIEWS")
[ -n "$CKPT" ]  && ARGS+=(--ckpt "$CKPT")

python -u analysis/patch_size_reproj_diag.py "${ARGS[@]}"
