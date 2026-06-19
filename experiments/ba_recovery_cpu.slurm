#!/bin/bash
#SBATCH --job-name=ba_recovery_cpu
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/ba_recovery_cpu_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/ba_recovery_cpu_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=cpu_homogen
#SBATCH --requeue
#SBATCH --time=12:00:00

set -euo pipefail
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

# CPU variant of the minimal pose-recovery test (see ba_recovery.slurm for the
# experiment design). No GPU: device auto-falls back to CPU and prepare_context
# keeps images fp32 there (fp16 grid_sample is unimplemented on CPU). Uses the
# whole 64-core node — pin the math-library / torch thread counts to match.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export CUDA_VISIBLE_DEVICES=""
python -c "import torch; torch.set_num_threads(${OMP_NUM_THREADS}); print('torch CPU threads:', torch.get_num_threads())"

RUN=${RUN:-outputs/run_20260602_220736_scan118_occ_4954645}
CKPT=${CKPT:-$RUN/ckpt/checkpoint_final.pt}
TAG=${TAG:-cam${PERTURB_CAM:-1}_r${ROT:-0.3}_t${TRANS:-0.5}${FREE_ONLY_CAM:+_freeonly${FREE_ONLY_CAM}}}

# RESUME=<path/to/ba_final.pt> continues that pass (φ + Adam state restored);
# pass the SAME ROT/TRANS/SEED/PERTURB_CAM so the perturbed base / GT match.
RESUME_ARG=()
[ -n "${RESUME:-}" ] && RESUME_ARG=(--resume "$RESUME")
# FREE_ONLY_CAM=N optimises only camera N (freeze the rest) for a clean
# single-camera recovery test; unset leaves all i>=1 free.
FREE_ARG=()
[ -n "${FREE_ONLY_CAM:-}" ] && FREE_ARG=(--free-only-cam "$FREE_ONLY_CAM")

python -u -m lip_tracer.bundle_adjustment \
    --config "$RUN/config.json" \
    --ckpt   "$CKPT" \
    --out    "$RUN/ba_recovery_cpu_$TAG" \
    --cycles          ${CYCLES:-1} \
    --block-phi       ${BLOCK_PHI:-600} \
    --block-theta     ${BLOCK_THETA:-0} \
    --lr-phi          ${LR_PHI:-1e-5} \
    ${LR_THETA:+--lr-theta $LR_THETA} \
    ${BATCH:+--batch $BATCH} \
    ${OPT_INTRINSICS:+--opt-intrinsics} \
    ${LR_INTRINSICS:+--lr-intrinsics $LR_INTRINSICS} \
    --log-every       ${LOG_EVERY:-10} \
    --perturb-cam     ${PERTURB_CAM:-1} \
    --perturb-rot-deg ${ROT:-0.3} \
    --perturb-trans-mm ${TRANS:-0.5} \
    --perturb-seed    ${SEED:-0} \
    "${FREE_ARG[@]}" \
    "${RESUME_ARG[@]}"
