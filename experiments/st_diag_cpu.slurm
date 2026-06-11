#!/bin/bash
#SBATCH --job-name=st-diag
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_diag_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_diag_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=cpu_homogen
#SBATCH --time=02:00:00

# Sphere-trace ONE view and write the diagnostic figure from
# analysis/sphere_traced_screen_mesh.py --diag:
#   analytic normals (∇f/‖∇f‖) vs screen-space geometric normals + angular error,
#   and ‖∇f‖, |f(x_hit)|, |n·d| mapped on the surface.
# Faithful trace: iters/eps/newton come from the run's config.json (no override).
# Usage:
#   sbatch st_diag_cpu.slurm <run_dir> [view] [ss] [ckpt]
# Defaults: view=50  ss=1  ckpt=checkpoint_final.pt
# Aliasing-study mode (SSAA sweep + intra-pixel normal/depth std + jitter maps):
#   ALIAS=1 [ALIAS_SS=1,2,4,8] sbatch st_diag_cpu.slurm <run_dir> [view] _ [ckpt]
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate

RUN_DIR="${1:?usage: sbatch st_diag_cpu.slurm <run_dir> [view] [ss] [ckpt]}"
VIEW="${2:-50}"
SS="${3:-1}"
CKPT="${4:-checkpoint_final.pt}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

OUT_DIR="$RUN_DIR/st_buffers_view${VIEW}"
mkdir -p "$OUT_DIR"

echo "run_dir: $RUN_DIR"
echo "view=$VIEW  ss=$SS  ckpt=$CKPT  threads=$OMP_NUM_THREADS"

if [[ "${ALIAS:-0}" == "1" ]]; then
    ALIAS_SS="${ALIAS_SS:-1,2,4,8}"
    # MAX_ITERS overrides the run's faithful trace iters (default = run config).
    # Tag the output dir by iter count so concurrent iter sweeps don't clobber
    # each other's view<vi>_ss*_*.png (filenames don't encode iters).
    ITER_ARG=()
    ALIAS_OUT="$OUT_DIR"
    if [[ -n "${MAX_ITERS:-}" ]]; then
        ITER_ARG=(--max-iters "$MAX_ITERS")
        ALIAS_OUT="${OUT_DIR}_it${MAX_ITERS}"
        mkdir -p "$ALIAS_OUT"
    fi
    echo "ALIAS sweep ss=$ALIAS_SS iters=${MAX_ITERS:-run-default} → $ALIAS_OUT/view${VIEW}_*"
    python analysis/sphere_traced_screen_mesh.py \
        "$RUN_DIR" \
        --ckpt "$CKPT" \
        --view "$VIEW" --alias-diag --alias-ss "$ALIAS_SS" "${ITER_ARG[@]}" \
        --out "$ALIAS_OUT/view${VIEW}.ply" \
        --device cpu
else
    echo "diag → $OUT_DIR/view${VIEW}_ss${SS}_diag.png"
    python analysis/sphere_traced_screen_mesh.py \
        "$RUN_DIR" \
        --ckpt "$CKPT" \
        --view "$VIEW" --supersample "$SS" --diag \
        --out "$OUT_DIR/view${VIEW}_ss${SS}.ply" \
        --device cpu
fi
