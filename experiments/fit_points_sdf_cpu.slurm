#!/bin/bash
#SBATCH --job-name=points-sdf-fit-cpu
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/points_sdf_fit_cpu_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/points_sdf_fit_cpu_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --partition=cpu_homogen
#SBATCH --time=03:00:00

# CPU variant of fit_points_sdf.slurm (GPU is ~20-40x faster; ~80 min on a full
# CPU node for 5000 steps + MC256). Same env vars / args as the GPU script.
source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export CUDA_VISIBLE_DEVICES=""

# Reusable: point it at any scene's sparse_sfm_points.txt.
# Default = closed-by-construction fit (eps-ball flood-filled SDF regression).
# LEGACY=1 restores the old IGR/SAL objective (uses LAM_OFF/ALPHA/ROI_PAD).
POINTS=${POINTS:-data/dtu_idr/scan122/sparse_sfm_points.txt}
OUT_DIR=${OUT_DIR:-outputs/points_sdf_fit_scan122}
STEPS=${STEPS:-5000}
ROI_PAD=${ROI_PAD:-0.1}
BOUND=${BOUND:-1.0}
MC_RES=${MC_RES:-256}
LAM_OFF=${LAM_OFF:-0.1}
ALPHA=${ALPHA:-100.0}
EPS=${EPS:-0}            # 0 = auto (EPS_SCALE x median NN spacing)
MAX_COMPONENTS=${MAX_COMPONENTS:-1}
SHIFT=${SHIFT:-0}        # deflate level set toward the cloud (risks re-opening gaps)
GRID_RES=${GRID_RES:-256}
LEGACY_ARG=()
[ "${LEGACY:-0}" = "1" ] && LEGACY_ARG=(--legacy-igr)

echo "points : $POINTS"
echo "out_dir: $OUT_DIR"
echo "threads=$OMP_NUM_THREADS steps=$STEPS bound=$BOUND mc_res=$MC_RES eps=$EPS max_comp=$MAX_COMPONENTS shift=$SHIFT grid_res=$GRID_RES legacy=${LEGACY:-0}"

python -u fit_points_sdf.py \
    --points "$POINTS" \
    --out-dir "$OUT_DIR" \
    --steps "$STEPS" \
    --roi-pad "$ROI_PAD" \
    --bound "$BOUND" \
    --mc-res "$MC_RES" \
    --lam-off "$LAM_OFF" \
    --alpha "$ALPHA" \
    --eps "$EPS" \
    --max-components "$MAX_COMPONENTS" \
    --shift "$SHIFT" \
    --grid-res "$GRID_RES" \
    "${LEGACY_ARG[@]}"
