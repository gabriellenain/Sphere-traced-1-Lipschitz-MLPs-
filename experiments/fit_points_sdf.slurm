#!/bin/bash
#SBATCH --job-name=points-sdf-fit
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/points_sdf_fit_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/points_sdf_fit_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

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

# LD=1 : MVSDF L_D-only fit (Zhang et al. 2021); --points is a fused scanXX.ply
# with a sibling depth dir (or set MVS_DEPTH_ROOT). Whole-space sampling by default.
LD_ARG=()
if [ "${LD:-0}" = "1" ]; then
    LD_ARG=(--ld
            --ld-pix-stride "${LD_PIX_STRIDE:-4}"
            --ld-conf-thr "${LD_CONF_THR:-0.5}"
            --ld-t-out "${LD_T_OUT:-2}"
            --ld-surface-frac "${LD_SURFACE_FRAC:-0.5}"
            --ld-eikonal-weight "${LD_EIKONAL_WEIGHT:-0.1}")
    [ -n "${MVS_DEPTH_ROOT:-}" ] && LD_ARG+=(--mvs-depth-root "$MVS_DEPTH_ROOT")
fi

echo "points : $POINTS"
echo "out_dir: $OUT_DIR"
echo "steps=$STEPS bound=$BOUND mc_res=$MC_RES eps=$EPS max_comp=$MAX_COMPONENTS shift=$SHIFT grid_res=$GRID_RES legacy=${LEGACY:-0} ld=${LD:-0}"

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
    "${LEGACY_ARG[@]}" \
    "${LD_ARG[@]}"
