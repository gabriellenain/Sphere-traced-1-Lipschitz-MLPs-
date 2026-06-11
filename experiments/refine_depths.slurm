#!/bin/bash
#SBATCH --job-name=refine-depths
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/refine_depths_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/refine_depths_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=gpu004,gpu011,gpu014
#SBATCH --time=04:00:00

python=/home/glenain/nerfstudio/.pixi/envs/default/bin/python

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

OUT=${OUT:-outputs/depth_refine_${SLURM_JOB_ID:-manual}}
DOWN=${DOWN:-2}
ITERS=${ITERS:-100}
LR=${LR:-5e-3}
W_SMOOTH=${W_SMOOTH:-0.1}
W_ANCHOR=${W_ANCHOR:-50}
N_ALT=${N_ALT:-8}
N_EVAL_ALT=${N_EVAL_ALT:-4}
N_POOL=${N_POOL:-16}
NCC_PATCH=${NCC_PATCH:-5}
NCC_HALF_PIX=${NCC_HALF_PIX:-2.0}
NCC_ACCUM_BACKWARD=${NCC_ACCUM_BACKWARD:-0}
VIEWS=${VIEWS:-}
SAVE_MC_MESH=${SAVE_MC_MESH:-0}
MC_MESH_PATH=${MC_MESH_PATH:-}
MC_RES=${MC_RES:-512}
MC_BOUND=${MC_BOUND:-1.0}
DTU_EVAL_DIR=${DTU_EVAL_DIR:-/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data}
N_CHAMFER_PTS=${N_CHAMFER_PTS:-1000000}
echo "out: $OUT"
echo "down: $DOWN  iters: $ITERS  lr: $LR  w_smooth: $W_SMOOTH  w_anchor: $W_ANCHOR  n_alt: $N_ALT  n_eval_alt: $N_EVAL_ALT  n_pool: $N_POOL"
echo "ncc_patch: $NCC_PATCH  ncc_half_pix: $NCC_HALF_PIX"
echo "ncc_accum_backward: $NCC_ACCUM_BACKWARD"
echo "views: ${VIEWS:-all}"
echo "save_mc_mesh: $SAVE_MC_MESH  mc_res: $MC_RES  mc_bound: $MC_BOUND"
echo "mc_mesh_path: ${MC_MESH_PATH:-none}"

EXTRA_ARGS=()
if [ -n "$VIEWS" ]; then
  EXTRA_ARGS+=(--views $VIEWS)
fi
if [ "$SAVE_MC_MESH" = "1" ]; then
  EXTRA_ARGS+=(--save_mc_mesh --mc_res "$MC_RES" --mc_bound "$MC_BOUND")
fi
if [ -n "$MC_MESH_PATH" ]; then
  EXTRA_ARGS+=(--mc_mesh_path "$MC_MESH_PATH")
fi
if [ -n "$DTU_EVAL_DIR" ]; then
  EXTRA_ARGS+=(--dtu_eval_dir "$DTU_EVAL_DIR" --n_chamfer_pts "$N_CHAMFER_PTS")
fi
if [ "$NCC_ACCUM_BACKWARD" = "1" ]; then
  EXTRA_ARGS+=(--ncc_accum_backward)
fi

$python analysis/refine_depths.py \
    --run   outputs/run_20260510_094943_scan65 \
    --down  "$DOWN" \
    --iters "$ITERS" \
    --lr "$LR" \
    --w_smooth "$W_SMOOTH" \
    --w_anchor "$W_ANCHOR" \
    --n_alt "$N_ALT" \
    --n_eval_alt "$N_EVAL_ALT" \
    --n_pool "$N_POOL" \
    --ncc_patch "$NCC_PATCH" \
    --ncc_half_pix "$NCC_HALF_PIX" \
    --out   "$OUT" \
    "${EXTRA_ARGS[@]}"
