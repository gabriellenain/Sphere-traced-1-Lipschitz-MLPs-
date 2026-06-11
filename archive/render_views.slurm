#!/bin/bash
#SBATCH --job-name=1lip-rviews
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_views_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_views_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu001,gpu004,gpu011,gpu014,gpu008

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

PT=${PT:-${1:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260430_171910_dtu/checkpoint_best_photo.pt}}
DATASET=${DATASET:-${2:-dtu}}
SCENE=${SCENE:-122}
SPLIT=${SPLIT:-train}
OUT=${OUT:-outputs/paper_renders_${SLURM_JOB_ID}}
REF_VIEWS=${REF_VIEWS:-8}
NOVEL_VIEWS=${NOVEL_VIEWS:-24}
NOVEL_MODE=${NOVEL_MODE:-train}
TRACE_ITERS=${TRACE_ITERS:-96}
EPS=${EPS:-0.0005}
T_FAR=${T_FAR:-10.0}
NEWTON_STEPS=${NEWTON_STEPS:-2}
RAY_CHUNK=${RAY_CHUNK:-65536}
RUN_VIS=${RUN_VIS:-1}
MC_RES=${MC_RES:-512}
RENDER_DOWN=${RENDER_DOWN:-1}

echo "checkpoint:   $PT"
echo "dataset:      $DATASET"
echo "scene:        ${SCENE:-default}"
echo "split:        $SPLIT"
echo "out:          $OUT"
echo "ref_views:    $REF_VIEWS"
echo "novel_views:  $NOVEL_VIEWS"
echo "novel_mode:   $NOVEL_MODE"
echo "trace_iters:  $TRACE_ITERS"
echo "eps:          $EPS"
echo "t_far:        $T_FAR"
echo "newton_steps: $NEWTON_STEPS"
echo "ray_chunk:    $RAY_CHUNK"
echo "run_vis:      $RUN_VIS"
echo "mc_res:       $MC_RES"

EXTRA_ARGS=()
if [ -n "$SCENE" ]; then
  EXTRA_ARGS+=(--scene "$SCENE")
fi

python -u -m lip_tracer.render_views \
    --pt "$PT" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --out-dir "$OUT" \
    --ref-views "$REF_VIEWS" \
    --novel-views "$NOVEL_VIEWS" \
    --novel-mode "$NOVEL_MODE" \
    --trace-iters "$TRACE_ITERS" \
    --eps "$EPS" \
    --t-far "$T_FAR" \
    --newton-steps "$NEWTON_STEPS" \
    --ray-chunk "$RAY_CHUNK" \
    "${EXTRA_ARGS[@]}"

if [ "$RUN_VIS" = "1" ]; then
  VIS_ARGS=(--no-train --pt "$PT" --dataset "$DATASET" --mc-res "$MC_RES" --render-down "$RENDER_DOWN")
  if [ -n "$SCENE" ]; then
    VIS_ARGS+=(--scene "$SCENE")
  fi
  python -u -m lip_tracer.train "${VIS_ARGS[@]}"
  cp outputs/surface_*.png "$OUT"/ 2>/dev/null || true
  cp outputs/render_vs_ref.png "$OUT"/ 2>/dev/null || true
fi
