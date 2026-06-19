#!/bin/bash
#SBATCH --job-name=1lip-lego-sphere-mvsf
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/lego_sphere_mvsformer_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/lego_sphere_mvsformer_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu004

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/nerf_synthetic/lego}
DEPTH_DIR=${DEPTH_DIR:-$SCENE/mvsformer_depth_800x800_800x800}
CARVE_RES=${CARVE_RES:-256}
BOUND=${BOUND:-1.5}
CARVE_OUT=${CARVE_OUT:-$ROOT/_diagnostics/mvsformer_lego_sphere/sphere_carve_auto_res${CARVE_RES}_mvsformer_close2}
INIT_SDF=${INIT_SDF:-$CARVE_OUT/sdf_sphere_mvsformer_carved.npy}
JOB_TAG=${JOB_TAG:-lego_sphere_mvsformer_init}
RESUME=${RESUME:-}
STEPS=${STEPS:-200000}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-${JOB_TAG}-${SLURM_JOB_ID}

mkdir -p "$OUTPUTS" "$OUTPUTS_HOME" "$CARVE_OUT"

SFM_ARGS=()
if [ -f "$SCENE/sparse_sfm_points.txt" ]; then
  SFM_ARGS=(--sfm-clip)
else
  echo "No sparse_sfm_points.txt in $SCENE; running sphere + MVSFormer++ carve without SfM ROI clip."
fi

if [ ! -f "$INIT_SDF" ]; then
  echo "Carving Lego init SDF -> $INIT_SDF"
  python -u tools/carve_sphere_with_mvsformer_truck.py \
    --scene "$SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$CARVE_OUT" \
    --blender \
    --res "$CARVE_RES" \
    --bound "$BOUND" \
    --conf-thr 0.5 \
    --votes-req 3 \
    --margin-voxels 3 \
    --close-iters 2 \
    "${SFM_ARGS[@]}"
else
  echo "Reusing existing init SDF: $INIT_SDF"
fi

RESUME_ARGS=()
if [ -z "$RESUME" ] && [ "${AUTO_RESUME:-1}" != "0" ]; then
  if [ -n "${RUN_DIR:-}" ]; then
    CANDIDATE_RUN="$RUN_DIR"
  else
    CANDIDATE_RUN=$(find -L "$OUTPUTS" -maxdepth 1 -type d \
      -name "run_*_${JOB_TAG}_*" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | awk 'NR==1 {print $2}')
  fi
  if [ -n "${CANDIDATE_RUN:-}" ]; then
    for name in ckpt/checkpoint_latest.pt ckpt/checkpoint.pt ckpt/checkpoint_final.pt checkpoint_latest.pt checkpoint.pt checkpoint_final.pt; do
      if [ -f "$CANDIDATE_RUN/$name" ]; then
        RESUME="$CANDIDATE_RUN/$name"
        echo "Auto-resuming from: $RESUME"
        break
      fi
    done
  fi
fi
if [ -n "$RESUME" ]; then
  if [ ! -f "$RESUME" ]; then
    echo "ERROR: missing resume checkpoint: $RESUME" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "$RESUME")
  if [ "$(basename "$(dirname "$RESUME")")" = "ckpt" ]; then
    RUN_DIR=$(dirname "$(dirname "$RESUME")")
  else
    RUN_DIR=$(dirname "$RESUME")
  fi
  LOG=$RUN_DIR/train.log
  echo "Resuming from:  $RESUME"
else
  TS=$(date +%Y%m%d_%H%M%S)
  RUN_DIR=$OUTPUTS/run_${TS}_${JOB_TAG}_${SLURM_JOB_ID}
  LOG=$RUN_DIR/train.log
fi
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

echo "Scene:          $SCENE"
echo "Depth dir:      $DEPTH_DIR"
echo "Init SDF grid:  $INIT_SDF"
echo "Run dir:        $RUN_DIR"
echo "Target steps:   $STEPS"

python -u -m lip_tracer.train \
  --dataset lego \
  --scene "$SCENE" \
  --run-dir "$RUN_DIR" \
  --init hull \
  --init-sdf-grid "$INIT_SDF" \
  --hull-res "$CARVE_RES" \
  --steps "$STEPS" \
  --batch 16384 \
  --eps 1e-4 \
  --down 1 \
  --no-use-masks \
  --w-eikonal 0.0 \
  --w-photo 0.0 \
  --w-ncc 1.0 \
  --w-mask-fg 0.0 \
  --w-mask-bg 0.0 \
  --fg-fraction 0.9 \
  --init-hit-sampling \
  --occ-iters 32 \
  --occ-newton-steps 0 \
  --occ-eps 1e-3 \
  "${RESUME_ARGS[@]}" \
  --wandb 2>&1 | tee -a "$LOG"
