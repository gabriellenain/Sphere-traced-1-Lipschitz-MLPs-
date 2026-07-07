#!/bin/bash
#SBATCH --job-name=1lip-epfl
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-epfl_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-epfl_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu,willow
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

# EPFL/Strecha pipeline matching the TnT/DTU sphere + MVSFormer++ carve recipe:
#   fixed-camera COLMAP triangulation -> sparse_sfm_points.txt
#   MVSFormer++ depths -> sphere free-carve init SDF
#   no-mask NCC training from the carved SDF
#
# Usage:
#   sbatch experiments/train_epfl_sphere_mvsformer_nomask.slurm fountain-P11
#   sbatch experiments/train_epfl_sphere_mvsformer_nomask.slurm Herz-Jesu-P8

SCENE_NAME=${1:-fountain-P11}
ROOT=${ROOT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-}
OUTPUTS=${OUTPUTS:-/scratch/_projets_/willow/1-lip-tracer-new/outputs}
OUTPUTS_HOME=$ROOT/outputs
SCENE=${SCENE:-$ROOT/data/epfl_strecha/$SCENE_NAME}

MAX_H=${MAX_H:-1152}
MAX_W=${MAX_W:-1536}
NUMDEPTH=${NUMDEPTH:-192}
NUM_VIEW=${NUM_VIEW:-5}
N_SRC=${N_SRC:-10}
MVS_BASE=${MVS_BASE:-$SCENE/mvsformer_depth_${MAX_W}x${MAX_H}}
DEPTH_DIR=${DEPTH_DIR:-${MVS_BASE}_${MAX_W}x${MAX_H}}
STAGING=${STAGING:-$ROOT/_diagnostics/mvsf_staging_${SCENE_NAME}}

CARVE_RES=${CARVE_RES:-384}
BOUND=${BOUND:-1.5}
CARVE_DIR=${CARVE_DIR:-$ROOT/_diagnostics/mvsformer_${SCENE_NAME,,}_sphere/sphere_carve_auto_res${CARVE_RES}_nomask_sfmclip_close2}
INIT_SDF=${INIT_SDF:-$CARVE_DIR/sdf_sphere_mvsformer_carved.npy}

STEPS=${STEPS:-300000}
BATCH=${BATCH:-16384}
TRACE_ITERS=${TRACE_ITERS:-64}
OCC_ITERS=${OCC_ITERS:-32}
OCC_NEWTON_STEPS=${OCC_NEWTON_STEPS:-0}
OCC_EPS=${OCC_EPS:-1e-3}
DOWN=${DOWN:-1}
JOB_TAG=${JOB_TAG:-${SCENE_NAME}_epfl_sphere_nomask_mvsformer}

source ${VENV:-/scratch/_projets_/willow/1-lip-tracer-new/.venv}/bin/activate
cd "$ROOT"
export PATH=${COLMAP_PATH:-/home/glenain/.pixi/bin}:$PATH
export MVSFORMER_REPO=${MVSFORMER_REPO:-/scratch/glenain/MVSFormerPlusPlus}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-${JOB_TAG}-${SLURM_JOB_ID:-0}

if [ ! -d "$SCENE" ]; then
  echo "ERROR: missing EPFL scene: $SCENE" >&2
  exit 1
fi

mkdir -p "$OUTPUTS" "$OUTPUTS_HOME" "$CARVE_DIR"

COLMAP_WORK=${COLMAP_WORK:-$SCENE/colmap}
if [ ! -s "$SCENE/sparse_sfm_points.txt" ]; then
  echo "Generating EPFL sparse_sfm_points.txt from fixed cameras (no GT PLY)."
  python -u tools/init_from_colmap.py \
    --scene "$SCENE" \
    --work "$COLMAP_WORK" \
    --out-png "$CARVE_DIR/colmap_init.png" \
    --no-igr
else
  echo "Reusing sparse SFM cloud: $SCENE/sparse_sfm_points.txt"
fi

if ! find "$DEPTH_DIR" -path '*/depth_est/*.pfm' -print -quit 2>/dev/null | grep -q .; then
  echo "Precomputing MVSFormer++ depths -> $DEPTH_DIR"
  python -u tools/precompute_mvsformer_depths.py \
    --scene "$SCENE" \
    --scan-name "$SCENE_NAME" \
    --out-dir "$MVS_BASE" \
    --staging "$STAGING" \
    --max-h "$MAX_H" --max-w "$MAX_W" \
    --num-view "$NUM_VIEW" --n-src "$N_SRC" \
    --numdepth "$NUMDEPTH"
else
  echo "Reusing MVSFormer++ depths: $DEPTH_DIR"
fi

if [ ! -f "$INIT_SDF" ]; then
  echo "Carving sphere + MVSFormer++ init -> $INIT_SDF"
  python -u tools/carve_sphere_with_mvsformer_truck.py \
    --scene "$SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$CARVE_DIR" \
    --res "$CARVE_RES" --bound "$BOUND" \
    --sphere-radius auto --sphere-radius-percentile 99.9 --sphere-radius-pad 0.02 \
    --center-from-sfm \
    --conf-thr 0.5 --votes-req 3 --margin-voxels 3 \
    --sfm-clip --clip-margin-voxels 6.0 \
    --close-iters 2 --largest-component
else
  echo "Reusing carved init SDF: $INIT_SDF"
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_${JOB_TAG}_${SLURM_JOB_ID}
LOG=$RUN_DIR/train.log
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

echo "Scene:          $SCENE"
echo "Sparse SFM:     $SCENE/sparse_sfm_points.txt"
echo "Depth dir:      $DEPTH_DIR"
echo "Init SDF grid:  $INIT_SDF"
echo "Run dir:        $RUN_DIR"
echo "Target steps:   $STEPS"

python -u -m lip_tracer.train \
  --scene "$SCENE" \
  --run-dir "$RUN_DIR" \
  --architecture cpl --hidden 256 --depth 8 \
  --init hull \
  --init-sdf-grid "$INIT_SDF" \
  --hull-res 256 \
  --steps "$STEPS" \
  --batch "$BATCH" \
  --trace-iters "$TRACE_ITERS" \
  --occ-iters "$OCC_ITERS" \
  --occ-newton-steps "$OCC_NEWTON_STEPS" \
  --occ-eps "$OCC_EPS" \
  --down "$DOWN" \
  --no-use-masks \
  --view-selection nearest \
  --w-photo 0.0 \
  --w-ncc 1.0 \
  --ncc-color gray \
  --ncc-patch 5 \
  --ncc-half-pix 2 \
  --w-eikonal 0.0 \
  --w-sfm 0.0 \
  --w-geo-sdf 0.0 \
  --wandb 2>&1 | tee "$LOG"
