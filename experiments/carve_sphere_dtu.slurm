#!/bin/bash
#SBATCH --job-name=carve-dtu
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/carve_dtu_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/carve_dtu_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

set -euo pipefail

# Generic sphere + MVSFormer++ free-carve init for a DTU(-idr) scan.
# Reproduces the scan122 no-mask carve recipe by default, but with AUTO sphere
# geometry (radius = SfM percentile + pad, auto center) so it retargets to any
# scan. Override via env vars:
#   SCENE      DTU-idr scene dir (must already have MVSFormer depths)
#   DEPTH_DIR  MVSFormer depth dir (default: $SCENE/mvsformer_depth_${MAX_W}x${MAX_H}_${MAX_W}x${MAX_H})
#   OUT_DIR    carve output dir (default derived from scene name)
#   RES BOUND CONF VOTES MARGIN CLIP_MARGIN CLOSE  carve knobs (scan122 defaults)

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-carve-${SLURM_JOB_ID:-0}

SCENE=${SCENE:?set SCENE to the DTU-idr scene dir}
MAX_H=${MAX_H:-1152}
MAX_W=${MAX_W:-1536}
DEPTH_DIR=${DEPTH_DIR:-$SCENE/mvsformer_depth_${MAX_W}x${MAX_H}_${MAX_W}x${MAX_H}}
SCAN=$(basename "$SCENE")
OUT_DIR=${OUT_DIR:-$ROOT/_diagnostics/mvsformer_${SCAN}_sphere/sphere_carve_auto_res256_nomask_clip2_close2}

RES=${RES:-256}
BOUND=${BOUND:-1.5}
CONF=${CONF:-0.5}
VOTES=${VOTES:-3}
MARGIN=${MARGIN:-3}
CLIP_MARGIN=${CLIP_MARGIN:-2}
CLOSE=${CLOSE:-2}

if [ ! -d "$DEPTH_DIR" ]; then
  echo "ERROR: missing MVSFormer depth dir: $DEPTH_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "Scene:      $SCENE"
echo "Depth dir:  $DEPTH_DIR"
echo "Out dir:    $OUT_DIR"
echo "Sphere:     auto radius + auto center"

python -u tools/carve_sphere_with_mvsformer_truck.py \
  --scene "$SCENE" \
  --depth-dir "$DEPTH_DIR" \
  --out-dir "$OUT_DIR" \
  --res "$RES" \
  --bound "$BOUND" \
  --conf-thr "$CONF" \
  --votes-req "$VOTES" \
  --margin-voxels "$MARGIN" \
  --sfm-clip \
  --clip-margin-voxels "$CLIP_MARGIN" \
  --close-iters "$CLOSE"

echo "Carved init SDF: $OUT_DIR/sdf_sphere_mvsformer_carved.npy"
