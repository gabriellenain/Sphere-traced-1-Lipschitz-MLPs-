#!/bin/bash
#SBATCH --job-name=sphere-carve
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sphere_carve_%j.slurm.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sphere_carve_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004
#
# MVSFormer++ sphere-carve + SFM ROI, exactly the recipe that produced the
# Ignatius arccos init (sphere_carve_auto_res384_nomask_sfmclip_close2):
#   res 384, bound 1.5, conf 0.5, votes 3, margin 3vox, sfm-clip (6vox),
#   close 2, largest-component, centre + auto-radius from sparse_sfm_points.txt.
# Requires, in the scene's own pose frame: MVSFormer++ depths AND
# <scene>/sparse_sfm_points.txt (NO_IGR=1 init_from_colmap.slurm produces it).
# Usage: sbatch carve_sphere_mvsformer.slurm <SceneName>
set -euo pipefail

SCENE_NAME=${1:-Barn}
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
# SCENE defaults to a TnT scene but is overridable (e.g. DTU on scratch:
#   SCENE=/scratch/.../data/dtu_idr/scan37).
SCENE=${SCENE:-$ROOT/data/tnt/${SCENE_NAME}}
DEPTH_DIR=${DEPTH_DIR:-$SCENE/mvsformer_depth_1152x640_1152x640}
OUT_DIR=${OUT_DIR:-$ROOT/_diagnostics/mvsformer_${SCENE_NAME,,}_sphere/sphere_carve_auto_res384_nomask_sfmclip_close2}
RES=${RES:-384}
# Carve aggressiveness knobs (override to be more conservative and avoid punching
# holes through real surface from a few bad MVSFormer depths). VOTES_REQ higher =
# more views must agree empty before a voxel is removed; DEPTH_CAP_AUTO drops
# runaway background/sky pixels; FILL_HOLES closes enclosed interior holes.
CONF_THR=${CONF_THR:-0.5}
VOTES_REQ=${VOTES_REQ:-3}
MARGIN_VOXELS=${MARGIN_VOXELS:-3.0}
CLOSE_ITERS=${CLOSE_ITERS:-2}
DEPTH_CAP_AUTO=${DEPTH_CAP_AUTO:-0}
DEPTH_CAP_PAD=${DEPTH_CAP_PAD:-0.25}
FILL_HOLES=${FILL_HOLES:-0}
GEO_CONSISTENCY=${GEO_CONSISTENCY:-0}
GEO_N_VIEWS=${GEO_N_VIEWS:-8}
GEO_N_CONSISTENT=${GEO_N_CONSISTENT:-2}
GEO_TAU_PIX=${GEO_TAU_PIX:-1.0}
GEO_TAU_DEPTH=${GEO_TAU_DEPTH:-0.01}
EXTRA_CARVE_ARGS=()
[ "$DEPTH_CAP_AUTO" = "1" ] && EXTRA_CARVE_ARGS+=(--depth-cap-auto --depth-cap-pad "$DEPTH_CAP_PAD")
[ "$FILL_HOLES" = "1" ] && EXTRA_CARVE_ARGS+=(--fill-holes)
[ "$GEO_CONSISTENCY" = "1" ] && EXTRA_CARVE_ARGS+=(--geo-consistency \
  --geo-n-views "$GEO_N_VIEWS" --geo-n-consistent "$GEO_N_CONSISTENT" \
  --geo-tau-pix "$GEO_TAU_PIX" --geo-tau-depth "$GEO_TAU_DEPTH")
SFM_ROI_DIST=${SFM_ROI_DIST:-}
[ -n "$SFM_ROI_DIST" ] && EXTRA_CARVE_ARGS+=(--sfm-roi-dist "$SFM_ROI_DIST")
DROP_SEETHROUGH=${DROP_SEETHROUGH:-0}
SEETHROUGH_WIN=${SEETHROUGH_WIN:-31}
SEETHROUGH_FACTOR=${SEETHROUGH_FACTOR:-1.3}
SEETHROUGH_DOWN=${SEETHROUGH_DOWN:-8}
[ "$DROP_SEETHROUGH" = "1" ] && EXTRA_CARVE_ARGS+=(--drop-seethrough \
  --seethrough-win "$SEETHROUGH_WIN" --seethrough-factor "$SEETHROUGH_FACTOR" \
  --seethrough-down "$SEETHROUGH_DOWN")
PROTECT_SFM_RADIUS=${PROTECT_SFM_RADIUS:-}
[ -n "$PROTECT_SFM_RADIUS" ] && EXTRA_CARVE_ARGS+=(--protect-sfm-radius "$PROTECT_SFM_RADIUS")

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-sphere-carve

if [ ! -f "$SCENE/sparse_sfm_points.txt" ]; then
    echo "ERROR: missing $SCENE/sparse_sfm_points.txt (run init_from_colmap NO_IGR=1 first)" >&2
    exit 1
fi
if [ ! -d "$DEPTH_DIR" ]; then
    echo "ERROR: missing MVSFormer depth dir: $DEPTH_DIR" >&2
    exit 1
fi

echo "Scene:     $SCENE_NAME"
echo "Depth dir: $DEPTH_DIR"
echo "Out dir:   $OUT_DIR  (carved grid -> sdf_sphere_mvsformer_carved.npy)"

python -u tools/carve_sphere_with_mvsformer_truck.py \
    --scene "$SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$OUT_DIR" \
    --res "$RES" --bound 1.5 \
    --sphere-radius auto --sphere-radius-percentile 99.9 --sphere-radius-pad 0.02 \
    --center-from-sfm \
    --conf-thr "$CONF_THR" --votes-req "$VOTES_REQ" --margin-voxels "$MARGIN_VOXELS" \
    --sfm-clip --clip-margin-voxels 6.0 \
    --close-iters "$CLOSE_ITERS" --largest-component \
    ${EXTRA_CARVE_ARGS[@]+"${EXTRA_CARVE_ARGS[@]}"}
