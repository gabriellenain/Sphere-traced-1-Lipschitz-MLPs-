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
SCENE=$ROOT/data/tnt/${SCENE_NAME}
DEPTH_DIR=${DEPTH_DIR:-$SCENE/mvsformer_depth_1152x640_1152x640}
OUT_DIR=${OUT_DIR:-$ROOT/_diagnostics/mvsformer_${SCENE_NAME,,}_sphere/sphere_carve_auto_res384_nomask_sfmclip_close2}
RES=${RES:-384}

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
    --conf-thr 0.5 --votes-req 3 --margin-voxels 3.0 \
    --sfm-clip --clip-margin-voxels 6.0 \
    --close-iters 2 --largest-component
