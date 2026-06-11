#!/bin/bash
#SBATCH --job-name=hull-target
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/hull_target_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/hull_target_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu_homogen
#SBATCH --time=01:00:00

set -euo pipefail

RUN_DIR=${1:?Usage: sbatch export_hull_target_blend.slurm <run_dir>}
SCAN=${SCAN:-24}
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUT_DIR=$ROOT/outputs/hull_target_scan${SCAN}_${SLURM_JOB_ID}
BLENDER=${BLENDER:-$HOME/blender/blender}
HULL_MIN_VIEWS=${HULL_MIN_VIEWS:-0}

cd "$ROOT"
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
mkdir -p "$OUT_DIR"

echo "Run config: $RUN_DIR/config.json"
echo "Output dir: $OUT_DIR"
python -u tools/export_hull_target.py \
    --run-dir "$RUN_DIR" \
    --out-dir "$OUT_DIR" \
    --min-views "$HULL_MIN_VIEWS"

"$BLENDER" --background \
    --python tools/save_normalized_mesh_blend.py -- \
    --mesh "$OUT_DIR/hull_target_sdf_norm.ply" \
    --out "$OUT_DIR/hull_target_sdf_norm.blend" \
    --name "scan${SCAN}_visual_hull_sdf_target"

ls -lh "$OUT_DIR"
