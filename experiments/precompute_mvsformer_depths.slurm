#!/bin/bash
#SBATCH --job-name=mvsf-depth
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvsformer_depth_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvsformer_depth_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004
#
# Generic MVSFormer++ depth+confidence precompute for ANY IDR/TnT/Blender scene.
# Consolidates the per-scene precompute_mvsformer_depths_*.slurm wrappers; the
# python entrypoint already auto-detects the scene layout. Override via env vars:
#   SCENE      scene dir (required)
#   SCAN_NAME  staging subdir name (default: scene basename)
#   OUT_DIR    depth output dir (default: $SCENE/mvsformer_depth_${MAX_W}x${MAX_H})
#   MAX_H MAX_W NUMDEPTH NUM_VIEW N_SRC INTERVAL_SCALE  MVSFormer knobs
#   BLENDER=1  parse transforms_train.json instead of cameras.npz
# Example (MVMannequin kinette):
#   SCENE=data/mvmannequin_neus/kinette-cos-hx MAX_H=1536 MAX_W=1536 \
#     sbatch experiments/precompute_mvsformer_depths.slurm
set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
cd "$ROOT"
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate

# Resolve SCENE to an absolute path: the python entrypoint runs MVSFormer's
# test.py with cwd=MVSF_REPO, so a relative --out-dir would land under the
# MVSFormer repo (on scratch), not here. Absolute paths avoid that surprise.
SCENE=$(readlink -f "${SCENE:?set SCENE to the scene dir}")
SCAN_NAME=${SCAN_NAME:-$(basename "$SCENE")}
MAX_H=${MAX_H:-1152}
MAX_W=${MAX_W:-1536}
NUMDEPTH=${NUMDEPTH:-192}
NUM_VIEW=${NUM_VIEW:-5}
N_SRC=${N_SRC:-10}
INTERVAL_SCALE=${INTERVAL_SCALE:-1.06}
OUT_DIR=${OUT_DIR:-$SCENE/mvsformer_depth_${MAX_W}x${MAX_H}}
STAGING=${STAGING:-$ROOT/_diagnostics/mvsf_staging_${SCAN_NAME}}

export MPLCONFIGDIR=/tmp/matplotlib-mvsformer-${SLURM_JOB_ID:-0}
export MVSFORMER_REPO=${MVSFORMER_REPO:-/scratch/glenain/MVSFormerPlusPlus}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA=()
[ "${BLENDER:-0}" = "1" ] && EXTRA+=(--blender)

echo "scene=$SCENE  scan=$SCAN_NAME  max=${MAX_W}x${MAX_H}  out=$OUT_DIR"
python -u tools/precompute_mvsformer_depths.py \
  --scene "$SCENE" \
  --scan-name "$SCAN_NAME" \
  --out-dir "$OUT_DIR" \
  --staging "$STAGING" \
  --max-h "$MAX_H" --max-w "$MAX_W" \
  --num-view "$NUM_VIEW" --n-src "$N_SRC" \
  --numdepth "$NUMDEPTH" \
  --interval-scale "$INTERVAL_SCALE" \
  ${EXTRA[@]+"${EXTRA[@]}"}

# NB: MVSFormer's test.py appends _${MAX_W}x${MAX_H} to --outdir, so the real
# per-view outputs (and the dir to feed the carve as DEPTH_DIR) live here:
echo "[done] depth_est + confidence under ${OUT_DIR}_${MAX_W}x${MAX_H}/${SCAN_NAME}"
