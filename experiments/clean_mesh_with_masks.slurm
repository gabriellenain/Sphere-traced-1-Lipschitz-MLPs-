#!/bin/bash
#SBATCH --job-name=mesh-mask-clean
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/clean_mesh_masks_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/clean_mesh_masks_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-

# Edit here or override at submit time:
#   SCAN=122 MIN_VIEWS=2 FACE_MODE=all sbatch clean_mesh_with_masks.slurm
SCAN=${SCAN:-122}
MESH=${MESH:-$ROOT/scan122_paper_like_clean_largest_512.ply}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan${SCAN}}
TAG=${TAG:-scan${SCAN}_mask_cleaned}
OUT=${OUT:-$ROOT/outputs/${TAG}_${SLURM_JOB_ID}.ply}
PREVIEW=${PREVIEW:-$ROOT/outputs/${TAG}_${SLURM_JOB_ID}_preview.png}

MIN_VIEWS=${MIN_VIEWS:-1}
DILATE=${DILATE:-3}
FACE_MODE=${FACE_MODE:-majority}   # any | majority | all
COORDS=${COORDS:-normalized}       # normalized | world
KEEP_LARGEST=${KEEP_LARGEST:-1}
CHUNK=${CHUNK:-200000}
PREVIEW_POINTS=${PREVIEW_POINTS:-150000}
SEED=${SEED:-0}

echo "=== mask-clean mesh ==="
echo "scan:           $SCAN"
echo "mesh:           $MESH"
echo "scene:          $SCENE"
echo "out:            $OUT"
echo "preview:        $PREVIEW"
echo "min_views:      $MIN_VIEWS"
echo "dilate:         $DILATE"
echo "face_mode:      $FACE_MODE"
echo "coords:         $COORDS"
echo "keep_largest:   $KEEP_LARGEST"
echo "chunk:          $CHUNK"

EXTRA_ARGS=()
if [[ "$KEEP_LARGEST" = "1" ]]; then
  EXTRA_ARGS+=(--keep-largest)
fi

python -u analysis/clean_mesh_with_masks.py \
  --mesh "$MESH" \
  --scene "$SCENE" \
  --out "$OUT" \
  --preview "$PREVIEW" \
  --min-views "$MIN_VIEWS" \
  --dilate "$DILATE" \
  --face-mode "$FACE_MODE" \
  --coords "$COORDS" \
  --chunk "$CHUNK" \
  --preview-points "$PREVIEW_POINTS" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
