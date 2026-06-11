#!/bin/bash
#SBATCH --job-name=lego-gt-mesh
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/lego_gt_mesh_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/lego_gt_mesh_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/nerf_synthetic/lego}
SPLIT=${SPLIT:-test}
OUT=${OUT:-data/gt_meshes/lego_depth_gt_mesh.ply}
STRIDE=${STRIDE:-4}
FAR=${FAR:-6.0}
MAX_EDGE=${MAX_EDGE:-0.08}
MAX_VIEWS=${MAX_VIEWS:-}

echo "scene:     $SCENE"
echo "split:     $SPLIT"
echo "out:       $OUT"
echo "stride:    $STRIDE"
echo "far:       $FAR"
echo "max_edge:  $MAX_EDGE"
echo "max_views: ${MAX_VIEWS:-all}"
echo "cpus:      $SLURM_CPUS_PER_TASK"

CMD=(
  python -u archive/make_lego_gt_mesh.py
  --scene "$SCENE"
  --split "$SPLIT"
  --out "$OUT"
  --stride "$STRIDE"
  --far "$FAR"
  --max-edge "$MAX_EDGE"
)

if [ -n "$MAX_VIEWS" ]; then
  CMD+=(--max-views "$MAX_VIEWS")
fi

"${CMD[@]}"
