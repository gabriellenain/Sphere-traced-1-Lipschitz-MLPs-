#!/bin/bash
#SBATCH --job-name=depth-chamfer
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/depth_chamfer_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/depth_chamfer_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

python=/home/glenain/nerfstudio/.pixi/envs/default/bin/python

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

OUT_DIR=${OUT_DIR:?set OUT_DIR=outputs/depth_refine_JOBID}
RUN=${RUN:-outputs/run_20260510_094943_scan65}
DOWN=${DOWN:-2}
DTU_EVAL_DIR=${DTU_EVAL_DIR:-/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS\ Data}
N_CHAMFER_PTS=${N_CHAMFER_PTS:-1000000}
VIEWS=${VIEWS:-}

echo "out_dir: $OUT_DIR"
echo "run: $RUN"
echo "down: $DOWN"
echo "dtu_eval_dir: $DTU_EVAL_DIR"
echo "n_chamfer_pts: $N_CHAMFER_PTS"
echo "views: ${VIEWS:-all saved}"

EXTRA_ARGS=()
if [ -n "$VIEWS" ]; then
  EXTRA_ARGS+=(--views $VIEWS)
fi

$python analysis/eval_depth_refine_chamfer.py \
  --out-dir "$OUT_DIR" \
  --run "$RUN" \
  --dtu-eval-dir "$DTU_EVAL_DIR" \
  --down "$DOWN" \
  --n-chamfer-pts "$N_CHAMFER_PTS" \
  "${EXTRA_ARGS[@]}"
