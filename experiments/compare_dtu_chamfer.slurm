#!/bin/bash
#SBATCH --job-name=dtu-chamfer
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_chamfer_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_chamfer_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

SCAN=${SCAN:-105}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}}
DTU_EVAL=${DTU_EVAL:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data}
PT=${PT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260515_100108_scan65/checkpoint_best_photo.pt}
PSI=${PSI:-}
OUT=${OUT:-outputs/dtu_chamfer_${SLURM_JOB_ID}}
RES=${RES:-512}
BOUND=${BOUND:-1.0}
N_PRED_POINTS=${N_PRED_POINTS:-1000000}
MAX_DIST=${MAX_DIST:-20}
GT_PLY=${GT_PLY:-$DTU_EVAL/Points/stl/stl$(printf "%03d" "$SCAN")_total.ply}
GT_CROP=${GT_CROP:-pred-bbox}
GT_CROP_PADDING=${GT_CROP_PADDING:-0.02}
GT_MAX_DIST_TO_PRED=${GT_MAX_DIST_TO_PRED:-}
GT_Z_MIN=${GT_Z_MIN:-}
GT_Z_MAX=${GT_Z_MAX:-}
SAVE_EVAL_CLOUDS=${SAVE_EVAL_CLOUDS:-0}
SAVE_PNG=${SAVE_PNG:-1}
GT_MASK_FILTER=${GT_MASK_FILTER:-1}
GT_MASK_MODE=${GT_MASK_MODE:-any}
GT_MASK_MIN_RATIO=${GT_MASK_MIN_RATIO:-0.95}
GT_MASK_MIN_VIEWS=${GT_MASK_MIN_VIEWS:-1}
PNG_VMAX=${PNG_VMAX:-}
PNG_ALIGN_PCA=${PNG_ALIGN_PCA:-1}
PNG_ROTATE_DEG=${PNG_ROTATE_DEG:-0}

echo "scan:          $SCAN"
echo "scene:         $SCENE"
echo "dtu_eval:      $DTU_EVAL"
echo "checkpoint:    $PT"
echo "psi:           ${PSI:-none}"
echo "out:           $OUT"
echo "res:           $RES"
echo "bound:         $BOUND"
echo "n_pred_points: $N_PRED_POINTS"
echo "gt_ply:        $GT_PLY"
echo "gt_crop:       $GT_CROP"
echo "crop_padding:  $GT_CROP_PADDING"
echo "gt_max_dist:   ${GT_MAX_DIST_TO_PRED:-none}"
echo "gt_mask_filter:$GT_MASK_FILTER"
echo "gt_mask_mode:  $GT_MASK_MODE"
echo "gt_mask_ratio: $GT_MASK_MIN_RATIO"
echo "gt_mask_views: $GT_MASK_MIN_VIEWS"
echo "save_clouds:   $SAVE_EVAL_CLOUDS"

EXTRA_ARGS=()
if [ "$SAVE_EVAL_CLOUDS" = "1" ]; then
  EXTRA_ARGS+=(--save-eval-clouds)
fi
if [ "$GT_MASK_FILTER" = "1" ]; then
  EXTRA_ARGS+=(--gt-mask-filter --gt-mask-mode "$GT_MASK_MODE" --gt-mask-min-ratio "$GT_MASK_MIN_RATIO" --gt-mask-min-views "$GT_MASK_MIN_VIEWS")
fi
if [ "$SAVE_PNG" = "1" ]; then
  EXTRA_ARGS+=(--save-png)
fi
if [ "$PNG_ALIGN_PCA" = "1" ]; then
  EXTRA_ARGS+=(--png-align-pca)
fi
if [ "${PNG_ROTATE_DEG}" != "0" ]; then
  EXTRA_ARGS+=(--png-rotate-deg "$PNG_ROTATE_DEG")
fi
if [ -n "$PNG_VMAX" ]; then
  EXTRA_ARGS+=(--png-vmax "$PNG_VMAX")
fi
if [ -n "$GT_MAX_DIST_TO_PRED" ]; then
  EXTRA_ARGS+=(--gt-max-dist-to-pred "$GT_MAX_DIST_TO_PRED")
fi
if [ -n "$GT_Z_MIN" ]; then
  EXTRA_ARGS+=(--gt-z-min "$GT_Z_MIN")
fi
if [ -n "$GT_Z_MAX" ]; then
  EXTRA_ARGS+=(--gt-z-max "$GT_Z_MAX")
fi
if [ -n "$PSI" ]; then
  EXTRA_ARGS+=(--psi "$PSI")
fi

python -u analysis/compare_dtu_chamfer.py \
  --ckpt "$PT" \
  --scene "$SCENE" \
  --scan-id "$SCAN" \
  --dtu-eval-dir "$DTU_EVAL" \
  --gt-ply "$GT_PLY" \
  --res "$RES" \
  --bound "$BOUND" \
  --n-pred-points "$N_PRED_POINTS" \
  --max-dist "$MAX_DIST" \
  --device auto \
  --gt-crop "$GT_CROP" \
  --gt-crop-padding "$GT_CROP_PADDING" \
  --out "$OUT" \
  --save-pred-mesh \
  "${EXTRA_ARGS[@]}"
