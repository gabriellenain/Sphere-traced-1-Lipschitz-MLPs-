#!/bin/bash
#SBATCH --job-name=dtu-gt-watertight
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_gt_watertight_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_gt_watertight_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}

PYTHON=${PYTHON:-/home/glenain/nerfstudio/.pixi/envs/default/bin/python3}
SCAN=${SCAN:-65}
DTU_EVAL=${DTU_EVAL:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data}
OUT=${OUT:-outputs/dtu_scan${SCAN}_gt_watertight.ply}
DEPTH=${DEPTH:-10}
SCALE=${SCALE:-1.08}
MAX_POINTS=${MAX_POINTS:-800000}

echo "scan:       $SCAN"
echo "python:     $PYTHON"
echo "dtu_eval:   $DTU_EVAL"
echo "out:        $OUT"
echo "depth:      $DEPTH"
echo "scale:      $SCALE"
echo "max_points: $MAX_POINTS"

"$PYTHON" -u analysis/make_dtu_gt_watertight.py \
    --scan-id "$SCAN" \
    --dtu-eval-dir "$DTU_EVAL" \
    --out "$OUT" \
    --depth "$DEPTH" \
    --scale "$SCALE" \
    --max-points "$MAX_POINTS"

LOGFILE="/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_gt_watertight_${SLURM_JOB_ID}.log"
OUTDIR=$(dirname "$OUT")
[ -f "$LOGFILE" ] && cp "$LOGFILE" "$OUTDIR/job.log"
