#!/bin/bash
#SBATCH --job-name=dtu-official
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_official_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_official_%j.log
# note: log is also copied into OUT dir at the end of the script
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

DTU_EVAL=${DTU_EVAL:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data}

# Use a pre-extracted world mesh if MESH is set, otherwise extract from checkpoint
MESH=${MESH:-}
PT=${PT:-outputs/run_20260519_203314_scan65/checkpoint_latest.pt}

SCAN=${SCAN:-}
if [ -z "$MESH" ] && [[ "$PT" =~ scan([0-9]+) ]]; then
    PT_SCAN="${BASH_REMATCH[1]}"
    if [ -n "$SCAN" ] && [ "$SCAN" != "$PT_SCAN" ]; then
        echo "[warn] SCAN=$SCAN conflicts with checkpoint path scan$PT_SCAN; using scan$PT_SCAN" >&2
    fi
    SCAN="$PT_SCAN"
fi
SCAN=${SCAN:-65}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}}
RES=${RES:-512}
BOUND=${BOUND:-1.0}
CURV_THRESH=${CURV_THRESH:-0.5}
CLEAN=${CLEAN:-0}                              # 1 = IDR largest-component cleaning (NeuS/VolSDF protocol)
CLEAN_ARG=""; [ "$CLEAN" = "1" ] && CLEAN_ARG="--clean-largest-component"
FG_MASK_CROP=${FG_MASK_CROP:-0}                # 1 = NeuralWarp-style dilated foreground-mask mesh crop
FG_MASK_DILATE=${FG_MASK_DILATE:-12}
FG_MASK_RATIO=${FG_MASK_RATIO:-1.0}
FG_MASK_MIN_VIEWS=${FG_MASK_MIN_VIEWS:-1}
FG_MASK_ARG=""
[ "$FG_MASK_CROP" = "1" ] && FG_MASK_ARG="--foreground-mask-crop --mask-dilate-px $FG_MASK_DILATE --mask-crop-min-ratio $FG_MASK_RATIO --mask-crop-min-views $FG_MASK_MIN_VIEWS"
TRACE_COMPARE=${TRACE_COMPARE:-0}              # 1 = also run expensive sphere-trace comparison
TRACE_ARG="--no-trace-compare"; [ "$TRACE_COMPARE" = "1" ] && TRACE_ARG=""
ERROR_VIEW=${ERROR_VIEW:-}                     # camera index N -> chamfer_error_viewN.png (image-plane projection)
EV_ARG=""; [ -n "$ERROR_VIEW" ] && EV_ARG="--error-view $ERROR_VIEW"
OUT=${OUT:-outputs/dtu_official_${SLURM_JOB_ID}}

echo "scan:     $SCAN"
echo "dtu_eval: $DTU_EVAL"
echo "mesh:     ${MESH:-<extract from checkpoint>}"
echo "ckpt:     $PT"
echo "out:      $OUT"

mkdir -p "$OUT"

if [ -n "$MESH" ]; then
    if [ ! -f "$MESH" ]; then
        echo "[error] mesh not found: $MESH" >&2
        exit 1
    fi
    python -u analysis/eval_dtu_official.py \
        --mesh "$MESH" \
        --scan-id "$SCAN" \
        --dtu-eval-dir "$DTU_EVAL" \
        --curv-thresh "$CURV_THRESH" \
        --scene "$SCENE" \
        $FG_MASK_ARG \
        $TRACE_ARG \
        $EV_ARG \
        --out "$OUT"
else
    if [ ! -f "$PT" ]; then
        echo "[error] checkpoint not found: $PT" >&2
        exit 1
    fi
    python -u analysis/eval_dtu_official.py \
        --ckpt "$PT" \
        --scene "$SCENE" \
        --scan-id "$SCAN" \
        --dtu-eval-dir "$DTU_EVAL" \
        --res "$RES" \
        --bound "$BOUND" \
        --curv-thresh "$CURV_THRESH" \
        $FG_MASK_ARG \
        $CLEAN_ARG \
        $TRACE_ARG \
        $EV_ARG \
        --out "$OUT"
fi

# copy slurm log into output dir for self-contained results
LOGFILE="/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_official_${SLURM_JOB_ID}.log"
[ -f "$LOGFILE" ] && cp "$LOGFILE" "$OUT/job.log"
