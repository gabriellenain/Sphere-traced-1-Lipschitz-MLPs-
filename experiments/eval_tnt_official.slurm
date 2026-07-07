#!/bin/bash
#SBATCH --job-name=tnt-official
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/tnt_official_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/tnt_official_%j.log
# note: log is also copied into OUT dir at the end of the script
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,willow
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

# --- inputs -------------------------------------------------------------
# Set MESH=... to evaluate a pre-extracted mesh OR point cloud (e.g. COLMAP
# fused.ply / poisson.ply); otherwise PT is used and we MC-extract the SDF.
MESH=${MESH:-}
PT=${PT:-outputs/run_20260528_084156_tnt_Barn_4918116/ckpt/checkpoint_best_photo.pt}
SCENE=${SCENE:-Barn}
SCENE_DIR=${SCENE_DIR:-data/tnt/${SCENE}}
# GT assets downloaded from tanksandtemples.org (Training set + auxiliary).
# Expect under $GT_DIR:  ${SCENE}.ply  ${SCENE}_trans.txt  ${SCENE}.json
GT_DIR=${GT_DIR:-data/tnt_gt/${SCENE}}
RES=${RES:-2048}
BOUND=${BOUND:-1.5}
N_SAMPLES=${N_SAMPLES:-2000000}
OUT=${OUT:-outputs/tnt_official_${SLURM_JOB_ID}}
FRAME=${FRAME:-colmap-pose}

echo "scene:     $SCENE"
echo "scene_dir: $SCENE_DIR"
echo "gt_dir:    $GT_DIR"
echo "mesh:      ${MESH:-<extract from checkpoint>}"
echo "ckpt:      ${PT}"
echo "frame:     $FRAME"
echo "out:       $OUT"

if [ -n "$MESH" ]; then
    if [ ! -f "$MESH" ]; then
        echo "[error] mesh not found: $MESH" >&2
        exit 1
    fi
else
    if [ ! -f "$PT" ]; then
        echo "[error] checkpoint not found: $PT" >&2
        exit 1
    fi
fi
for f in "${GT_DIR}/${SCENE}.ply" "${GT_DIR}/${SCENE}_trans.txt" "${GT_DIR}/${SCENE}.json"; do
    if [ ! -f "$f" ]; then
        echo "[error] GT asset missing: $f" >&2
        echo "        download from https://www.tanksandtemples.org/download/" >&2
        exit 1
    fi
done

mkdir -p "$OUT"

if [ -n "$MESH" ]; then
    # FRAME=colmap-pose for meshes/checkpoints in the raw pose/*.txt frame
    # (current COLMAP-derived TnT scenes); FRAME=colmap-local for COLMAP MVS
    # outputs built from LOCAL sparse/0; FRAME=colmap-sfm only if already in
    # the shipped COLMAP-SfM frame.
    python -u analysis/eval_tnt_official.py \
        --mesh      "$MESH" \
        --scene     "$SCENE" \
        --scene-dir "$SCENE_DIR" \
        --gt-dir    "$GT_DIR" \
        --n-samples "$N_SAMPLES" \
        --frame     "$FRAME" \
        --out       "$OUT"
else
    python -u analysis/eval_tnt_official.py \
        --ckpt      "$PT" \
        --scene     "$SCENE" \
        --scene-dir "$SCENE_DIR" \
        --gt-dir    "$GT_DIR" \
        --mc-res    "$RES" \
        --bound     "$BOUND" \
        --n-samples "$N_SAMPLES" \
        --frame     "$FRAME" \
        --out       "$OUT"
fi

# DTU-style error PNGs (chamfer_error / worst10 / threshold) from the
# cropped+aligned clouds the toolbox just wrote ($OUT/<scene>.{precision,recall}.ply)
python -u analysis/tnt_error_pngs.py "$OUT" --scene "$SCENE" || echo "[warn] error-png step failed"

# copy slurm log into output dir for self-contained results
LOGFILE="/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/tnt_official_${SLURM_JOB_ID}.log"
[ -f "$LOGFILE" ] && cp "$LOGFILE" "$OUT/job.log"
