#!/bin/bash
#SBATCH --job-name=render-hq
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_hq_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_hq_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu004


PT=${PT:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260510_094943_scan65/checkpoint_final.pt}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65}
OUT_DIR=${OUT_DIR:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_hq_$(basename $(dirname $PT))}
# trace params (env-overridable; defaults preserve prior behaviour). Set these to
# a run's config.json trace params when rendering that run's checkpoint.
DATASET=${DATASET:-skull}
TRACE_ITERS=${TRACE_ITERS:-256}
EPS=${EPS:-2e-4}
NEWTON=${NEWTON:-8}
T_FAR=${T_FAR:-10.0}
REF_VIEWS=${REF_VIEWS:-12}
NOVEL_VIEWS=${NOVEL_VIEWS:-24}

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "PT=$PT"
echo "OUT=$OUT_DIR"
echo "trace: dataset=$DATASET iters=$TRACE_ITERS eps=$EPS newton=$NEWTON t_far=$T_FAR ref=$REF_VIEWS novel=$NOVEL_VIEWS"

python -u -m lip_tracer.render_views \
    --pt "$PT" \
    --dataset "$DATASET" \
    --scene "$SCENE" \
    --out-dir "$OUT_DIR" \
    --trace-iters "$TRACE_ITERS" \
    --eps "$EPS" \
    --newton-steps "$NEWTON" \
    --t-far "$T_FAR" \
    --ref-views "$REF_VIEWS" \
    --novel-views "$NOVEL_VIEWS" \
    --ray-chunk 32768
