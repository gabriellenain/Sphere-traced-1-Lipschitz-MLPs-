#!/bin/bash
#SBATCH --job-name=mvm-official
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvm_official_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvm_official_%j.log
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

# Use a pre-extracted world mesh if MESH is set, otherwise extract from checkpoint
MESH=${MESH:-}
PT=${PT:-outputs/run_20260528_210303_mvm_kinette-cos-hx_4922534/ckpt/checkpoint_latest.pt}
SCENE=${SCENE:-/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/mvmannequin_neus/kinette-cos-hx}
RES=${RES:-512}
BOUND=${BOUND:-1.0}
OUT=${OUT:-outputs/mvm_official_${SLURM_JOB_ID}}

echo "scene: $SCENE"
echo "mesh:  ${MESH:-<extract from checkpoint>}"
echo "ckpt:  ${PT:-<none>}"
echo "out:   $OUT"

mkdir -p "$OUT"

if [ -n "$MESH" ]; then
    if [ ! -f "$MESH" ]; then
        echo "[error] mesh not found: $MESH" >&2
        exit 1
    fi
    python -u analysis/eval_mvmannequin_official.py \
        --mesh "$MESH" \
        --scene "$SCENE" \
        --out "$OUT"
else
    if [ ! -f "$PT" ]; then
        echo "[error] checkpoint not found: $PT" >&2
        exit 1
    fi
    python -u analysis/eval_mvmannequin_official.py \
        --ckpt "$PT" \
        --scene "$SCENE" \
        --res "$RES" \
        --bound "$BOUND" \
        --out "$OUT"
fi

# copy slurm log into output dir for self-contained results
LOGFILE="/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/mvm_official_${SLURM_JOB_ID}.log"
[ -f "$LOGFILE" ] && cp "$LOGFILE" "$OUT/job.log"
