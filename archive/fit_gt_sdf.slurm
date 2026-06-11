#!/bin/bash
#SBATCH --job-name=gt-sdf-fit
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/gt_sdf_fit_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/gt_sdf_fit_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu004,gpu011,gpu014

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

MESH=${MESH:-}
OUT_DIR=${OUT_DIR:-outputs/gt_sdf_fit}
STEPS=${STEPS:-10000}
BATCH=${BATCH:-8192}
LR=${LR:-0.001}
BOUND=${BOUND:-1.5}
NEAR_STD=${NEAR_STD:-0.02}
N_NEAR=${N_NEAR:-200000}
N_VOL=${N_VOL:-200000}
NEAR_FRAC=${NEAR_FRAC:-0.7}
SDF_CHUNK=${SDF_CHUNK:-20000}
MC_RES=${MC_RES:-256}
HIDDEN=${HIDDEN:-512}
DEPTH=${DEPTH:-24}
GROUP_SIZE=${GROUP_SIZE:-2}
ACTIVATION=${ACTIVATION:-groupsort}
INPUT_ENCODING=${INPUT_ENCODING:-identity}
MULTIRES=${MULTIRES:-4}
SEED=${SEED:-0}

if [[ -z "$MESH" ]]; then
    echo "ERROR: set MESH=/path/to/gt_mesh.ply or .obj"
    exit 2
fi

echo "mesh: $MESH"
echo "out_dir: $OUT_DIR"
echo "steps: $STEPS batch: $BATCH lr: $LR"
echo "model: hidden=$HIDDEN depth=$DEPTH activation=$ACTIVATION enc=$INPUT_ENCODING multires=$MULTIRES"

python -u fit_gt_sdf.py \
    --mesh "$MESH" \
    --out-dir "$OUT_DIR" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --bound "$BOUND" \
    --near-std "$NEAR_STD" \
    --n-near "$N_NEAR" \
    --n-vol "$N_VOL" \
    --near-frac "$NEAR_FRAC" \
    --sdf-chunk "$SDF_CHUNK" \
    --mc-res "$MC_RES" \
    --hidden "$HIDDEN" \
    --depth "$DEPTH" \
    --group-size "$GROUP_SIZE" \
    --activation "$ACTIVATION" \
    --input-encoding "$INPUT_ENCODING" \
    --multires "$MULTIRES" \
    --seed "$SEED"
