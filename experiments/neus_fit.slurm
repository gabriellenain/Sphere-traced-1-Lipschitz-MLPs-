#!/bin/bash
#SBATCH --job-name=neus-fit
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/neus_fit_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/neus_fit_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-

# ---- edit these, or override them at submit time:
#   SCAN=122 STEPS=100000 sbatch neus_fit.slurm
SCAN=${SCAN:-122}
DEFAULT_MESH=$ROOT/neus_100k_meshes/scan${SCAN}/sdf_marching_cubes_mesh.ply
if [[ "$SCAN" = "122" && -f "$ROOT/scan122_paper_like_clean_largest_512.ply" ]]; then
  DEFAULT_MESH=$ROOT/scan122_paper_like_clean_largest_512.ply
fi
MESH=${MESH:-$DEFAULT_MESH}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan${SCAN}}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/cpl_pe_from_neus_scan${SCAN}_${SLURM_JOB_ID}}

# ---- CPL+PE model params
HIDDEN=${HIDDEN:-256}
DEPTH=${DEPTH:-24}
GROUP_SIZE=${GROUP_SIZE:-2}
ACTIVATION=${ACTIVATION:-groupsort}
INPUT_ENCODING=${INPUT_ENCODING:-pe}
MULTIRES=${MULTIRES:-6}

# ---- distillation params
STEPS=${STEPS:-50000}
BATCH=${BATCH:-8192}
LR=${LR:-1e-3}
BOUND=${BOUND:-1.0}
NEAR_STD=${NEAR_STD:-0.005}
N_SURFACE=${N_SURFACE:-300000}
N_NEAR=${N_NEAR:-1000000}
N_UNIFORM=${N_UNIFORM:-0}
SDF_CHUNK=${SDF_CHUNK:-20000}
TARGET_MODE=${TARGET_MODE:-normal-offset}
PREVIEW_MAX_FACES=${PREVIEW_MAX_FACES:-0}
W_SURFACE=${W_SURFACE:-10.0}
W_EIKONAL=${W_EIKONAL:-0.1}
LOG_EVERY=${LOG_EVERY:-100}
SEED=${SEED:-0}

echo "=== NeuS mesh -> CPL+PE distillation ==="
echo "scan:           $SCAN"
echo "mesh:           $MESH"
echo "scene:          $SCENE"
echo "out_dir:        $OUT_DIR"
echo "hidden/depth:   $HIDDEN / $DEPTH"
echo "encoding:       $INPUT_ENCODING multires=$MULTIRES"
echo "steps/batch/lr: $STEPS / $BATCH / $LR"
echo "bound:          $BOUND"
echo "near_std:       $NEAR_STD"
echo "target_mode:    $TARGET_MODE"
echo "preview_faces:  $PREVIEW_MAX_FACES"
echo "samples:        surface=$N_SURFACE near=$N_NEAR uniform=$N_UNIFORM"
echo "weights:        surface=$W_SURFACE eikonal=$W_EIKONAL"
echo "log_every:      $LOG_EVERY"

python -u experiments/NeuS_fit.py \
  --mesh "$MESH" \
  --scene "$SCENE" \
  --out-dir "$OUT_DIR" \
  --hidden "$HIDDEN" \
  --depth "$DEPTH" \
  --group-size "$GROUP_SIZE" \
  --activation "$ACTIVATION" \
  --input-encoding "$INPUT_ENCODING" \
  --multires "$MULTIRES" \
  --steps "$STEPS" \
  --batch "$BATCH" \
  --lr "$LR" \
  --bound "$BOUND" \
  --near-std "$NEAR_STD" \
  --n-surface "$N_SURFACE" \
  --n-near "$N_NEAR" \
  --n-uniform "$N_UNIFORM" \
  --sdf-chunk "$SDF_CHUNK" \
  --target-mode "$TARGET_MODE" \
  --preview-max-faces "$PREVIEW_MAX_FACES" \
  --w-surface "$W_SURFACE" \
  --w-eikonal "$W_EIKONAL" \
  --log-every "$LOG_EVERY" \
  --seed "$SEED"
