#!/bin/bash
#SBATCH --job-name=dtu-levels
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_levels_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/dtu_levels_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu004

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

SCAN=${SCAN:-65}
DTU_EVAL=${DTU_EVAL:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data}
PT=${PT:-$ROOT/outputs/run_20260515_212859_scan65/checkpoint_latest.pt}
SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan${SCAN}}
LEVELS=${LEVELS:--0.005,0.000,0.0025,0.005,0.0075,0.010}
RES=${RES:-512}
BOUND=${BOUND:-1.0}
DOWN=${DOWN:-1}
SS=${SS:-1}
N_VIEWS=${N_VIEWS:-5}
VIEWS=${VIEWS:-}
OUT=${OUT:-$ROOT/outputs/dtu_level_sweep_scan${SCAN}_${SLURM_JOB_ID}}

echo "scan:     $SCAN"
echo "ckpt:     $PT"
echo "scene:    $SCENE"
echo "dtu_eval: $DTU_EVAL"
echo "levels:   $LEVELS"
echo "out:      $OUT"

view_args=()
if [[ -n "$VIEWS" ]]; then
  view_args=(--views "$VIEWS")
else
  view_args=(--n-views "$N_VIEWS")
fi

python -u analysis/eval_dtu_level_sweep.py \
  --ckpt "$PT" \
  --scene "$SCENE" \
  --scan-id "$SCAN" \
  --dtu-eval-dir "$DTU_EVAL" \
  --levels="$LEVELS" \
  --res "$RES" \
  --bound "$BOUND" \
  --down "$DOWN" \
  --ss "$SS" \
  --out "$OUT" \
  "${view_args[@]}"

LOGFILE="$ROOT/outputs/dtu_levels_${SLURM_JOB_ID}.log"
[ -f "$LOGFILE" ] && cp "$LOGFILE" "$OUT/job.log"
