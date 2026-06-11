#!/bin/bash
#SBATCH --job-name=1lip-s118-occ
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s118-occ_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s118-occ_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

# scan118 run with EXACTLY the same params as the scan122 coarse-occ run
# (run_20260602_211933_scan122_gray_occ_4954503): same train hyperparams,
# same coarse occlusion trace (occ 32 / 0 / 1e-3). Scene -> scan118, and the
# hull init matches run_20260602_134230_scan118_4951319: hull_border_aware=True
# (the scan118 hull-bloat fix). Everything else unchanged from the occ run.

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
SRC_CONFIG=$OUTPUTS/run_20260602_211933_scan122_gray_occ_4954503/config.json
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan118
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan118_occ_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS/1lip-s118-occ_${SLURM_JOB_ID}.rundir"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/1lip-s118-occ_${SLURM_JOB_ID}.rundir"

# Clone the occ-run config, repoint scene to scan118, drop the scan122-specific
# (and w_feature=0, so unused) feature_maps path.
python - "$SRC_CONFIG" "$DERIVED_CONFIG" "$SCENE" <<'PY'
import json, sys
from pathlib import Path
src, dst, scene = sys.argv[1:]
cfg = json.loads(Path(src).read_text())
cfg["scene"] = scene
cfg["train"]["feature_maps"] = None
# init from run_20260602_134230_scan118_4951319: only diff is border-aware hull.
cfg["init"]["hull_border_aware"] = True
Path(dst).write_text(json.dumps(cfg, indent=2) + "\n")
print(f"scene -> {scene}")
print(f"feature_maps -> None  (w_feature={cfg['train']['w_feature']})")
print(f"init.hull_border_aware -> {cfg['init']['hull_border_aware']}")
print(f"occ trace: iters={cfg['trace']['occ_iters']} newton={cfg['trace']['occ_newton_steps']} eps={cfg['trace']['occ_eps']}")
print(f"batch={cfg['train']['batch']} compile={cfg['train']['compile']} ncc_color={cfg['train']['ncc_color']}")
PY

echo "Source config:  $SRC_CONFIG"
echo "Derived config: $DERIVED_CONFIG"
echo "Run dir:        $RUN_DIR"
echo "Logging to:     $LOG"

# occ_* and dtu_official_* must be passed on the CLI: the --config override path
# always takes the arg value for these (defaults would otherwise clobber the
# config's occ=32/0/1e-3 back to 64/2/1e-4 and disable the official eval).
python -u -m lip_tracer.train \
    --config "$DERIVED_CONFIG" \
    --run-dir "$RUN_DIR" \
    --occ-iters 32 \
    --occ-newton-steps 0 \
    --occ-eps 1e-3 \
    --dtu-eval-dir "$DTU_EVAL" \
    --dtu-official-freq 10000 \
    --dtu-official-res 384 \
    --dtu-official-bound 1.0 \
    --wandb 2>&1 | tee "$LOG"
