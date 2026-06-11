#!/bin/bash
#SBATCH --job-name=1lip-s122-gray-occ
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s122-gray-occ_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s122-gray-occ_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

# Comparison run for run_20260602_173124_scan122_gray_4952656.
# Identical config (scan122, ncc_color=gray, same source) EXCEPT the photo-loss
# occlusion trace is coarsened: 32 iters + 0 Newton + eps 1e-3 (visibility boolean
# does not need primary-trace precision; looser eps lets rays drop out of the
# compacted trace sooner). Everything else is matched so the official DTU
# Chamfer @10k is directly comparable.

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
SOURCE_CONFIG=$OUTPUTS/run_20260525_223035_scan122_4911693/config.json
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan122_gray_occ_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_gray.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS/1lip-s122-gray-occ_${SLURM_JOB_ID}.rundir"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/1lip-s122-gray-occ_${SLURM_JOB_ID}.rundir"

python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" <<'PY'
import copy
import json
import sys
from pathlib import Path

source_path, derived_path = map(Path, sys.argv[1:])
source = json.loads(source_path.read_text())
derived = copy.deepcopy(source)
derived["train"]["ncc_color"] = "gray"

def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: value}

before = flatten(source)
after = flatten(derived)
changes = [(key, before.get(key), after.get(key))
           for key in sorted(before.keys() | after.keys())
           if before.get(key) != after.get(key)]
expected = [("train.ncc_color", "rgb", "gray")]
if changes != expected:
    raise SystemExit(f"unexpected config changes: {changes!r}")
derived_path.write_text(json.dumps(derived, indent=2) + "\n")
print(f"Derived config differs only at: {changes[0]}")
PY

echo "Source config:  $SOURCE_CONFIG"
echo "Derived config: $DERIVED_CONFIG"
echo "Run dir:        $RUN_DIR"
echo "Logging to:     $LOG"
echo "Occlusion trace: --occ-iters 32 --occ-newton-steps 0 --occ-eps 1e-3 (baseline used 64 / 2 / 1e-4)"

python -u -m lip_tracer.train \
    --config "$DERIVED_CONFIG" \
    --run-dir "$RUN_DIR" \
    --occ-iters 32 \
    --occ-newton-steps 0 \
    --occ-eps 1e-3 \
    --dtu-eval-dir "/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data" \
    --dtu-official-freq 10000 \
    --dtu-official-res 384 \
    --dtu-official-bound 1.0 \
    --wandb 2>&1 | tee "$LOG"
