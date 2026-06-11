#!/bin/bash
#SBATCH --job-name=1lip-s24-rgbocc
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-rgbocc_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-rgbocc_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# scan24 RGB twin of run_20260603_093037_scan24_4962197 (job 4962197):
#   EXACTLY that run's config, with the single change ncc_color gray -> rgb.
#   Everything else (border-aware hull init, occlusion tracing 32/0/1e-3,
#   batch 16384, trace eps 1e-4, 300k steps, dtu-official eval) is inherited
#   verbatim from that run's saved config.json.

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"
SOURCE_CONFIG=$OUTPUTS/run_20260603_093037_scan24_4962197/config.json
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan24_rgbocc_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_scan24_rgbocc_border.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" <<'PY'
import copy, json, sys
from pathlib import Path

source_path, derived_path = Path(sys.argv[1]), Path(sys.argv[2])
source = json.loads(source_path.read_text())
derived = copy.deepcopy(source)
derived.pop("cmd", None)                      # stale launch command
derived["train"]["ncc_color"] = "rgb"         # the ONLY intended change

def flatten(v, p=""):
    if isinstance(v, dict):
        out = {}
        for k, it in v.items():
            out.update(flatten(it, f"{p}.{k}" if p else k))
        return out
    return {p: v}

before, after = flatten(source), flatten(derived)
changes = [(k, before.get(k), after.get(k))
           for k in sorted(before.keys() | after.keys())
           if before.get(k) != after.get(k)]
expected = [
    ("cmd", source.get("cmd"), None),
    ("train.ncc_color", "gray", "rgb"),
]
if sorted(changes) != sorted(expected):
    raise SystemExit(f"unexpected config changes:\n{changes!r}\nexpected:\n{expected!r}")
derived_path.write_text(json.dumps(derived, indent=2) + "\n")
print("Derived config differs only at:")
for c in sorted(expected):
    print("  ", c)
PY

echo "Source config:  $SOURCE_CONFIG"
echo "Derived config: $DERIVED_CONFIG"
echo "Run dir:        $RUN_DIR  (scan24, rgb NCC, otherwise == 4962197)"

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
