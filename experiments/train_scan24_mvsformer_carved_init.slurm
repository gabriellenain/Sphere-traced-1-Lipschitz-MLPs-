#!/bin/bash
#SBATCH --job-name=1lip-s24-mvsfinit
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-mvsfinit_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-mvsfinit_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
REFERENCE_RUN=$OUTPUTS/run_20260603_141206_scan24_4963407
SOURCE_CONFIG=$REFERENCE_RUN/config.json
INIT_SDF=$ROOT/_diagnostics/mvsformer_scan24_idr/hull_carve/sdf_vh_mvsformer_carved.npy
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan24_mvsformer_carved_init_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_scan24_mvsformer_carved_init.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$INIT_SDF" <<'PY'
import copy, json, sys
from pathlib import Path

source_path, derived_path, init_sdf = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
source = json.loads(source_path.read_text())
derived = copy.deepcopy(source)
derived.pop("cmd", None)

if derived.get("init", {}).get("init") != "hull":
    raise SystemExit(f"reference run is not hull init: {derived.get('init', {}).get('init')!r}")
if not Path(init_sdf).exists():
    raise SystemExit(f"missing MVSFormer++ carved SDF grid: {init_sdf}")

derived.setdefault("init", {})["init_sdf_grid"] = init_sdf

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
    ("init.init_sdf_grid", source.get("init", {}).get("init_sdf_grid"), init_sdf),
]
if sorted(changes) != sorted(expected):
    raise SystemExit(f"unexpected config changes:\n{changes!r}\nexpected:\n{expected!r}")

derived_path.write_text(json.dumps(derived, indent=2) + "\n")
print("Derived config differs only at:")
for c in sorted(expected):
    print("  ", c)
PY

echo "Reference run:  $REFERENCE_RUN"
echo "Init SDF grid:  $INIT_SDF"
echo "Derived config: $DERIVED_CONFIG"
echo "Run dir:        $RUN_DIR"

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
