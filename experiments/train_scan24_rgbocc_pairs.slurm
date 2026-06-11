#!/bin/bash
#SBATCH --job-name=1lip-s24-rgbpairs
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-rgbpairs_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-rgbpairs_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# scan24 pairs.txt twin of run_20260603_103549_scan24_rgbocc_4962427 (job 4962427):
#   EXACTLY that RGB run's config, with the only changes being the alternative-
#   view selector: view_selection nearest -> pairs_file, pairs_path -> the
#   NeuralWarp-style pair.txt (top-n_alt source views by co-visible SfM points).
#   n_alt stays 6; ncc_color stays rgb; everything else inherited verbatim.

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24
PAIRS=$SCENE/pair.txt
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"
SOURCE_CONFIG=$OUTPUTS/run_20260603_103549_scan24_rgbocc_4962427/config.json
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan24_rgbpairs_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_scan24_rgbocc_pairs.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$PAIRS" <<'PY'
import copy, json, sys
from pathlib import Path

source_path, derived_path, pairs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not Path(pairs).exists():
    raise SystemExit(f"pair file not found: {pairs}")
source = json.loads(source_path.read_text())
derived = copy.deepcopy(source)
derived.pop("cmd", None)                          # stale launch command
derived["train"]["view_selection"] = "pairs_file" # the ONLY intended changes
derived["train"]["pairs_path"]     = pairs

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
    ("train.view_selection", "nearest", "pairs_file"),
    ("train.pairs_path", None, pairs),
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
echo "Pair file:      $PAIRS"
echo "Run dir:        $RUN_DIR  (scan24, rgb NCC, pairs_file selection, otherwise == 4962427)"

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
