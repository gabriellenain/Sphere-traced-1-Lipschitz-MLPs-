#!/bin/bash
#SBATCH --job-name=1lip-s24-grayocc
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-grayocc_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-s24-grayocc_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# scan24 with EXACTLY the scan122_gray_occ config (job 4954503):
#   gray NCC (patch 5), occlusion tracing (occ-iters 32 / newton 0 / eps 1e-3),
#   batch 16384, trace eps 1e-4, 300k steps.
# Only two intended changes: scene -> scan24, and hull init -> border-aware
# (the good init that works well on scan24). feature_maps repointed to scan24
# (unused: w_feature=0, only loaded when w_feature>0).

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
SCENE=/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24
DTU_EVAL="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data"
SOURCE_CONFIG=$OUTPUTS/run_20260602_211933_scan122_gray_occ_4954503/config.json
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_scan24_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_scan24_grayocc_border.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$SCENE" <<'PY'
import copy, json, sys
from pathlib import Path

source_path, derived_path, scene = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
source = json.loads(source_path.read_text())
derived = copy.deepcopy(source)
derived.pop("cmd", None)                              # stale launch command
derived["scene"] = scene
derived["init"]["hull_border_aware"] = True
derived["init"]["hull_sfm_roi"] = True
derived["train"]["feature_maps"] = "outputs/scan24_mast3r_features_idr.pt"

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
    ("init.hull_border_aware", False, True),
    ("init.hull_sfm_roi", False, True),
    ("scene",
     "/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan122", scene),
    ("train.feature_maps",
     "outputs/scan122_mast3r_features_idr.pt",
     "outputs/scan24_mast3r_features_idr.pt"),
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
echo "Run dir:        $RUN_DIR  (scan24, gray_occ config, border-aware init)"

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
