#!/bin/bash
#SBATCH --job-name=1lip-clock-mvsfinit
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-clock-mvsfinit_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-clock-mvsfinit_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
# bmvs_clock counterpart of train_scan24_mvsformer_carved_init.slurm.
# Reuses that run's exact model/trace/train config; only the scene, the carved
# init-SDF grid, and the DTU-only feature maps differ. bmvs ships no COLMAP
# points, so the carve runs with no SFM ROI / clip (edge-aware VH + largest CC).
#
# Prereq: precompute_mvsformer_depths_bmvs_clock.slurm has produced depths under
#   _diagnostics/mvsformer_bmvs_clock/depths/bmvs_clock/{depth_est,confidence}

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
REFERENCE_RUN=$OUTPUTS/run_20260603_180638_scan24_mvsformer_carved_init_4965606
SOURCE_CONFIG=$REFERENCE_RUN/source_config_scan24_mvsformer_carved_init.json
# Scene-parameterized (default keeps the original bmvs_clock behaviour):
#   BMVS=bmvs_sculpture sbatch experiments/train_bmvs_clock_mvsformer_carved_init.slurm
BMVS=${BMVS:-bmvs_clock}
RES_TAG=${RES_TAG:-768x576}
SCENE=$ROOT/data/bmvs/$BMVS
# MVSFormer++ analysis/test.py appends the resolution to --out-dir (depths -> depths_768x576)
DEPTH_DIR=$ROOT/_diagnostics/mvsformer_${BMVS}/depths_${RES_TAG}
CARVE_DIR=$ROOT/_diagnostics/mvsformer_${BMVS}/hull_carve
INIT_SDF=$CARVE_DIR/sdf_vh_mvsformer_carved.npy

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-carve-${BMVS}

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_${BMVS}_mvsformer_carved_init_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_${BMVS}_mvsformer_carved_init.json
LOG=$RUN_DIR/train.log
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

# --- 1. carve the MVSFormer++-refined visual hull (skips if already built) ----
if [ ! -f "$INIT_SDF" ]; then
  if [ ! -d "$DEPTH_DIR" ]; then
    echo "ERROR: depths missing at $DEPTH_DIR — run precompute_mvsformer_depths_bmvs_clock.slurm first" >&2
    exit 1
  fi
  echo "Carving MVSFormer++ hull -> $INIT_SDF"
  python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
    --scene "$SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$CARVE_DIR" \
    --res 256 --bound 1.5 \
    --conf-thr 0.5 --votes-req 3
fi

# --- 2. derive the training config from the scan24 reference run --------------
python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$SCENE" "$INIT_SDF" <<'PY'
import copy, json, sys
from pathlib import Path

source_path, derived_path, scene, init_sdf = (Path(sys.argv[1]), Path(sys.argv[2]),
                                              sys.argv[3], sys.argv[4])
source = json.loads(source_path.read_text())
derived = copy.deepcopy(source)
derived.pop("cmd", None)

if derived.get("init", {}).get("init") != "hull":
    raise SystemExit("reference run is not hull init")
if not Path(init_sdf).exists():
    raise SystemExit(f"missing carved SDF grid: {init_sdf}")

derived["scene"] = scene
derived.setdefault("init", {})["init_sdf_grid"] = init_sdf
derived["train"]["feature_maps"] = None  # DTU-only MASt3R maps (w_feature=0 anyway)

def flatten(v, p=""):
    if isinstance(v, dict):
        out = {}
        for k, it in v.items():
            out.update(flatten(it, f"{p}.{k}" if p else k))
        return out
    return {p: v}

before, after = flatten(source), flatten(derived)
changes = sorted((k, before.get(k), after.get(k))
                 for k in before.keys() | after.keys()
                 if before.get(k) != after.get(k))
expected = sorted([
    ("scene", source.get("scene"), scene),
    ("init.init_sdf_grid", source["init"].get("init_sdf_grid"), init_sdf),
    ("train.feature_maps", source["train"].get("feature_maps"), None),
])
if changes != expected:
    raise SystemExit(f"unexpected config changes:\n{changes!r}\nexpected:\n{expected!r}")

derived_path.write_text(json.dumps(derived, indent=2) + "\n")
print("Derived config differs only at:")
for c in expected:
    print("  ", c)
PY

echo "Reference run:  $REFERENCE_RUN"
echo "Scene:          $SCENE"
echo "Init SDF grid:  $INIT_SDF"
echo "Run dir:        $RUN_DIR"

# --- 3. train (same occ-trace overrides as scan24; no DTU eval for bmvs) ------
python -u -m lip_tracer.train \
    --config "$DERIVED_CONFIG" \
    --run-dir "$RUN_DIR" \
    --occ-iters 32 \
    --occ-newton-steps 0 \
    --occ-eps 1e-3 \
    --wandb 2>&1 | tee "$LOG"
