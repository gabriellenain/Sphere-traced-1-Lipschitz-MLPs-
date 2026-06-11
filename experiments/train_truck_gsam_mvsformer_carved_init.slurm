#!/bin/bash
#SBATCH --job-name=1lip-truck-mvsfinit
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-truck-mvsfinit_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-truck-mvsfinit_%j.slurm.log
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

REFERENCE_RUN=$OUTPUTS/run_20260603_180638_scan24_mvsformer_carved_init_4965606
SOURCE_CONFIG=$REFERENCE_RUN/source_config_scan24_mvsformer_carved_init.json

SRC_SCENE=$ROOT/data/tnt/Truck
TNT_SCENE_NAME=Truck
TNT_EVAL=${TNT_EVAL:-$ROOT/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}
MASK_SOURCE=$SRC_SCENE/mask_truck
STAGED_SCENE=$ROOT/_diagnostics/truck_gsam_scene

DEPTH_DIR=$ROOT/_diagnostics/mvsformer_truck/depths_1152x640
CARVE_DIR=$ROOT/_diagnostics/mvsformer_truck_gsam/hull_carve
INIT_SDF=$CARVE_DIR/sdf_vh_mvsformer_carved.npy

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_truck_gsam_mvsformer_carved_init_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_truck_gsam_mvsformer_carved_init.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-carve-truck-mvsf

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME" "$STAGED_SCENE" "$CARVE_DIR"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

# Stage Truck with the GSAM truck masks as scene/mask, leaving data/tnt/Truck
# untouched for other experiments.
ln -sfn "$SRC_SCENE/rgb" "$STAGED_SCENE/rgb"
ln -sfn "$SRC_SCENE/pose" "$STAGED_SCENE/pose"
ln -sfn "$SRC_SCENE/intrinsics.txt" "$STAGED_SCENE/intrinsics.txt"
ln -sfn "$SRC_SCENE/bbox.txt" "$STAGED_SCENE/bbox.txt"
ln -sfn "$MASK_SOURCE" "$STAGED_SCENE/mask"

N_POSES=$(find "$SRC_SCENE/pose" -maxdepth 1 -type f -name '0_*.txt' | wc -l)
N_MASKS=$(find "$MASK_SOURCE" -maxdepth 1 -type f -name '0_*.png' | wc -l || true)
N_DEPTHS=$(find "$DEPTH_DIR/Truck/depth_est" -maxdepth 1 -type f -name '*.pfm' 2>/dev/null | wc -l || true)

if [ "$N_MASKS" -lt "$N_POSES" ]; then
  echo "ERROR: expected full GSAM Truck masks at $MASK_SOURCE ($N_MASKS/$N_POSES)" >&2
  exit 1
fi
if [ "$N_DEPTHS" -lt "$N_POSES" ]; then
  echo "ERROR: expected full MVSFormer++ Truck depths at $DEPTH_DIR ($N_DEPTHS/$N_POSES)" >&2
  exit 1
fi

if [ ! -f "$INIT_SDF" ]; then
  echo "Carving GSAM-masked MVSFormer++ Truck hull -> $INIT_SDF"
  python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
    --scene "$STAGED_SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$CARVE_DIR" \
    --res 256 --bound 1.5 \
    --conf-thr 0.5 --votes-req 3
fi

python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$STAGED_SCENE" "$INIT_SDF" <<'PY'
import copy, json, sys
from pathlib import Path

source_path, derived_path, scene, init_sdf = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
)
source = json.loads(source_path.read_text())
derived = copy.deepcopy(source)
derived.pop("cmd", None)

if derived.get("init", {}).get("init") != "hull":
    raise SystemExit(f"reference run is not hull init: {derived.get('init', {}).get('init')!r}")
if not Path(init_sdf).exists():
    raise SystemExit(f"missing MVSFormer++ carved SDF grid: {init_sdf}")

derived["scene"] = scene
derived.setdefault("init", {})["init_sdf_grid"] = init_sdf
derived["train"]["feature_maps"] = None

def flatten(v, p=""):
    if isinstance(v, dict):
        out = {}
        for k, it in v.items():
            out.update(flatten(it, f"{p}.{k}" if p else k))
        return out
    return {p: v}

before, after = flatten(source), flatten(derived)
changes = sorted(
    (k, before.get(k), after.get(k))
    for k in before.keys() | after.keys()
    if k != "cmd" and before.get(k) != after.get(k)
)
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
echo "Source scene:    $SRC_SCENE"
echo "Staged scene:    $STAGED_SCENE"
echo "GSAM masks:      $MASK_SOURCE"
echo "MVSFormer maps:  $DEPTH_DIR"
echo "Init SDF grid:   $INIT_SDF"
echo "Derived config:  $DERIVED_CONFIG"
echo "Run dir:         $RUN_DIR"
echo "Official TnT eval: $TNT_SCENE_NAME every ${TNT_OFFICIAL_FREQ} steps  res=${TNT_OFFICIAL_RES}  n=${TNT_OFFICIAL_N_SAMPLES}  frame=${TNT_OFFICIAL_FRAME}"

python -u -m lip_tracer.train \
    --config "$DERIVED_CONFIG" \
    --run-dir "$RUN_DIR" \
    --occ-iters 32 \
    --occ-newton-steps 0 \
    --occ-eps 1e-3 \
    --tnt-eval-dir "$TNT_EVAL" \
    --tnt-official-scene "$TNT_SCENE_NAME" \
    --tnt-official-frame "$TNT_OFFICIAL_FRAME" \
    --tnt-official-freq "$TNT_OFFICIAL_FREQ" \
    --tnt-official-res "$TNT_OFFICIAL_RES" \
    --tnt-official-n-samples "$TNT_OFFICIAL_N_SAMPLES" \
    --wandb 2>&1 | tee "$LOG"
