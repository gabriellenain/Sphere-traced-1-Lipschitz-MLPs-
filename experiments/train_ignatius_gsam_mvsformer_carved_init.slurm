#!/bin/bash
#SBATCH --job-name=1lip-ign-gsam-mvsfinit
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-ign-gsam-mvsfinit_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-ign-gsam-mvsfinit_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
#
# Ignatius statue run matching the bmvs_clock MVSFormer-carved-init launch:
#   outputs/run_20260604_094256_bmvs_clock_mvsformer_carved_init_4972680
#
# Uses the full low-threshold GSAM statue masks:
#   prompt = statue + book + pedestal, box_thr=0.28, text_thr=0.18
# and MVSFormer++ depths to carve the mask visual hull before training.

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs

REFERENCE_RUN=$OUTPUTS/run_20260604_094256_bmvs_clock_mvsformer_carved_init_4972680
SOURCE_CONFIG=$REFERENCE_RUN/source_config_bmvs_clock_mvsformer_carved_init.json

SRC_SCENE=$ROOT/data/tnt/Ignatius
TNT_SCENE_NAME=Ignatius
TNT_EVAL=${TNT_EVAL:-$ROOT/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}
MASK_SOURCE=$SRC_SCENE/mask_statue
STAGED_SCENE=$ROOT/_diagnostics/ignatius_gsam_statue_lowthr_scene

MVS_BASE=$ROOT/_diagnostics/mvsformer_ignatius
DEPTH_DIR=$MVS_BASE/depths_1152x640

P_THRESH=${P_THRESH:-0.90}
P_TAG=$(python -c "print(f'p{round(float(\"$P_THRESH\") * 100):03d}')")
CARVE_DIR=$ROOT/_diagnostics/mvsformer_ignatius_gsam_statue_lowthr/${P_TAG}_hull_carve
INIT_SDF=$CARVE_DIR/sdf_vh_mvsformer_carved.npy

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export HF_HOME=/scratch/_projets_/willow/1-lip-tracer-new/hf_cache
export MVSFORMER_REPO=/scratch/glenain/MVSFormerPlusPlus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-ignatius-gsam-mvsf

mkdir -p "$OUTPUTS_HOME" "$STAGED_SCENE" "$CARVE_DIR"

# Stage a scene whose mask/ is the low-threshold GSAM statue mask set, while
# preserving the original data/tnt/Ignatius/mask directory.
ln -sfn "$SRC_SCENE/rgb" "$STAGED_SCENE/rgb"
ln -sfn "$SRC_SCENE/pose" "$STAGED_SCENE/pose"
ln -sfn "$SRC_SCENE/intrinsics.txt" "$STAGED_SCENE/intrinsics.txt"
ln -sfn "$SRC_SCENE/bbox.txt" "$STAGED_SCENE/bbox.txt"
ln -sfn "$MASK_SOURCE" "$STAGED_SCENE/mask"

N_POSES=$(find "$SRC_SCENE/pose" -maxdepth 1 -type f -name '0_*.txt' | wc -l)
N_MASKS=$(find "$MASK_SOURCE" -maxdepth 1 -type f -name '0_*.png' | wc -l || true)

if [ "$N_MASKS" -lt "$N_POSES" ]; then
  echo "GSAM statue masks incomplete ($N_MASKS/$N_POSES); regenerating low-threshold masks."
  mkdir -p "$MASK_SOURCE"
  python -u tools/grounded_sam_segment.py \
    --image-dir "$SRC_SCENE/rgb" \
    --out-dir "$MASK_SOURCE" \
    --prompt "statue." --prompt "book." --prompt "pedestal." \
    --box-threshold 0.28 --text-threshold 0.18 \
    --keep-top 1 --largest-cc --fill-holes \
    --overview-views 0 52 105 158 210 262
fi

if [ ! -d "$DEPTH_DIR/Ignatius/depth_est" ]; then
  echo "MVSFormer++ depths missing; precomputing -> $DEPTH_DIR"
  python -u tools/precompute_mvsformer_depths.py \
    --scene "$SRC_SCENE" \
    --scan-name Ignatius \
    --out-dir "$MVS_BASE/depths" \
    --staging "$MVS_BASE/staging" \
    --max-h 640 --max-w 1152 \
    --num-view 5 --n-src 10 \
    --numdepth 128 \
    --keep-staging
fi

if [ ! -f "$INIT_SDF" ]; then
  echo "Carving GSAM-masked MVSFormer++ hull -> $INIT_SDF"
  python -u tools/carve_visual_hull_with_mvsformer_scan24.py \
    --scene "$STAGED_SCENE" \
    --depth-dir "$DEPTH_DIR" \
    --out-dir "$CARVE_DIR" \
    --res 256 --bound 1.5 \
    --conf-thr 0.5 --votes-req 3 \
    --vh-percentile "$P_THRESH" --vh-min-views 8
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_ignatius_gsamlow_${P_TAG}_mvsformer_carved_init_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_ignatius_gsamlow_mvsformer_carved_init.json
LOG=$RUN_DIR/train.log
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

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
    raise SystemExit("reference run is not hull init")
if not Path(init_sdf).exists():
    raise SystemExit(f"missing carved SDF grid: {init_sdf}")

derived["scene"] = scene
derived.setdefault("init", {})["init_sdf_grid"] = init_sdf

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
echo "VH percentile:   $P_THRESH"
echo "Init SDF grid:   $INIT_SDF"
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
