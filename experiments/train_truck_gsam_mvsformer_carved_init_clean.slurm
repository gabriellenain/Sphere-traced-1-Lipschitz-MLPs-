#!/bin/bash
#SBATCH --job-name=1lip-truck-mvsfinit-clean
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-truck-mvsfinit-clean_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-truck-mvsfinit-clean_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

# Relaunch of run_20260604_225002_truck_gsam_mvsformer_carved_init_4978196 (cancelled).
# SAME params and SAME scene PATH (_diagnostics/truck_gsam_scene) -- but that staged
# scene's symlinks now resolve to the CLEAN full-res COLMAP Truck (rgb/pose/masks)
# after the data swap. Only change vs the cancelled run: init SDF grid -> the clean
# p=0.85 SFM-ROI-clipped MVSFormer-carved hull.

set -euo pipefail

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs

PREV_RUN=$OUTPUTS/run_20260604_225002_truck_gsam_mvsformer_carved_init_4978196
SOURCE_CONFIG=$PREV_RUN/source_config_truck_gsam_mvsformer_carved_init.json
NEW_INIT_SDF=$ROOT/_diagnostics/mvsformer_truck_gsam_clean/hull_carve_p085_gentle/sdf_vh_mvsformer_carved.npy
TNT_SCENE_NAME=Truck
TNT_EVAL=${TNT_EVAL:-$ROOT/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_truck_gsam_mvsformer_carved_init_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_truck_gsam_mvsformer_carved_init.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-truck-mvsf-clean

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

if [ ! -f "$NEW_INIT_SDF" ]; then
  echo "ERROR: missing clean init SDF grid: $NEW_INIT_SDF" >&2
  exit 1
fi

# Derive config: copy the cancelled run's source config, change ONLY init.init_sdf_grid.
python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$NEW_INIT_SDF" <<'PY'
import copy, json, sys
from pathlib import Path
src, dst, init_sdf = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cfg = json.loads(src.read_text())
cfg.pop("cmd", None)
out = copy.deepcopy(cfg)
if out.get("init", {}).get("init") != "hull":
    raise SystemExit(f"reference run is not hull init: {out.get('init',{}).get('init')!r}")
old = out["init"].get("init_sdf_grid")
out["init"]["init_sdf_grid"] = init_sdf

def flatten(v, p=""):
    if isinstance(v, dict):
        o = {}
        for k, it in v.items():
            o.update(flatten(it, f"{p}.{k}" if p else k))
        return o
    return {p: v}

before, after = flatten(cfg), flatten(out)
changes = sorted((k, before.get(k), after.get(k))
                 for k in before.keys() | after.keys()
                 if k != "cmd" and before.get(k) != after.get(k))
expected = [("init.init_sdf_grid", old, init_sdf)]
if changes != expected:
    raise SystemExit(f"unexpected config changes:\n{changes!r}\nexpected:\n{expected!r}")
dst.write_text(json.dumps(out, indent=2) + "\n")
print("Derived config differs from cancelled run only at:")
print("   init.init_sdf_grid:", old, "->", init_sdf)
print("   scene (unchanged):", out.get("scene"))
PY

echo "Previous (cancelled): $PREV_RUN"
echo "Init SDF grid (new):  $NEW_INIT_SDF"
echo "Derived config:       $DERIVED_CONFIG"
echo "Run dir:              $RUN_DIR"
echo "Official TnT eval:    $TNT_SCENE_NAME every ${TNT_OFFICIAL_FREQ} steps  res=${TNT_OFFICIAL_RES}  n=${TNT_OFFICIAL_N_SAMPLES}  frame=${TNT_OFFICIAL_FRAME}"

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
