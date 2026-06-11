#!/bin/bash
#SBATCH --job-name=1lip-sphere-nomask
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-sphere-nomask_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-sphere-nomask_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

# Sphere-carved, no-mask init launcher. Defaults reproduce the Truck run; override
# the env vars below to retarget (e.g. scan122). All config changes vs the source
# are applied as an explicit dotted-path override map and validated to differ at
# exactly those keys, so retargeting can't silently drift other settings.
#   JOB_TAG        run-dir / derived-config tag
#   SOURCE_CONFIG  source run config.json to derive from
#   NEW_INIT_SDF   sphere-carved no-mask init SDF grid (.npy)
#   EXTRA_OVERRIDES  JSON dict of dotted-path -> value applied on top of the
#                    always-on {init.init_sdf_grid, train.use_masks=false}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs

JOB_TAG=${JOB_TAG:-truck_gsam_sphere_nomask_init}
SOURCE_CONFIG=${SOURCE_CONFIG:-$OUTPUTS/run_20260605_102117_truck_gsam_mvsformer_carved_init_4982729/source_config_truck_gsam_mvsformer_carved_init.json}
NEW_INIT_SDF=${NEW_INIT_SDF:-$ROOT/_diagnostics/mvsformer_truck_sphere/sphere_carve_auto_res192_nomask_v3_sfmclip_clean_close2/sdf_sphere_mvsformer_carved_clean.npy}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-'{}'}

# Optional DTU official Chamfer eval. The train CLI unconditionally overrides
# eval.dtu_official_freq from its arg default (0), so the config value alone is
# NOT enough — these flags must be passed to enable it. Off by default (truck/TnT).
DTU_EVAL_DIR=${DTU_EVAL_DIR:-}
DTU_OFFICIAL_FREQ=${DTU_OFFICIAL_FREQ:-}
DTU_OFFICIAL_RES=${DTU_OFFICIAL_RES:-384}
DTU_OFFICIAL_BOUND=${DTU_OFFICIAL_BOUND:-1.0}
DTU_ARGS=()
if [ -n "$DTU_EVAL_DIR" ] && [ -n "$DTU_OFFICIAL_FREQ" ]; then
  DTU_ARGS=(--dtu-eval-dir "$DTU_EVAL_DIR" --dtu-official-freq "$DTU_OFFICIAL_FREQ" \
            --dtu-official-res "$DTU_OFFICIAL_RES" --dtu-official-bound "$DTU_OFFICIAL_BOUND")
fi

# Optional official TnT F-score eval. Identical good-alignment recipe as a standalone
# eval_tnt_official.py run (MC -> area-uniform points -> multi-scale ICP registration
# -> get_f1_score_histo2): the periodic path calls the same run_tnt_eval(). Like the
# DTU block, the train CLI overrides eval.tnt_official_freq from its arg default, so
# these flags must be passed. Set TNT_SCENE_NAME (e.g. Truck / Ignatius) to turn it
# on; it then runs every TNT_OFFICIAL_FREQ steps, just like the DTU eval above.
TNT_SCENE_NAME=${TNT_SCENE_NAME:-}
TNT_EVAL=${TNT_EVAL:-$ROOT/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}
TNT_ARGS=()
if [ -n "$TNT_SCENE_NAME" ]; then
  TNT_ARGS=(--tnt-eval-dir "$TNT_EVAL" --tnt-official-scene "$TNT_SCENE_NAME" \
            --tnt-official-frame "$TNT_OFFICIAL_FRAME" --tnt-official-freq "$TNT_OFFICIAL_FREQ" \
            --tnt-official-res "$TNT_OFFICIAL_RES" --tnt-official-n-samples "$TNT_OFFICIAL_N_SAMPLES")
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_${JOB_TAG}_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_${JOB_TAG}.json
LOG=$RUN_DIR/train.log

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-${JOB_TAG}

mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

if [ ! -f "$SOURCE_CONFIG" ]; then
  echo "ERROR: missing source config: $SOURCE_CONFIG" >&2
  exit 1
fi
if [ ! -f "$NEW_INIT_SDF" ]; then
  echo "ERROR: missing sphere-carved no-mask init SDF grid: $NEW_INIT_SDF" >&2
  exit 1
fi

python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$NEW_INIT_SDF" "$EXTRA_OVERRIDES" <<'PY'
import copy
import json
import sys
from pathlib import Path

src, dst, init_sdf, extra_json = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = json.loads(src.read_text())
cfg.pop("cmd", None)
out = copy.deepcopy(cfg)
if out.get("init", {}).get("init") != "hull":
    raise SystemExit(f"reference run is not hull init: {out.get('init',{}).get('init')!r}")

# always-on overrides + caller-supplied extras (dotted paths)
overrides = {"init.init_sdf_grid": init_sdf, "train.use_masks": False}
overrides.update(json.loads(extra_json))

def flatten(v, p=""):
    if isinstance(v, dict):
        o = {}
        for k, it in v.items():
            o.update(flatten(it, f"{p}.{k}" if p else k))
        return o
    return {p: v}

before = flatten(cfg)
expected = sorted((k, before.get(k), val) for k, val in overrides.items())
for path, val in overrides.items():
    node = out
    *parents, leaf = path.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = val

after = flatten(out)
changes = sorted((k, before.get(k), after.get(k))
                 for k in before.keys() | after.keys()
                 if k != "cmd" and before.get(k) != after.get(k))

if changes != expected:
    raise SystemExit(f"unexpected config changes:\n{changes!r}\nexpected:\n{expected!r}")

dst.write_text(json.dumps(out, indent=2) + "\n")
print("Derived config differs only at:")
for item in expected:
    print("  ", item)
print("   scene (unchanged):", out.get("scene"))
PY

echo "Source config:     $SOURCE_CONFIG"
echo "Init SDF grid:     $NEW_INIT_SDF"
echo "Extra overrides:   $EXTRA_OVERRIDES"
echo "Derived config:    $DERIVED_CONFIG"
echo "Run dir:           $RUN_DIR"
echo "Mask mode:         train.use_masks=false / --no-use-masks"
echo "DTU eval:          ${DTU_ARGS[*]:-<off>}"
echo "TnT eval:          ${TNT_ARGS[*]:-<off>}"

python -u -m lip_tracer.train \
    --config "$DERIVED_CONFIG" \
    --run-dir "$RUN_DIR" \
    --occ-iters 32 \
    --occ-newton-steps 0 \
    --occ-eps 1e-3 \
    --no-use-masks \
    ${DTU_ARGS[@]+"${DTU_ARGS[@]}"} \
    ${TNT_ARGS[@]+"${TNT_ARGS[@]}"} \
    --wandb 2>&1 | tee "$LOG"
