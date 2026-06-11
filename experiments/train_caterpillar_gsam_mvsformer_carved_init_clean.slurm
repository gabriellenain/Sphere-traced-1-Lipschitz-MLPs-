#!/bin/bash
#SBATCH --job-name=1lip-cat-mvsfinit-clean
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-cat-mvsfinit-clean_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-cat-mvsfinit-clean_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014
set -euo pipefail
ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs
BASE_CONFIG=$OUTPUTS/run_20260605_102117_truck_gsam_mvsformer_carved_init_4982729/source_config_truck_gsam_mvsformer_carved_init.json
SCENE=$ROOT/data/tnt/Caterpillar
TNT_SCENE_NAME=Caterpillar
TNT_EVAL=${TNT_EVAL:-$ROOT/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}
NEW_INIT_SDF=$ROOT/_diagnostics/mvsformer_caterpillar_gsam_clean/hull_carve_goodviews/sdf_vh_mvsformer_carved.npy
GOOD_VIEWS=$ROOT/_diagnostics/mvsformer_caterpillar_gsam_clean/good_views_p90.txt
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_caterpillar_gsam_mvsformer_carved_init_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_caterpillar_gsam_mvsformer_carved_init.json
LOG=$RUN_DIR/train.log
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-cat-mvsf-clean
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"
[ -f "$NEW_INIT_SDF" ] || { echo "ERROR: missing init SDF: $NEW_INIT_SDF" >&2; exit 1; }
[ -f "$GOOD_VIEWS" ] || { echo "ERROR: missing good-views list: $GOOD_VIEWS" >&2; exit 1; }
python - "$BASE_CONFIG" "$DERIVED_CONFIG" "$SCENE" "$NEW_INIT_SDF" <<'PY'
import copy, json, sys
from pathlib import Path
base, dst, scene, init_sdf = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = json.loads(base.read_text()); cfg.pop("cmd", None)
out = copy.deepcopy(cfg)
if out.get("init", {}).get("init") != "hull":
    raise SystemExit("base run is not hull init")
old_scene, old_init = out.get("scene"), out["init"].get("init_sdf_grid")
out["scene"] = scene
out["init"]["init_sdf_grid"] = init_sdf
def flat(v,p=""):
    if isinstance(v,dict):
        o={}
        for k,it in v.items(): o.update(flat(it,f"{p}.{k}" if p else k))
        return o
    return {p:v}
changes=sorted((k,flat(cfg).get(k),flat(out).get(k)) for k in flat(cfg).keys()|flat(out).keys() if k!="cmd" and flat(cfg).get(k)!=flat(out).get(k))
expected=sorted([("scene",old_scene,scene),("init.init_sdf_grid",old_init,init_sdf)])
if changes!=expected:
    raise SystemExit(f"unexpected changes:\n{changes}\nexpected:\n{expected}")
dst.write_text(json.dumps(out,indent=2)+"\n")
print("Derived caterpillar config differs from truck base only at:")
for c in expected: print("  ",c)
PY
echo "Run dir: $RUN_DIR"
echo "Official TnT eval: $TNT_SCENE_NAME every ${TNT_OFFICIAL_FREQ} steps  res=${TNT_OFFICIAL_RES}  n=${TNT_OFFICIAL_N_SAMPLES}  frame=${TNT_OFFICIAL_FRAME}"
python -u -m lip_tracer.train --config "$DERIVED_CONFIG" --run-dir "$RUN_DIR" \
    --good-views "$GOOD_VIEWS" \
    --occ-iters 32 --occ-newton-steps 0 --occ-eps 1e-3 \
    --tnt-eval-dir "$TNT_EVAL" \
    --tnt-official-scene "$TNT_SCENE_NAME" \
    --tnt-official-frame "$TNT_OFFICIAL_FRAME" \
    --tnt-official-freq "$TNT_OFFICIAL_FREQ" \
    --tnt-official-res "$TNT_OFFICIAL_RES" \
    --tnt-official-n-samples "$TNT_OFFICIAL_N_SAMPLES" \
    --wandb 2>&1 | tee "$LOG"
