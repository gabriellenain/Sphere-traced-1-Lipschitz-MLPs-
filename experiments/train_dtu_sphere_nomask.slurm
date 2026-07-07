#!/bin/bash
#SBATCH --job-name=1lip-dtu-nomask
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-dtu-nomask_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-dtu-nomask_%j.slurm.log
#SBATCH --ntasks=1
# Sphere-trace step is host-dispatch-bound (~250 tiny sequential kernels/step);
# 2 cores get starved by noisy node neighbours (~5K vs ~12K steps/h). Give it headroom.
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

# NCC-only, no-mask, sphere-carved-init DTU training. Reproduces the scan122 run
# (run_20260606_142255_..._4991521) exactly: all train/trace params come from its
# source_config; only scene, init SDF grid, and the scan-specific feature_maps
# path are retargeted. Override via env vars:
#   SCENE          DTU-idr scene dir (default scan37)
#   SOURCE_CONFIG  reference run source config to derive from (default scan122)
#   NEW_INIT_SDF   sphere-carved no-mask init SDF grid (.npy) for this scan
#   JOB_TAG        run-dir / derived-config tag

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs

SCENE=${SCENE:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan37}
SCAN=$(basename "$SCENE")
JOB_TAG=${JOB_TAG:-${SCAN}_sphere_nomask_init}
SOURCE_CONFIG=${SOURCE_CONFIG:-$OUTPUTS/run_20260606_142255_scan122_sphere_nomask_init_4991521/source_config_scan122_sphere_nomask_init.json}
NEW_INIT_SDF=${NEW_INIT_SDF:-$ROOT/_diagnostics/mvsformer_${SCAN}_sphere/sphere_carve_auto_res256_nomask_clip2_close2/sdf_sphere_mvsformer_carved.npy}

# BAKE_BG=0 → truly mask-free: keep the real photographed background (no img[~msk]=0
# bake at load). Default 1 reproduces the historic "nomask supervision, baked bg" runs.
BAKE_BG=${BAKE_BG:-1}
if [ "$BAKE_BG" = "0" ]; then BAKE_BG_FLAG=--no-bake-background; else BAKE_BG_FLAG=--bake-background; fi

# NCC_SAT_TAU>0 → reference-saturation gate (drop rays whose reference pixel max
# channel ≥ τ, i.e. specular highlights). Empty/<=0 → disabled (unchanged behaviour).
NCC_SAT_TAU=${NCC_SAT_TAU:-}
if [ -n "$NCC_SAT_TAU" ]; then SAT_FLAG="--ncc-sat-tau $NCC_SAT_TAU"; else SAT_FLAG=""; fi

# T_FAR>0 → override the source config's flat sphere-trace t_far. The shared
# source config uses t_far=5, which is too short for scenes whose cameras sit far
# from the origin in the normalized frame (e.g. bmvs_jade cams at ~7-8 → 0 hits).
T_FAR=${T_FAR:-}
if [ -n "$T_FAR" ]; then TFAR_FLAG="--t-far $T_FAR"; else TFAR_FLAG=""; fi

# W_EIKONAL / MULTIRES: baked into the DERIVED CONFIG below (NOT passed as CLI
# flags — on the --config path train.py ignores --w-eikonal and --multires).
# Empty → leave the source config's value untouched (unchanged behaviour).
#   W_EIKONAL  activate the eikonal loss, e.g. 0.1 (source config carries 0.0)
#   MULTIRES   positional-encoding L (frequency bands); 0 = no Fourier bands
W_EIKONAL=${W_EIKONAL:-}
MULTIRES=${MULTIRES:-}

DTU_EVAL_DIR=${DTU_EVAL_DIR:-/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/SampleSet/MVS Data}
DTU_OFFICIAL_FREQ=${DTU_OFFICIAL_FREQ:-10000}
DTU_OFFICIAL_RES=${DTU_OFFICIAL_RES:-384}
DTU_OFFICIAL_BOUND=${DTU_OFFICIAL_BOUND:-1.0}

# Optional in-training BlendedMVS Chamfer (for BMVS scenes). Off unless
# BMVS_GT_MESH is set. Uses the corrected eval baked into
# lip_tracer/train.py:_run_bmvs_official_eval — gt-space normalized
# GroundTruth.ply, point-to-mesh distance, DTU-style mask crop+dilate12 —
# at step 0 then every BMVS_OFFICIAL_FREQ steps. DTU runs are unaffected.
BMVS_GT_MESH=${BMVS_GT_MESH:-}
BMVS_FLAGS=()
if [ -n "$BMVS_GT_MESH" ]; then
  BMVS_FLAGS=(--bmvs-gt-mesh "$BMVS_GT_MESH"
    --bmvs-official-freq "${BMVS_OFFICIAL_FREQ:-10000}"
    --bmvs-official-res "${BMVS_OFFICIAL_RES:-512}"
    --bmvs-official-bound "${BMVS_OFFICIAL_BOUND:-1.5}"
    --bmvs-official-protocol "${BMVS_OFFICIAL_PROTOCOL:-probesdf}"
    --bmvs-official-n-samples "${BMVS_OFFICIAL_N_SAMPLES:-300000}")
fi

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-${JOB_TAG}

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_${JOB_TAG}_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_${JOB_TAG}.json
LOG=$RUN_DIR/train.log
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

if [ ! -f "$SOURCE_CONFIG" ]; then
  echo "ERROR: missing source config: $SOURCE_CONFIG" >&2; exit 1
fi
if [ ! -f "$NEW_INIT_SDF" ]; then
  echo "ERROR: missing sphere-carved init SDF: $NEW_INIT_SDF" >&2; exit 1
fi

# Derive config: change ONLY scene, init SDF grid, and scan-specific feature_maps.
python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$SCENE" "$NEW_INIT_SDF" <<'PY'
import copy, json, os, sys
from pathlib import Path
src, dst, scene, init_sdf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = json.loads(Path(src).read_text()); cfg.pop("cmd", None)
out = copy.deepcopy(cfg)
if out.get("init", {}).get("init") != "hull":
    raise SystemExit(f"reference is not hull init: {out.get('init',{}).get('init')!r}")
out["scene"] = scene
out["init"]["init_sdf_grid"] = init_sdf
out["train"]["feature_maps"] = None  # scan122-specific path; w_feature=0 so unused

# feature_maps may already be null in the source config (no diff then)
required = {"scene", "init.init_sdf_grid"}
allowed = required | {"train.feature_maps"}
# Model/loss overrides baked into the config (train.py ignores the matching CLI
# flags on the --config path). Each is applied only when its env var is set.
w_eik = os.environ.get("W_EIKONAL", "")
if w_eik:
    out["train"]["w_eikonal"] = float(w_eik)
    allowed.add("train.w_eikonal")
multires = os.environ.get("MULTIRES", "")
if multires:
    out["model"]["multires"] = int(multires)
    allowed.add("model.multires")

def flat(v, p=""):
    if isinstance(v, dict):
        o = {}
        for k, it in v.items(): o.update(flat(it, f"{p}.{k}" if p else k))
        return o
    return {p: v}
before, after = flat(cfg), flat(out)
changes = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
if not (required <= set(changes) <= allowed):
    raise SystemExit(f"unexpected config changes: {changes!r}\nallowed: {sorted(allowed)!r}")
Path(dst).write_text(json.dumps(out, indent=2) + "\n")
print("Derived config differs only at:", changes)
print("  scene:", scene)
print("  init_sdf_grid:", init_sdf)
PY

echo "Source config:  $SOURCE_CONFIG"
echo "Init SDF grid:  $NEW_INIT_SDF"
echo "Run dir:        $RUN_DIR"

python -u -m lip_tracer.train \
  --config "$DERIVED_CONFIG" \
  --run-dir "$RUN_DIR" \
  --occ-iters 32 \
  --occ-newton-steps 0 \
  --occ-eps 1e-3 \
  --no-use-masks \
  $BAKE_BG_FLAG \
  $SAT_FLAG \
  $TFAR_FLAG \
  --dtu-eval-dir "$DTU_EVAL_DIR" \
  --dtu-official-freq "$DTU_OFFICIAL_FREQ" \
  --dtu-official-res "$DTU_OFFICIAL_RES" \
  --dtu-official-bound "$DTU_OFFICIAL_BOUND" \
  "${BMVS_FLAGS[@]}" \
  --wandb 2>&1 | tee "$LOG"
