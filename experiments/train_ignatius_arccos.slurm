#!/bin/bash
#SBATCH --job-name=1lip-arccos
#SBATCH --output=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-arccos_%j.slurm.log
#SBATCH --error=/scratch/_projets_/willow/1-lip-tracer-new/outputs/1lip-arccos_%j.slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

set -euo pipefail

# Fresh-from-init (step 0) sphere-nomask run, identical to the reference
# run 4992251 EXCEPT for the camera/view sampling changes:
#   --view-selection arccos : alt cams ranked by arccos(viewing-dir) angular dist
#   --init-hit-sampling      : fg = rays hitting the init SDF (mask-free)
#   --fg-fraction 0.9        : per batch 90% init-hit fg, 10% uniform bg
#   + 2-level ray sampling   : camera ~ Uniform(V), then ray ~ Uniform within cam
# Official TnT F-score logged every 10000 steps (same recipe as the reference).
#
# Arg 1 (default Ignatius) selects the TnT scene. For any scene other than
# Ignatius the Ignatius-specific carved init SDF grid does not apply, so the
# derived config drops init_sdf_grid and the hull is carved IN-LOOP with the
# same hull params (hull_res / hull_border_aware / hull_sfm_roi) — matching the
# proven Barn scan118cfg init. Reference cam + rays are chosen identically.

SCENE_NAME=${1:-Ignatius}

# MODE selects how much of the reference recipe changes:
#   full    : --view-selection arccos  + init-hit fg sampling, fg-fraction 0.9,
#             force-fg-bg-split, 2-level uniform-cam ray sampling (original).
#   minimal : --view-selection arccos_nn — ONLY the alt-camera neighbour ranking
#             changes; reference-view and ray selection stay EXACTLY as the
#             baseline nearest run (all other train flags inherited from config).
MODE=${MODE:-full}

ROOT=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-
OUTPUTS=/scratch/_projets_/willow/1-lip-tracer-new/outputs
OUTPUTS_HOME=$ROOT/outputs

REFERENCE_RUN=$OUTPUTS/run_20260606_211917_ignatius_sphere_nomask_res384_init_4992251
SOURCE_CONFIG=$REFERENCE_RUN/config.json
SCENE=$ROOT/data/tnt/${SCENE_NAME}

STEPS=${STEPS:-600000}
# WOBJ (optional): object-fixed ZNCC patch footprint in WORLD units. When set,
# overrides the reference run's view-fixed ncc_half_pix sizing with a constant
# metric footprint (--ncc-world-patch). Ignatius 8mm @0.883 m/unit ≈ 0.009.
# Unset → identical to the reference arccos run (view-fixed half_pix=2).
WOBJ=${WOBJ:-}
# DOWN: image downsample factor for training (overrides config train.down).
# Per-pixel ray precompute is O(V*H*W) on CPU; for many-view full-res scenes
# (e.g. Courthouse: 1106 views @ 1091x1958 -> ~125 GB rays) down=1 OOMs the
# 128 GB host. DOWN=2 cuts rays/images/GPU-flats 4x; harmless for many-view TnT.
DOWN=${DOWN:-1}
# BA_INLOOP defined here (before RUN_DIR) so the run dir carries a BA tag.
BA_INLOOP=${BA_INLOOP:-0}
BA_TAG=""; [ "$BA_INLOOP" = "1" ] && BA_TAG="_bainloop"
TNT_SCENE_NAME=$SCENE_NAME
TNT_EVAL=${TNT_EVAL:-$ROOT/data/tnt_gt}
TNT_OFFICIAL_FRAME=${TNT_OFFICIAL_FRAME:-colmap-pose}
TNT_OFFICIAL_FREQ=${TNT_OFFICIAL_FREQ:-10000}
TNT_OFFICIAL_RES=${TNT_OFFICIAL_RES:-512}
TNT_OFFICIAL_N_SAMPLES=${TNT_OFFICIAL_N_SAMPLES:-2000000}

if [ ! -f "$SOURCE_CONFIG" ]; then
  echo "ERROR: missing reference config: $SOURCE_CONFIG" >&2
  exit 1
fi
if [ ! -d "$SCENE" ]; then
  echo "ERROR: missing TnT scene: $SCENE" >&2
  exit 1
fi
if [ ! -f "$TNT_EVAL/${SCENE_NAME}/${SCENE_NAME}.ply" ]; then
  echo "ERROR: missing official TnT GT: $TNT_EVAL/${SCENE_NAME}/${SCENE_NAME}.ply" >&2
  exit 1
fi

source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib-1lip-arccos

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUTPUTS/run_${TS}_${SCENE_NAME}_sphere_nomask_arccos_${MODE}${WOBJ:+_wobj${WOBJ}}${BA_TAG}_${SLURM_JOB_ID}
DERIVED_CONFIG=$RUN_DIR/source_config_arccos.json
LOG=$RUN_DIR/train.log
mkdir -p "$RUN_DIR" "$OUTPUTS_HOME"
ln -sfn "$RUN_DIR" "$OUTPUTS_HOME/$(basename "$RUN_DIR")"

# Derive a from-scratch config from the reference: keep everything, strip the
# embedded resume cmd, set the new end-step horizon, and point scene + TnT eval
# at SCENE_NAME. For Ignatius the reference carved init SDF grid is reused as-is;
# for any other scene that grid does not apply, so it is dropped and the hull is
# carved IN-LOOP with the same hull params.
# INIT_SDF_GRID (optional): use this precomputed MVSFormer-carved SDF grid as the
# hull-init target (same mechanism as Ignatius). If unset, Ignatius keeps its
# reference grid and any other scene carves the hull in-loop.
INIT_SDF_GRID=${INIT_SDF_GRID:-}
# In-loop bundle adjustment (interleaved φ-blocks during training). Default OFF;
# set BA_INLOOP=1 to refine poses while training (weak-pose TnT scenes only).
# BA_WARMUP : θ-only steps before the first φ-block (let normals sharpen).
# BA_INTERVAL: θ-steps between φ-blocks.  BA_PHI: φ-steps per block.
# BA_LR     : extrinsic learning rate.   BA_RAMP: φ-blocks to ramp LR over.
# (BA_INLOOP defaulted above so the run dir can carry its tag.)
BA_WARMUP=${BA_WARMUP:-10000}
BA_INTERVAL=${BA_INTERVAL:-2000}
BA_PHI=${BA_PHI:-200}
BA_LR=${BA_LR:-1e-5}
BA_RAMP=${BA_RAMP:-2}
python - "$SOURCE_CONFIG" "$DERIVED_CONFIG" "$STEPS" "$SCENE_NAME" "$SCENE" "$INIT_SDF_GRID" \
    "$BA_INLOOP" "$BA_WARMUP" "$BA_INTERVAL" "$BA_PHI" "$BA_LR" "$BA_RAMP" <<'PY'
import json, sys
from pathlib import Path
src, dst, steps, scene_name, scene, init_grid = (
    Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]),
    sys.argv[4], sys.argv[5], sys.argv[6])
ba_inloop, ba_warmup, ba_interval, ba_phi, ba_lr, ba_ramp = (
    sys.argv[7] == "1", int(sys.argv[8]), int(sys.argv[9]),
    int(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12]))
cfg = json.loads(src.read_text())
cfg.pop("cmd", None)
cfg["train"]["steps"] = steps
cfg["scene"] = scene
cfg.setdefault("eval", {})["tnt_official_scene"] = scene_name
if ba_inloop:
    bundle = cfg["train"].setdefault("bundle", {})
    bundle.update({"in_loop": True, "warmup_steps": ba_warmup,
                   "interval": ba_interval, "block_phi": ba_phi,
                   "lr": ba_lr, "lr_ramp_blocks": ba_ramp, "lock_first": True})
    print(f"in-loop BA: warmup={ba_warmup} interval={ba_interval} "
          f"block_phi={ba_phi} lr={ba_lr:g} ramp={ba_ramp}")
if cfg.get("init", {}).get("init") != "hull":
    raise SystemExit("reference run is not hull init")
if init_grid:
    if not Path(init_grid).exists():
        raise SystemExit(f"missing INIT_SDF_GRID: {init_grid}")
    cfg["init"]["init_sdf_grid"] = init_grid
elif scene_name == "Ignatius":
    if not Path(cfg["init"]["init_sdf_grid"]).exists():
        raise SystemExit(f"missing init SDF grid: {cfg['init']['init_sdf_grid']}")
else:
    cfg["init"]["init_sdf_grid"] = None   # carve hull in-loop for this scene
dst.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"derived config -> {dst}  (scene={scene_name}, steps={steps}, "
      f"init_sdf_grid={cfg['init']['init_sdf_grid']})")
PY

# Sampling flags differ by MODE. full = original arccos recipe (changes neighbour
# ranking AND fg/ray sampling). minimal = arccos_nn (only neighbour ranking; all
# sampling flags inherited from the reference config, i.e. nearest-run behaviour).
if [ "$MODE" = "none" ]; then
    VIEW_FLAGS=()
    MODE_DESC="plain sphere_nomask — NO arccos, NO init-hit, NO mask (exactly the DTU sphere_nomask recipe + carved init)"
elif [ "$MODE" = "minimal" ]; then
    VIEW_FLAGS=(--view-selection arccos_nn)
    MODE_DESC="arccos_nn — neighbour ranking ONLY (ref-view + ray sampling = nearest baseline)"
else
    VIEW_FLAGS=(--view-selection arccos --init-hit-sampling --fg-fraction 0.9 --force-fg-bg-split)
    MODE_DESC="arccos + init-hit + fg-fraction 0.9 (2-level uniform-cam)"
fi

# Object-fixed patch footprint (optional). Unset → reference view-fixed sizing.
WOBJ_FLAG=()
if [ -n "$WOBJ" ]; then WOBJ_FLAG=(--ncc-world-patch "$WOBJ"); fi

echo "Reference run:     $REFERENCE_RUN"
echo "Scene:             $SCENE_NAME ($SCENE)"
echo "Run dir:           $RUN_DIR"
echo "Steps (from 0):    $STEPS"
echo "Downsample:        $DOWN"
echo "Mode:              $MODE — $MODE_DESC"
echo "Patch sizing:      ${WOBJ:+object-fixed --ncc-world-patch $WOBJ}${WOBJ:-view-fixed (reference ncc_half_pix)}"
echo "Official TnT eval: $TNT_SCENE_NAME every ${TNT_OFFICIAL_FREQ} steps  res=${TNT_OFFICIAL_RES}"
if [ "$BA_INLOOP" = "1" ]; then
  echo "In-loop BA:        ON  warmup=${BA_WARMUP} interval=${BA_INTERVAL} block_phi=${BA_PHI} lr=${BA_LR} ramp=${BA_RAMP}"
else
  echo "In-loop BA:        off"
fi

python -u -m lip_tracer.train \
    --config "$DERIVED_CONFIG" \
    --run-dir "$RUN_DIR" \
    --down "$DOWN" \
    --occ-iters 32 \
    --occ-newton-steps 0 \
    --occ-eps 1e-3 \
    --no-use-masks \
    "${VIEW_FLAGS[@]}" \
    "${WOBJ_FLAG[@]}" \
    --tnt-eval-dir "$TNT_EVAL" \
    --tnt-official-scene "$TNT_SCENE_NAME" \
    --tnt-official-frame "$TNT_OFFICIAL_FRAME" \
    --tnt-official-freq "$TNT_OFFICIAL_FREQ" \
    --tnt-official-res "$TNT_OFFICIAL_RES" \
    --tnt-official-n-samples "$TNT_OFFICIAL_N_SAMPLES" \
    --wandb 2>&1 | tee "$LOG"
