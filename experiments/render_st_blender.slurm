#!/bin/bash
#SBATCH --job-name=st-blend
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_blend_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_blend_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00

# Sphere-trace ONE view's surface (no marching cubes) into a screen-space mesh,
# then Blender-AO render it + save a .blend — same recipe used for DTU scan65/122.
# Usage:
#   sbatch render_st_blender.slurm <run_dir> [ckpt] [view] [down] [res] [out_dir]
# Defaults: ckpt=checkpoint_latest.pt view=16 down=2 res=1024
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${1:?usage: sbatch render_st_blender.slurm <run_dir> [ckpt] [views] [down] [res] [out_dir]}"
CKPT="${2:-checkpoint_latest.pt}"
VIEWS="${3:-16}"          # comma list, e.g. 0,16,32
DOWN="${4:-2}"
RES="${5:-1024}"
OUT="${6:-outputs/st_blend_${SLURM_JOB_ID}}"
BLENDER="${BLENDER:-$HOME/blender/blender}"
# Mesh-builder culls. Faithful default: grazing_cos=0 keeps EVERY hit the run's
# tracer produced (grazing surface shows as slivers, not holes). rel_depth_gap
# still separates true silhouette depth-jumps so faces don't bridge fg→bg
# (that bridging is a meshing artifact, not something the run did).
GRAZING_COS="${GRAZING_COS:-0.0}"
RELGAP="${RELGAP:-0.02}"
# KEEP_LARGEST=1: preserve the raw sphere-traced PLY, then render a sibling
# _largest.ply containing only the largest face-connected component.
KEEP_LARGEST="${KEEP_LARGEST:-0}"
# SS>1: supersample the fg-mask bbox ss× per axis → DTU-level vertex density on a
# subject that fills little of the frame (e.g. MVMannequin ~1%). 1 = native grid.
SS="${SS:-1}"
# Blender render frame. The positional RES gives a square frame (legacy default);
# set RES_X/RES_Y to render the subject's NATIVE aspect (DTU = 1600x1200) so the
# zoom/framing matches the paper figures (tools/render_blender.py / blender_st_paper6).
# A square 1024x1024 frame with native K silently zooms+crops differently.
RES_X="${RES_X:-$RES}"
RES_Y="${RES_Y:-$RES}"

[ -d "$RUN_DIR" ] || { echo "[error] run_dir not found: $RUN_DIR"; exit 1; }
[ -x "$BLENDER" ] || { echo "[error] Blender not at $BLENDER"; exit 1; }
mkdir -p "$OUT"
SCENE=$(python -c "import json;print(json.load(open('$RUN_DIR/config.json'))['scene'])")
# Match the trace exactly to the run that made this ckpt by default:
# eps + newton_steps load from cfg['trace'] inside the script; iters is the one
# value the script overrides via --max-iters, so pull it from the config unless
# TRACE_ITERS is set for an intentional re-render.
ITERS="${TRACE_ITERS:-$(python -c "import json;print(json.load(open('$RUN_DIR/config.json'))['trace']['iters'])")}"

copy_blender_gt_view() {
    local view="$1"
    [[ -f "$SCENE/transforms_train.json" ]] || return 0
    python -c '
import json, sys
from pathlib import Path
from PIL import Image

scene = Path(sys.argv[1])
view = int(sys.argv[2])
out_dir = Path(sys.argv[3])
res = (int(sys.argv[4]), int(sys.argv[5]))
meta = json.loads((scene / "transforms_train.json").read_text())
src = scene / (meta["frames"][view]["file_path"] + ".png")
rgba = Image.open(src).convert("RGBA")
bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
rgb = Image.alpha_composite(bg, rgba).convert("RGB")
if rgb.size != res:
    rgb = rgb.resize(res, Image.Resampling.LANCZOS)
dst = out_dir / f"view{view:03d}_gt.png"
rgb.save(dst)
print(f"[gt] saved {dst} <- {src}", flush=True)
' "$SCENE" "$view" "$OUT" "$RES_X" "$RES_Y"
}

echo "run_dir: $RUN_DIR"
echo "ckpt:    $CKPT"
echo "scene:   $SCENE"
if [[ -n "${TRACE_ITERS:-}" ]]; then
    echo "views:   $VIEWS  down=$DOWN  res=${RES_X}x${RES_Y}  trace_iters=$ITERS (override)"
else
    echo "views:   $VIEWS  down=$DOWN  res=${RES_X}x${RES_Y}  trace_iters=$ITERS (from run cfg)"
fi
echo "culls:   grazing_cos=$GRAZING_COS  rel_depth_gap=$RELGAP  (0 grazing = keep all hits)"
echo "clean:   keep_largest=$KEEP_LARGEST"
echo "out:     $OUT"

# One GPU job: trace each view -> world-frame screen mesh, then Blender AO + .blend.
for VIEW in ${VIEWS//,/ }; do
    echo "=================== view $VIEW ==================="
    ST_PLY="$OUT/st_view${VIEW}_world.ply"

    # ---- 1. GPU sphere-trace -> world-frame screen mesh (∇f normals point outward → toward cam)
    # EXPAND_PX>0 pads the traced frustum by N px/side so a subject clipped by the
    # landscape frame (e.g. a tall bust's crown) is captured whole (still one trace).
    EXPAND_ARG=()
    [[ "${EXPAND_PX:-0}" != "0" ]] && EXPAND_ARG=(--expand-px "$EXPAND_PX")
    # CHUNK lowers per-step ray batch → smaller GPU peak (use on busy/shared GPUs to avoid OOM).
    python -u analysis/sphere_traced_screen_mesh.py "$RUN_DIR" \
        --ckpt "$CKPT" --view "$VIEW" --down "$DOWN" --max-iters "$ITERS" \
        --supersample "$SS" "${EXPAND_ARG[@]}" --chunk "${CHUNK:-32768}" \
        --grazing-cos "$GRAZING_COS" --rel-depth-gap "$RELGAP" \
        --out "$ST_PLY" || { echo "[error] trace failed for view $VIEW"; continue; }

    RENDER_PLY="$ST_PLY"
    if [[ "$KEEP_LARGEST" == "1" ]]; then
        RENDER_PLY="$OUT/st_view${VIEW}_world_largest.ply"
        python -u tools/keep_largest_mesh_component.py \
            --mesh "$ST_PLY" --out "$RENDER_PLY" \
            || { echo "[error] largest-component cleanup failed for view $VIEW"; continue; }
    fi

    # ---- 2. Blender Cycles render + .blend, same view's camera.
    # STUDIO=1 (default) => hero look: studio 3-point light, marble material, soft
    # shadow, AgX tonemap, denoise, tight centred auto-frame, floaters isolated.
    # STUDIO=0 => the original flat demo-AO recipe.
    STUDIO="${STUDIO:-1}"
    STUDIO_ARGS=()
    if [[ "$STUDIO" == "1" ]]; then
        STUDIO_ARGS=(--studio --studio-samples "${STUDIO_SAMPLES:-256}" \
                     --studio-fill "${STUDIO_FILL:-0.82}" --studio-bg "${STUDIO_BG:-0.21}" \
                     --studio-style "${STUDIO_STYLE:-marble}")
        # STUDIO_FRAME_PCT>0: auto-frame on the [pct,100-pct] projected percentile box
        # so kept floaters (STUDIO_KEEP_FLOATERS=1) don't shrink/offset the subject.
        [[ "${STUDIO_FRAME_PCT:-0}" != "0" ]] && STUDIO_ARGS+=(--studio-frame-pct "$STUDIO_FRAME_PCT")
        [[ -n "${STUDIO_TRANSPARENT:-}" ]] && STUDIO_ARGS+=(--studio-transparent)
        # Keep every piece of a fragmented single-view shell instead of only the
        # largest connected component. This matches KEEP_LARGEST=0 in the PLY
        # stage, so the whole pipeline preserves all components by default.
        [[ "${STUDIO_KEEP_FLOATERS:-1}" == "1" ]] && STUDIO_ARGS+=(--studio-keep-floaters)
        # STUDIO_NO_FRAME=1: render from the EXACT training camera (skip auto-frame).
        # auto-frame re-aims+zooms on the subject; a tall subject (e.g. Ignatius)
        # drives the zoom so the wide single-view ground shell spills past the frame
        # and the border is clipped — set this to keep the faithful view framing.
        [[ "${STUDIO_NO_FRAME:-0}" == "1" ]] && STUDIO_ARGS+=(--studio-no-frame)
        # STUDIO_SOLIDIFY>0: thicken the open single-view shell (+filled rim) so
        # silhouette cracks at depth discontinuities (e.g. the arm) render as a wall,
        # not the backdrop bleeding through as bright white contour lines.
        [[ "${STUDIO_SOLIDIFY:-0}" != "0" ]] && STUDIO_ARGS+=(--studio-solidify "$STUDIO_SOLIDIFY")
    fi
    if [[ -d "$SCENE/pose" && -f "$SCENE/intrinsics.txt" ]]; then
        "$BLENDER" --background --python tools/render_blender_tnt.py -- \
            --mesh "$RENDER_PLY" --scene "$SCENE" --views "$VIEW" \
            --out_dir "$OUT" --res_x "$RES_X" --res_y "$RES_Y" "${STUDIO_ARGS[@]}"
    else
        "$BLENDER" --background --python tools/render_blender.py -- \
            --mesh "$RENDER_PLY" --scene "$SCENE" --views "$VIEW" \
            --out_dir "$OUT" --res_x "$RES_X" --res_y "$RES_Y" "${STUDIO_ARGS[@]}"
    fi
    copy_blender_gt_view "$VIEW"
done
