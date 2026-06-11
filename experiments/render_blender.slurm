#!/bin/bash
#SBATCH --job-name=render-bl
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_blender_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_blender_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

# Renders a mesh via Blender Cycles (GPU on a gpu node) for paper-grade figures.
# Usage:
#   sbatch render_blender.slurm <mesh.ply> <scene_dir> [views] [out_dir] [samples]
# Defaults: views=0,8,16,32,40  out_dir=outputs/blender_render_<jobid>  samples=128
#
# Studio look via env (same names as render_st_blender.slurm):
#   STUDIO=1 STUDIO_STYLE=clay|marble STUDIO_FILL=.. STUDIO_BG=.. STUDIO_SAMPLES=..
#   STUDIO_KEEP_FLOATERS=1  STUDIO_WHITE_BG=1  STUDIO_TRANSPARENT=1  NORMALIZED_MESH=1
# Swap mode (render <mesh.ply> into an existing .blend, reusing its camera/lights/bg):
#   SWAP_INTO=<base.blend>  [SOLIDIFY=0.012] [ALREADY_NORM=1]  -> render_blend_swap_mesh.py
#   In swap mode OUT is treated as the output PNG path (a sibling .blend is saved).

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

BLENDER=${BLENDER:-$HOME/blender/blender}
if [[ ! -x "$BLENDER" ]]; then
    echo "ERROR: Blender not found at $BLENDER. Install once via:"
    echo "  mkdir -p ~/blender && cd ~/blender && \\"
    echo "  wget -q https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz && \\"
    echo "  tar xf blender-4.2.3-linux-x64.tar.xz --strip-components=1 && rm blender-4.2.3-linux-x64.tar.xz"
    exit 1
fi

MESH="${1:?usage: sbatch render_blender.slurm <mesh.ply> <scene_dir> [views] [out_dir] [samples]}"
SCENE="${2:?need scene dir, e.g. /scratch/.../dtu/scan65}"
VIEWS="${3:-0,8,16,32,40}"
OUT="${4:-outputs/blender_render_${SLURM_JOB_ID}}"
SAMPLES="${5:-128}"

# In swap mode OUT is the output PNG *file*; otherwise it's the output dir.
if [[ -n "${SWAP_INTO:-}" ]]; then mkdir -p "$(dirname "$OUT")"; else mkdir -p "$OUT"; fi
echo "mesh:    $MESH"
echo "scene:   $SCENE"
echo "views:   $VIEWS"
echo "out:     $OUT"
echo "samples: $SAMPLES"

RES_ARGS=()
[ -n "$RES_X" ] && RES_ARGS+=(--res_x "$RES_X")
[ -n "$RES_Y" ] && RES_ARGS+=(--res_y "$RES_Y")

# ---- swap mode: drop <mesh> into an existing .blend (camera/lights/backdrop reused)
if [[ -n "${SWAP_INTO:-}" ]]; then
    SWAP_ARGS=(--blend "$SWAP_INTO" --new-mesh "$MESH" --scene-dir "$SCENE" \
               --out "$OUT" --save-blend "${OUT%.png}.blend")
    [[ -n "${SOLIDIFY:-}" ]] && SWAP_ARGS+=(--solidify "$SOLIDIFY")
    [[ "${ALREADY_NORM:-0}" == "1" ]] && SWAP_ARGS+=(--already-normalized)
    # EXTRA_MESH: colon-separated neighbour PLYs to depth-composite behind the primary.
    # BACK_OFFSET: how far (normalised units) to push them back (default 0.004).
    if [[ -n "${EXTRA_MESH:-}" ]]; then
        IFS=':' read -ra _EM <<< "$EXTRA_MESH"
        for _ep in "${_EM[@]}"; do SWAP_ARGS+=(--extra-mesh "$_ep"); done
        SWAP_ARGS+=(--back-offset "${BACK_OFFSET:-0.004}")
    fi
    echo "swap: $MESH -> $SWAP_INTO  (out $OUT)"
    "$BLENDER" -b --python tools/render_blend_swap_mesh.py -- "${SWAP_ARGS[@]}"
    exit $?
fi

# ---- studio look (off unless STUDIO=1), mirrors render_st_blender.slurm
STUDIO_ARGS=()
if [[ "${STUDIO:-0}" == "1" ]]; then
    STUDIO_ARGS=(--studio --studio-samples "${STUDIO_SAMPLES:-$SAMPLES}" \
                 --studio-fill "${STUDIO_FILL:-0.82}" --studio-bg "${STUDIO_BG:-0.21}" \
                 --studio-style "${STUDIO_STYLE:-marble}")
    [[ "${STUDIO_KEEP_FLOATERS:-0}" == "1" ]] && STUDIO_ARGS+=(--studio-keep-floaters)
    [[ "${STUDIO_WHITE_BG:-0}" == "1" ]]      && STUDIO_ARGS+=(--studio-white-bg)
    [[ -n "${STUDIO_SOLIDIFY:-}" ]]           && STUDIO_ARGS+=(--studio-solidify "$STUDIO_SOLIDIFY")
    [[ -n "${STUDIO_TRANSPARENT:-}" ]]        && STUDIO_ARGS+=(--studio-transparent)
fi
[[ "${NORMALIZED_MESH:-0}" == "1" ]] && STUDIO_ARGS+=(--normalized-mesh)

"$BLENDER" --background --python tools/render_blender.py -- \
    --mesh "$MESH" \
    --scene "$SCENE" \
    --views "$VIEWS" \
    --out_dir "$OUT" \
    "${RES_ARGS[@]}" "${STUDIO_ARGS[@]}"
