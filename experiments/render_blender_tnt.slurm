#!/bin/bash
#SBATCH --job-name=render-bl-tnt
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_blender_tnt_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/render_blender_tnt_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=4:00:00

# Same flow as render_blender.slurm but for TnT: extract a world-frame mesh
# from a checkpoint via marching cubes -> run Blender Cycles -> save .blend.
# Usage:
#   sbatch render_blender_tnt.slurm <ckpt.pt> [SceneName] [views] [out_dir]
# Defaults: SceneName=Barn  views=0,80,160,240  out_dir=outputs/blender_render_tnt_<jobid>

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate

CKPT="${1:?usage: sbatch render_blender_tnt.slurm <ckpt.pt> [SceneName] [views] [out_dir]}"
SCENE_NAME="${2:-Barn}"
VIEWS="${3:-0,80,160,240}"
OUT="${4:-outputs/blender_render_tnt_${SLURM_JOB_ID}}"
RES=${RES:-512}
BOUND=${BOUND:-1.5}

SCENE_DIR=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/data/tnt/${SCENE_NAME}
BLENDER=${BLENDER:-$HOME/blender/blender}

if [[ ! -x "$BLENDER" ]]; then
    echo "ERROR: Blender not found at $BLENDER. Install once via:"
    echo "  mkdir -p ~/blender && cd ~/blender && \\"
    echo "  wget -q https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz && \\"
    echo "  tar xf blender-4.2.3-linux-x64.tar.xz --strip-components=1 && rm blender-4.2.3-linux-x64.tar.xz"
    exit 1
fi
[ -f "$CKPT" ]      || { echo "[error] ckpt not found: $CKPT";       exit 1; }
[ -d "$SCENE_DIR" ] || { echo "[error] scene dir not found: $SCENE_DIR"; exit 1; }

mkdir -p "$OUT"
MESH=$OUT/pred_world_mesh_tnt.ply

echo "ckpt:      $CKPT"
echo "scene:     $SCENE_DIR  (name=$SCENE_NAME)"
echo "views:     $VIEWS"
echo "out:       $OUT"
echo "mc res:    $RES   bound: $BOUND"

# ---- 1. Extract world-frame mesh (re-uses analysis/eval_tnt_official.py's MC+un-norm step)
python -u - <<EOF
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from eval_tnt_official import extract_mesh_from_ckpt
import open3d as o3d

mesh = extract_mesh_from_ckpt(
    Path("$CKPT"),
    Path("$SCENE_DIR"),
    mc_res=$RES,
    bound=$BOUND,
)
o3d.io.write_triangle_mesh("$MESH", mesh)
print(f"[mesh] wrote $MESH  verts={len(mesh.vertices):,}", flush=True)
EOF

# ---- 2. Blender Cycles render + .blend save
"$BLENDER" --background --python tools/render_blender_tnt.py -- \
    --mesh    "$MESH" \
    --scene   "$SCENE_DIR" \
    --views   "$VIEWS" \
    --out_dir "$OUT"
