#!/bin/bash
#SBATCH --job-name=st-front
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_front_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/st_front_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu001,gpu002,gpu003,gpu004,gpu006,gpu007,gpu008,gpu009,gpu011,gpu014

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-
source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=outputs/run_20260604_094256_bmvs_clock_mvsformer_carved_init_4972680
OUT=$RUN/blender_st_front_step70000
SCENE=$(python -c "import json;print(json.load(open('$RUN/config.json'))['scene'])")
BLENDER="${BLENDER:-$HOME/blender/blender}"

# ---- 1. GPU sphere-trace the front face -> normalized screen mesh + camera npz
python -u analysis/_front_st_mesh.py || { echo "[error] front trace failed"; exit 1; }

# ---- 2. Blender Cycles AO render + .blend with the custom front camera
"$BLENDER" --background --python tools/render_blender.py -- \
    --mesh "$OUT/st_front_world.ply" --scene "$SCENE" \
    --cam-npz "$OUT/front_cam.npz" --normalized-mesh \
    --name front_clockwork --out_dir "$OUT" \
    --res_x 1100 --res_y 1100
