#!/bin/bash
#SBATCH --job-name=1lip-init-lego
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/init_lego_%A_%a.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/init_lego_%A_%a.log
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu004

set -euo pipefail

source /scratch/_projets_/willow/1-lip-tracer/.venv/bin/activate
cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INIT_MODES=(sphere hull)
INIT_MODE="${INIT_MODE:-${INIT_MODES[$SLURM_ARRAY_TASK_ID]}}"
INIT_STEPS="${INIT_STEPS:-5000}"
INIT_BATCH="${INIT_BATCH:-2048}"
INIT_LR="${INIT_LR:-1e-3}"
HULL_RES="${HULL_RES:-512}"
HULL_MODE="${HULL_MODE:-grid}"
OCTREE_MIN_DEPTH="${OCTREE_MIN_DEPTH:-8}"

RUN_DIR="/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/init_${INIT_MODE}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN_DIR"

echo "run dir: $RUN_DIR"
echo "mode=$INIT_MODE steps=$INIT_STEPS batch=$INIT_BATCH lr=$INIT_LR hull_res=$HULL_RES hull_mode=$HULL_MODE octree_min_depth=$OCTREE_MIN_DEPTH"

export INIT_MODE INIT_STEPS INIT_BATCH INIT_LR HULL_RES HULL_MODE OCTREE_MIN_DEPTH RUN_DIR

python -u <<'PY' 2>&1 | tee "$RUN_DIR/test_init.log"
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lip_tracer.config import BLENDER_SCENE, EvalConfig, InitConfig, ModelConfig
from lip_tracer.train import fit_hull_init, fit_sphere_init
from lip_tracer.visual_hull import render_views
from lip_tracer.visualize import visualize


run_dir = Path(os.environ["RUN_DIR"])
init_mode = os.environ["INIT_MODE"]
init_cfg = InitConfig(
    init=init_mode,
    steps=int(os.environ["INIT_STEPS"]),
    batch=int(os.environ["INIT_BATCH"]),
    lr=float(os.environ["INIT_LR"]),
    hull_res=int(os.environ["HULL_RES"]),
    hull_mode=os.environ["HULL_MODE"],
    octree_min_depth=int(os.environ["OCTREE_MIN_DEPTH"]),
)
model_cfg = ModelConfig()
scene = BLENDER_SCENE

cfg = {
    "scene": str(scene),
    "init": {
        "mode": init_cfg.init,
        "steps": init_cfg.steps,
        "batch": init_cfg.batch,
        "lr": init_cfg.lr,
        "hull_res": init_cfg.hull_res,
        "hull_mode": init_cfg.hull_mode,
        "octree_min_depth": init_cfg.octree_min_depth,
    },
    "model": {
        "hidden": model_cfg.hidden,
        "depth": model_cfg.depth,
        "group_size": model_cfg.group_size,
        "activation": model_cfg.activation,
    },
}
(run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

print(f"scene: {scene}")
print(f"init mode: {init_mode}")

if init_mode == "hull":
    f = fit_hull_init(model_cfg=model_cfg, init_cfg=init_cfg, scene=scene, bound=1.5)
else:
    f, radius = fit_sphere_init(model_cfg=model_cfg, init_cfg=init_cfg, scene=scene)
    print(f"sphere radius: {radius:.4f}")

device = "cuda" if torch.cuda.is_available() else "cpu"
f = f.to(device)

ckpt = {
    "f": f.state_dict(),
    "group_size": f.group_size,
    "depth": f.depth,
    "activation": f.activation,
    "init_mode": init_mode,
}
ckpt_path = run_dir / "checkpoint_init.pt"
torch.save(ckpt, ckpt_path)
print(f"saved checkpoint -> {ckpt_path}")

eval_cfg = EvalConfig(mc_res=128, render_down=2)
visualize(f, eval_cfg=eval_cfg, use_blender=True, out=run_dir / "surface.png")

img = render_views(f, scene=scene, n_views=8, down=4)
Image.fromarray(np.asarray(img, dtype=np.uint8)).save(run_dir / "render_strip.png")
print(f"saved render strip -> {run_dir / 'render_strip.png'}")
PY
