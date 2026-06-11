"""One-off: backfill normals_step_*.png for an existing run's step checkpoints,
matching lip_tracer.train._dump_mc_normal_maps output exactly."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lip_tracer.data import load_views
from render_paper_marching import (extract_mesh, _make_intersector,
                                   render_normals_only)

RUN      = Path(sys.argv[1])
SCENE    = Path(sys.argv[2])
VIEWS    = [16, 32]
MC_RES   = 512
BOUND    = 1.5
DOWN     = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
views  = load_views(scene=SCENE)
V      = views["c2w"].shape[0]
ids    = [v for v in VIEWS if 0 <= v < V]
H_full, W_full = views["H"], views["W"]
d        = max(1, DOWN)
H_d, W_d = H_full // d, W_full // d

ckpts = sorted(RUN.glob("checkpoint_step_*.pt"))
print(f"device={device}  V={V}  ids={ids}  ckpts={[c.name for c in ckpts]}")

for ck in ckpts:
    step = int(re.search(r"checkpoint_step_(\d+)\.pt", ck.name).group(1))
    out  = RUN / f"normals_step_{step:06d}.png"
    mesh, _f, _vol = extract_mesh(ck, BOUND, MC_RES, device,
                                  level=0.0, keep_model=False)
    intersector = _make_intersector(mesh)
    fig, axes = plt.subplots(1, len(ids), figsize=(5 * len(ids), 5),
                             squeeze=False)
    for ax, vi in zip(axes[0], ids):
        K = views["K"][vi].numpy().copy()
        K[0, 0] /= d; K[1, 1] /= d
        K[0, 2] = (K[0, 2] + 0.5) / d - 0.5
        K[1, 2] = (K[1, 2] + 0.5) / d - 0.5
        img = render_normals_only(mesh, intersector,
                                  views["c2w"][vi].numpy(), K,
                                  H_d, W_d, 1)
        ax.imshow(np.clip(img, 0, 1)); ax.axis("off")
        ax.set_title(f"view {vi}", fontsize=10)
    fig.suptitle(f"MC normals — step {step}  (mc_res={MC_RES})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [normals] → {out.name}", flush=True)

print("done")
