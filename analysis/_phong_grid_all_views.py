"""Phong-shaded grid of a checkpoint sphere-traced from EVERY training view.

Reuses analysis/render_paper.py's tracer + clay shader, but takes the trace parameters
from the checkpoint's own run config.json (iters/eps/t_far/newton/...) so the
render is faithful to how the model was trained — not a prettier fine trace.
Lays all views out in a near-square grid with view-index titles.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lip_tracer.config import TraceConfig
from lip_tracer.data import load_blender_views, load_views
import render_paper as rp


def run_cfg(ckpt: Path) -> tuple[TraceConfig, str, Path]:
    """Trace params + scene path from the checkpoint's run config.json.

    The checkpoint lives in <run>/ckpt/, so config.json is one level up; fall
    back to alongside the .pt for flat layouts.
    """
    cfg_path = ckpt.parent.parent / "config.json"
    if not cfg_path.exists():
        cfg_path = ckpt.parent / "config.json"
    d = json.loads(cfg_path.read_text())
    t = d["trace"]
    cfg = TraceConfig(
        iters=t["iters"], eps=t["eps"], t_far=t["t_far"],
        eik_stride=t.get("eik_stride", 8), newton_steps=t["newton_steps"],
        grad_mode=t.get("grad_mode", "idr"),
        bsphere_radius=t.get("bsphere_radius", 0.0),
        sdf_min_beta=t.get("sdf_min_beta", 200.0),
    )
    label = (f"iters={cfg.iters} eps={cfg.eps:g} t_far={cfg.t_far:g} "
             f"newton={cfg.newton_steps}")
    scene = d.get("scene")
    return cfg, label, (Path(scene) if scene else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <run>/phong_grid_<ckptstem>.png)")
    ap.add_argument("--down", type=int, default=2,
                    help="image downsample for speed (grid cells are small)")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--cols", type=int, default=0, help="0 = near-square")
    ap.add_argument("--cell", type=float, default=2.2, help="inches per grid cell")
    args = ap.parse_args()

    cfg, cfg_label, scene = run_cfg(args.ckpt)
    if scene is None:
        raise SystemExit(f"no 'scene' in run config.json for {args.ckpt}")
    dataset = rp._infer_dataset(scene)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grid] ckpt={args.ckpt}\n[grid] scene={scene} dataset={dataset} "
          f"device={device}\n[grid] run trace: {cfg_label}", flush=True)

    f = rp.load_model(args.ckpt, device)
    views = (load_blender_views(scene=scene, split="train", down=1)
             if dataset == "lego" else load_views(scene=scene, down=args.down))
    N = views["c2w"].shape[0]
    H, W = views["H"], views["W"]
    print(f"[grid] {N} views  {H}x{W}  ss={args.ss}", flush=True)

    rp.render._normal_blur_sigma = 0.0
    shaded_imgs = []
    t0 = time.time()
    for vi in range(N):
        print(f"[grid] === view {vi}/{N-1} ===", flush=True)
        shaded, _, _, _ = rp.render(
            f, views["c2w"][vi].numpy(), views["K"][vi].numpy(),
            H, W, device, cfg, ss=args.ss, src_imgs=None,
        )
        shaded_imgs.append(np.clip(shaded, 0, 1))
    print(f"[grid] traced {N} views in {time.time()-t0:.1f}s", flush=True)

    cols = args.cols or int(math.ceil(math.sqrt(N)))
    rows = int(math.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * args.cell, rows * args.cell * H / W))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, img in enumerate(shaded_imgs):
        axes[i].imshow(img)
        axes[i].set_title(f"v{i}", fontsize=8, pad=2)
    fig.suptitle(f"{args.ckpt.parent.parent.name} / {args.ckpt.name}   "
                 f"[{cfg_label}]", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out = args.out or (args.ckpt.parent.parent /
                       f"phong_grid_{args.ckpt.stem}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[grid] wrote {out}  ({N} views, {rows}x{cols})", flush=True)


if __name__ == "__main__":
    main()
