"""Visualise mask vs sphere-tracer disagreements per view.

Colours:
  green   = hit & fg   (correct)
  dark    = miss & bg  (correct background)
  RED     = miss & fg  (false negative — IDR fn, "holes")
  YELLOW  = hit & ~fg  (false positive — surface outside mask)

Usage:
    python viz_mask_errors.py \
        --pt outputs/run_20260510_094943_scan65/checkpoint_final.pt \
        --views 16 0 24 32 40 --down 2 --out outputs/mask_error_viz
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.config import TraceConfig
from lip_tracer.data import load_views
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt",    required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 16, 24, 32, 40])
    ap.add_argument("--down",  type=int, default=2)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out",   default="outputs/mask_error_viz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ── checkpoint ──────────────────────────────────────────────────────────
    ckpt = torch.load(args.pt, map_location="cpu")
    cfg: dict = ckpt.get("config", {})
    if not cfg:
        cfg_path = Path(args.pt).parent / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
    mcfg = cfg.get("model", {})
    trace_cfg = TraceConfig(
        **{k: v for k, v in cfg.get("trace", {}).items()
           if k in TraceConfig.__dataclass_fields__}
    )

    f = make_model(
        hidden=mcfg.get("hidden", 256), depth=mcfg.get("depth", 8),
        group_size=mcfg.get("group_size", 2),
        activation=mcfg.get("activation", "groupsort"),
        input_encoding=mcfg.get("input_encoding", "pe"),
        multires=mcfg.get("multires", 6),
        architecture=mcfg.get("architecture", "cpl"),
    ).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    f.eval()
    for p in f.parameters(): p.requires_grad_(False)
    print(f"loaded  t_far={trace_cfg.t_far}  iters={trace_cfg.iters}")

    # ── scene ────────────────────────────────────────────────────────────────
    if args.scene is None:
        args.scene = cfg["scene"]
    views   = load_views(Path(args.scene), down=args.down)
    images  = views["images"].to(device)
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    K_all   = views["K"].to(device)
    c2w_all = views["c2w"].to(device)
    w2c_all = torch.linalg.inv(c2w_all)
    masks   = views["masks"].to(device)   # (V, H, W) bool
    H, W    = images.shape[1], images.shape[2]
    origins = c2w_all[:, :3, 3]
    print(f"scene: {images.shape[0]} views  {H}×{W}")

    for vi in args.views:
        print(f"\nview {vi} …")

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        px = xs.reshape(-1); py = ys.reshape(-1); N = px.shape[0]

        d_cam = torch.stack([
            (px - K_all[vi, 0, 2]) / K_all[vi, 0, 0],
            (py - K_all[vi, 1, 2]) / K_all[vi, 1, 1],
            torch.ones(N, device=device),
        ], dim=-1)
        R_cw    = c2w_all[vi, :3, :3]
        d_world = F.normalize(d_cam @ R_cw.T, dim=-1)
        o       = origins[vi].unsqueeze(0).expand(N, 3)

        hit_map = torch.zeros(N, dtype=torch.bool, device=device)
        for s in range(0, N, args.batch):
            e = min(s + args.batch, N)
            with torch.no_grad():
                _, _, hit = trace_nograd(f, o[s:e], d_world[s:e], trace_cfg)
            hit_map[s:e] = hit

        hit_img = hit_map.reshape(H, W).cpu().numpy()
        fg_img  = masks[vi].cpu().numpy()

        tp = hit_img  &  fg_img   # correct hit
        fn = ~hit_img &  fg_img   # false negative (hole)
        fp =  hit_img & ~fg_img   # false positive (outside mask)
        tn = ~hit_img & ~fg_img   # correct background

        # ── colour image ────────────────────────────────────────────────────
        orig = images[vi].cpu().numpy()
        vis  = np.zeros((H, W, 3), dtype=np.float32)
        vis[tn] = orig[tn] * 0.15                          # dark background
        vis[tp] = orig[tp] * 0.5 + np.array([0, 0.4, 0])  # green tint
        vis[fn] = [1.0, 0.0, 0.0]                          # pure red
        vis[fp] = [1.0, 1.0, 0.0]                          # pure yellow

        vis_u8 = (vis * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(vis_u8).save(out / f"view{vi:03d}_mask_err.png")

        n_fn = fn.sum(); n_fp = fp.sum(); n_tp = tp.sum()
        iou  = n_tp / max(tp.sum() + fn.sum() + fp.sum(), 1)
        print(f"  tp={n_tp}  fn={n_fn} ({100*n_fn/max(fg_img.sum(),1):.1f}% of fg)  "
              f"fp={n_fp}  IoU={iou:.4f}")

    print(f"\nlegend: green=correct  RED=hole(fn)  YELLOW=fp  dark=background")
    print(f"output → {out}/")


if __name__ == "__main__":
    main()
