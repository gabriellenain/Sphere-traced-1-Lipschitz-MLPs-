#!/usr/bin/env python3
"""Precompute frozen MVSFormer++ feature maps for the 1-Lip trainer.

Saves a .pt file with {"features": (V, C, H, W)} using the stage4 FPN+FMT
features from DINOv2-B. Compatible with --feature-maps in train.py.

Usage:
    python precompute_mvsformer_features.py \
        --scene /scratch/.../dtu/scan65 \
        --out outputs/scan65_mvsformer_feats.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from lip_tracer.data import load_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path,
                    default=Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65"))
    ap.add_argument("--out", type=Path, default=Path("outputs/scan65_mvsformer_feats.pt"))
    ap.add_argument("--stage", default="stage4",
                    choices=["stage1", "stage2", "stage3", "stage4"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=4, help="views per forward pass to limit VRAM")
    args = ap.parse_args()

    from lip_tracer.MVS_former import MVSFormerFeatures

    views = load_views(scene=args.scene)
    imgs = views["images"].permute(0, 3, 1, 2).float()  # (V,3,H,W) keep on CPU until chunked

    # MVSFormer++ needs H,W divisible by 64
    H, W = imgs.shape[-2:]
    Hc, Wc = (H // 64) * 64, (W // 64) * 64
    dh, dw = (H - Hc) // 2, (W - Wc) // 2
    imgs = imgs[..., dh:dh + Hc, dw:dw + Wc]
    V = imgs.shape[0]
    print(f"input: {tuple(imgs.shape)}  device={args.device}  chunk={args.chunk}")

    net = MVSFormerFeatures(device=args.device)

    feat_chunks = []
    for i in range(0, V, args.chunk):
        chunk = imgs[i:i + args.chunk].to(args.device)
        n = chunk.shape[0]
        print(f"  views {i}–{i + n - 1} ...", flush=True)
        if n == 1:
            chunk = chunk.repeat(2, 1, 1, 1)  # FMT needs >= 2 views
        feats = net(chunk)
        feat_chunks.append(feats[args.stage][:n].float().cpu())  # drop padded view
        del feats, chunk
        torch.cuda.empty_cache()
    feat = torch.cat(feat_chunks, dim=0)   # (V, C, h, w)

    # upsample back to cropped image resolution so pixel coords align
    feat = F.interpolate(feat, size=(Hc, Wc), mode="bilinear", align_corners=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feat}, args.out)
    print(f"saved {args.stage} features {tuple(feat.shape)} → {args.out}")


if __name__ == "__main__":
    main()
