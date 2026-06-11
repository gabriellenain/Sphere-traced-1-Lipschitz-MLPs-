"""Smoke test: extract MVSFormer++ features on DTU scan65."""
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from lip_tracer.MVS_former import MVSFormerFeatures

SCAN = Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu/scan65/image")

def load(p: Path) -> torch.Tensor:
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)              # (3, H, W)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_views = 2 if device == "cpu" else 3
    paths = sorted(p for p in SCAN.glob("*.png") if not p.name.startswith("._"))[:n_views]
    imgs = torch.stack([load(p) for p in paths]).to(device)     # (V, 3, H, W)
    # CPU smoke test: aggressive downscale to ~256x320 to avoid OOM.
    if device == "cpu":
        imgs = torch.nn.functional.interpolate(
            imgs, size=(256, 320), mode="bilinear", align_corners=False)
    # MVSFormer++ needs H, W divisible by 64.
    H, W = imgs.shape[-2:]
    Hc, Wc = (H // 64) * 64, (W // 64) * 64
    dh, dw = (H - Hc) // 2, (W - Wc) // 2
    imgs = imgs[..., dh:dh + Hc, dw:dw + Wc]
    print("input:", imgs.shape, "range:", imgs.min().item(), imgs.max().item())

    net = MVSFormerFeatures(device=device)                      # default DTU ckpt
    feats = net(imgs)
    for k, v in feats.items():
        print(f"  {k}: {tuple(v.shape)}  mean={v.mean():.4f}  std={v.std():.4f}")

if __name__ == "__main__":
    main()
