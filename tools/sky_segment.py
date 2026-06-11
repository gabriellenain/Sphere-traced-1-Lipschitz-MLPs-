"""Generate per-image "not-sky" foreground masks with SegFormer-ADE20k.

ADE20k class id 2 = "sky". Output is a uint8 PNG per image where
255 = foreground (not sky), 0 = sky. Saved under <scene>/mask/<stem>.png.

Usage:
    python tools/sky_segment.py --scene data/tnt/Barn

Also dumps a 4-view contact-sheet PNG so you can eyeball mask quality.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ADE20K_SKY = 2  # class index in ADE20k for "sky"


def segment_scene(scene: Path, model_name: str, device: str,
                  overview_views: list[int]) -> Path:
    print(f"[seg] loading {model_name}")
    proc = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()

    mask_dir = scene / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths = sorted((scene / "rgb").glob("0_*.png"))
    print(f"[seg] {len(rgb_paths)} images -> {mask_dir}")

    overview_imgs = []
    t0 = time.time()
    for i, ip in enumerate(rgb_paths):
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        with torch.no_grad():
            inputs = proc(images=img, return_tensors="pt").to(device)
            logits = model(**inputs).logits                       # (1, C, h, w)
            logits = F.interpolate(logits, size=(H, W),
                                   mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        not_sky = (pred != ADE20K_SKY).astype(np.uint8) * 255     # fg = not sky
        out = mask_dir / (ip.stem + ".png")
        imageio.imwrite(out, not_sky)
        if i in overview_views:
            overview_imgs.append((ip.stem, np.asarray(img), not_sky))
        if (i + 1) % 50 == 0 or (i + 1) == len(rgb_paths):
            dt = time.time() - t0
            print(f"  [seg] {i+1}/{len(rgb_paths)}  "
                  f"({dt/(i+1)*1000:.0f} ms/img)", flush=True)

    # contact sheet
    ov_path = scene / "mask" / "_overview.png"
    n = len(overview_imgs)
    if n:
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n), squeeze=False)
        for r, (stem, rgb, msk) in enumerate(overview_imgs):
            overlay = rgb.copy().astype(np.float32)
            overlay[msk == 0] = overlay[msk == 0] * 0.4 + np.array([0, 0, 200]) * 0.6
            axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"{stem} rgb"); axes[r, 0].axis("off")
            axes[r, 1].imshow(msk, cmap="gray"); axes[r, 1].set_title("not-sky (white)"); axes[r, 1].axis("off")
            axes[r, 2].imshow(overlay.clip(0, 255).astype(np.uint8))
            axes[r, 2].set_title("sky tinted blue"); axes[r, 2].axis("off")
        fig.tight_layout()
        fig.savefig(ov_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[seg] overview -> {ov_path}")
    return mask_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--model", type=str,
                    default="nvidia/segformer-b4-finetuned-ade-512-512")
    ap.add_argument("--overview-views", type=int, nargs="+",
                    default=[0, 96, 192, 288])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segment_scene(args.scene.resolve(), args.model, device, args.overview_views)


if __name__ == "__main__":
    main()
