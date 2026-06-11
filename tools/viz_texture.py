"""Visualise which image regions are "textured" under the NCC patch-std criterion.

For each view, computes the per-pixel local patch std (same 5x5 patch as NCC),
overlays with the fg mask, and marks silhouette pixels (fg boundary) in red.

Usage:
    python viz_texture.py --scene data/dtu_idr/scan65 --patch 5 --std-thr 1e-4 \
                          --views 0 16 32 --out outputs/texture_viz
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def local_patch_std(img: np.ndarray, patch: int = 5) -> np.ndarray:
    """Per-pixel std over a (patch x patch) neighbourhood, per channel, then mean."""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    k = patch
    pad = k // 2
    patches = t.unfold(2, k, 1).unfold(3, k, 1)              # (1,3,H',W',k,k)
    std = patches.reshape(1, 3, *patches.shape[2:4], k * k).std(dim=-1)  # (1,3,H',W')
    std = std.mean(dim=1).squeeze(0).numpy()                  # (H', W')

    ph, pw = img.shape[0] - std.shape[0], img.shape[1] - std.shape[1]
    std = np.pad(std, ((pad, ph - pad), (pad, pw - pad)), mode='edge')
    return std


def silhouette_boundary(mask: np.ndarray, dilation: int = 3) -> np.ndarray:
    """Boolean map of fg mask boundary pixels via morphological difference."""
    from scipy.ndimage import binary_dilation, binary_erosion
    se = np.ones((dilation, dilation), dtype=bool)
    return binary_dilation(mask, se) & ~binary_erosion(mask, se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",   default="data/dtu_idr/scan65")
    ap.add_argument("--patch",   type=int,   default=5)
    ap.add_argument("--std-thr", type=float, default=1e-4)
    ap.add_argument("--views",   type=int,   nargs="+", default=[0, 16, 24, 32, 40])
    ap.add_argument("--out",     default="outputs/texture_viz")
    args = ap.parse_args()

    scene = Path(args.scene)
    out   = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_dir  = scene / "image"
    mask_dir = scene / "mask"

    img_files  = sorted(p for p in img_dir.glob("*.png")  if not p.name.startswith("._"))
    mask_files = sorted(p for p in mask_dir.glob("*.png") if not p.name.startswith("._"))

    view_ids = args.views if args.views else list(range(len(img_files)))

    for vi in view_ids:
        if vi >= len(img_files):
            print(f"view {vi} out of range ({len(img_files)} views), skipping")
            continue

        img  = np.array(Image.open(img_files[vi]).convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(Image.open(mask_files[vi]).convert("L")) > 128

        std  = local_patch_std(img, patch=args.patch)  # (H, W)
        textured = std > args.std_thr                  # True = patch passes NCC gate

        boundary = silhouette_boundary(mask)

        # ── Composite image ──────────────────────────────────────────────────
        # Background: original image darkened
        vis = (img * 0.5).copy()

        # Textured + in mask → green tint
        tex_fg = textured & mask
        vis[tex_fg] = vis[tex_fg] * 0.5 + np.array([0, 0.5, 0])

        # Untextured + in mask → magenta (these are the silent NCC zones)
        untex_fg = ~textured & mask
        vis[untex_fg] = vis[untex_fg] * 0.5 + np.array([0.5, 0, 0.5])

        # Silhouette boundary → red overlay
        vis[boundary] = [1.0, 0.0, 0.0]

        vis_u8 = (vis * 255).clip(0, 255).astype(np.uint8)

        # ── Std heatmap (log scale) ──────────────────────────────────────────
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        log_std = np.log10(std.clip(1e-8, 1) + 1e-8)
        lo = log_std.min(); hi = max(log_std.max(), lo + 1e-6)
        norm    = (log_std - lo) / (hi - lo)
        heat_c  = (plt.cm.viridis(norm)[:, :, :3] * 255).astype(np.uint8)

        thr_log  = np.log10(args.std_thr + 1e-8)
        thr_band = (log_std > thr_log - 0.1) & (log_std < thr_log + 0.1)
        heat_c[thr_band] = [255, 0, 0]

        # ── Side-by-side output ──────────────────────────────────────────────
        H, W = vis_u8.shape[:2]
        tw, th = 800, 600
        vis_r  = np.array(Image.fromarray(vis_u8).resize((tw, th)))
        heat_r = np.array(Image.fromarray(heat_c).resize((tw, th)))
        combined = np.concatenate([vis_r, heat_r], axis=1)

        out_path = out / f"view{vi:03d}_texture.png"
        Image.fromarray(combined).save(out_path)

        # ── Stats ────────────────────────────────────────────────────────────
        n_fg      = mask.sum()
        n_tex_fg  = tex_fg.sum()
        n_bnd     = boundary.sum()
        n_tex_bnd = (textured & boundary).sum()
        print(f"view {vi:3d}: fg={n_fg}  textured_fg={n_tex_fg} ({100*n_tex_fg/max(n_fg,1):.1f}%)  "
              f"boundary={n_bnd}  textured_bnd={n_tex_bnd} ({100*n_tex_bnd/max(n_bnd,1):.1f}%)")

    print(f"\noutput → {out}/")
    print("legend: green=textured+fg  magenta=untextured+fg  red=silhouette boundary")


if __name__ == "__main__":
    main()
