"""Visualize depth map overlaid on a chosen view.

Two panels: depth map | reference image.

Usage (DTU skull — SDFStudio format):
  python viz_depth.py --scene .../dtu/scan65 --view 16

Usage (DTU IDR + MVSFormer):
  python viz_depth.py --scene .../dtu_idr/scan122 \
      --mvsformer-dir .../scan122/mvsformer_depth_... --view 0

Usage (Lego):
  python viz_depth.py --scene .../nerf_synthetic/lego --blender \
      --mvsformer-dir .../lego/mvsformer_depth_800x800_800x800 --view 0
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------- PFM reader ----------

def _read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        hdr = f.readline().decode().strip()
        assert hdr in ("PF", "Pf"), f"not a PFM file: {path}"
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=np.dtype(f"{endian}f4"))
        channels = 3 if hdr == "PF" else 1
        data = data.reshape(h, w, channels) if channels > 1 else data.reshape(h, w)
    return np.flipud(data).astype(np.float32)


# ---------- loaders ----------

def _load_dtu_sdfstudio(scene: Path, view: int):
    """meta_data.json format (dtu/scan*)."""
    meta = json.loads((scene / "meta_data.json").read_text())
    fr = meta["frames"][view]
    from PIL import Image as PILImage
    img = np.array(PILImage.open(scene / fr["rgb_path"])).astype(np.float32) / 255.0
    d = np.load(scene / fr["mono_depth_path"])
    if d.ndim == 3:
        d = d.squeeze()
    v = np.load(scene / fr["mono_depth_path"])   # reuse: valid where depth > 0
    valid = d > 0
    return img, d, valid


def _load_mvsformer(depth_dir: Path, view: int, conf_thr: float):
    """PFM depth + confidence from MVSFormer output dir."""
    scan_dirs = [d for d in depth_dir.iterdir() if d.is_dir() and (d / "depth_est").exists()]
    scan_root = scan_dirs[0] if scan_dirs else depth_dir
    pfm_files = sorted((scan_root / "depth_est").glob("*.pfm"))
    if not pfm_files:
        raise FileNotFoundError(f"No PFM files in {scan_root / 'depth_est'}")
    pfm = pfm_files[view]
    depth = _read_pfm(pfm)
    conf_path = scan_root / "confidence" / f"{pfm.stem}.npy"
    if conf_path.exists():
        conf_raw = np.load(conf_path)
        conf = conf_raw.astype(np.float32)
        if conf_raw.dtype == np.uint8:
            conf /= 255.0
        valid = (conf > conf_thr) & (depth > 1e-3)
    else:
        conf = np.ones_like(depth)
        valid = depth > 1e-3
    return depth, valid, conf


def _load_blender_image(scene: Path, view: int):
    from PIL import Image as PILImage
    meta = json.loads((scene / "transforms_train.json").read_text())
    fr = meta["frames"][view]
    p = scene / (fr["file_path"] + ".png")
    img = np.array(PILImage.open(p))[..., :3].astype(np.float32) / 255.0
    return img


def _load_dtu_idr_image(scene: Path, view: int):
    from PIL import Image as PILImage
    img_paths = sorted(p for p in (scene / "image").glob("*.png")
                       if not p.name.startswith("._"))
    img = np.array(PILImage.open(img_paths[view]))[..., :3].astype(np.float32) / 255.0
    return img


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--blender", action="store_true",
                    help="Blender/NeRF synthetic scene (transforms_train.json)")
    ap.add_argument("--mvsformer-dir", type=Path, default=None,
                    help="MVSFormer depth dir (contains <scan>/depth_est/*.pfm)")
    ap.add_argument("--conf-thr", type=float, default=0.7)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/viz_depth.png"))
    args = ap.parse_args()

    print(f"Scene: {args.scene}  view: {args.view}")

    # --- load image ---
    if args.blender:
        img = _load_blender_image(args.scene, args.view)
    elif (args.scene / "meta_data.json").exists():
        img, depth, valid = _load_dtu_sdfstudio(args.scene, args.view)
    else:
        img = _load_dtu_idr_image(args.scene, args.view)

    # --- load depth ---
    conf = None
    if args.mvsformer_dir is not None:
        depth, valid, conf = _load_mvsformer(args.mvsformer_dir, args.view, args.conf_thr)
    elif not (args.scene / "meta_data.json").exists():
        ap.error("Need --mvsformer-dir for IDR/Blender scenes without meta_data.json")

    depth_show = np.where(valid, depth, np.nan)
    valid_frac = valid.mean() * 100

    # --- plot ---
    ncols = 3 if hasattr(args, "mvsformer_dir") and args.mvsformer_dir is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))

    im = axes[0].imshow(depth_show, cmap="plasma")
    plt.colorbar(im, ax=axes[0], fraction=0.03, label="z-depth")
    axes[0].set_title(f"Depth (conf>{args.conf_thr}) — view {args.view}  ({valid_frac:.1f}% valid)")
    axes[0].axis("off")

    if ncols == 3:
        im2 = axes[1].imshow(conf if conf is not None else np.ones_like(depth),
                             cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im2, ax=axes[1], fraction=0.03, label="confidence")
        axes[1].set_title(f"Confidence map — view {args.view}")
        axes[1].axis("off")

    axes[-1].imshow(img)
    axes[-1].set_title(f"Reference image — view {args.view}")
    axes[-1].axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
