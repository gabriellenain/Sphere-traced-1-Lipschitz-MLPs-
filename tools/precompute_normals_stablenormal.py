"""Run StableNormal (YOSO variant) on IDR-style DTU images at full resolution.

Outputs per-frame normals in camera space as .npy (H, W, 3) float32, saved to
<scene>/normals_stablenormal/000000_normal.npy etc.

Install stablenormal on first run automatically.

Usage:
    python precompute_normals_stablenormal.py --scene <idr_scan_dir>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STABLENORMAL_DIR  = Path("/scratch/glenain/StableNormal")
STABLENORMAL_REPO = "https://github.com/Stable-X/StableNormal.git"
HF_MODELS_DIR     = Path("/scratch/glenain/hf_models")


def ensure_stablenormal():
    if not STABLENORMAL_DIR.exists():
        print(f"Cloning StableNormal → {STABLENORMAL_DIR} …")
        subprocess.run(["git", "clone", STABLENORMAL_REPO, str(STABLENORMAL_DIR)], check=True)
    if str(STABLENORMAL_DIR) not in sys.path:
        sys.path.insert(0, str(STABLENORMAL_DIR))


def load_pipeline(device: str):
    ensure_stablenormal()
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
    from stablenormal.pipeline_stablenormal import StableNormalPipeline

    yoso = YOSONormalsPipeline.from_pretrained(
        "Stable-X/yoso-normal-v1-8-1", trust_remote_code=True,
        variant="fp16", torch_dtype=torch.float16,
    ).to(device)

    pipe = StableNormalPipeline.from_pretrained(
        "Stable-X/stable-normal-v0-1", trust_remote_code=True,
        variant="fp16", torch_dtype=torch.float16,
        scheduler=yoso.scheduler, yoso_version=yoso,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_stablenormal(pipe, img_pil: Image.Image) -> np.ndarray:
    """Returns (H, W, 3) float32 camera-space normals in [-1, 1]."""
    out = pipe(img_pil, match_input_res=True)
    normal = out.prediction  # (H, W, 3) float32, already [-1,1]
    return normal.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    scene: Path = args.scene
    device = args.device if torch.cuda.is_available() else "cpu"

    pipe = load_pipeline(device)

    img_paths = sorted(p for p in (scene / "image").iterdir()
                       if p.suffix.lower() in {".png", ".jpg"}
                       and not p.name.startswith("._"))

    out_dir = scene / "normals_stablenormal"
    out_dir.mkdir(exist_ok=True)
    print(f"Running StableNormal (YOSO) on {len(img_paths)} images → {out_dir}")

    for i, img_path in enumerate(img_paths):
        img_pil = Image.open(img_path).convert("RGB")
        normal = run_stablenormal(pipe, img_pil)  # (H, W, 3) in [-1, 1]

        stem = f"{i:06d}_normal"
        np.save(out_dir / f"{stem}.npy", normal)
        png = ((normal * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(png).save(out_dir / f"{stem}.png")

        if i % 8 == 0:
            print(f"  [{i}/{len(img_paths)}] {img_path.name}  shape={normal.shape}")

    print(f"Done — {len(img_paths)} normal maps saved to {out_dir}")


if __name__ == "__main__":
    main()
