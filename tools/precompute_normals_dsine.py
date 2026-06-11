"""Run DSINE monocular normal estimation on IDR-style DTU images at full resolution.

Outputs per-frame normals in camera space as .npy (H, W, 3) float32, saved to
<scene>/normals_dsine/000000_normal.npy etc.

DSINE is cloned on first run if --dsine-dir is not found. Weights are downloaded
from HuggingFace (baegwangbin/DSINE) on first run.

Usage:
    python precompute_normals_dsine.py --scene <idr_scan_dir>
    python precompute_normals_dsine.py --scene .../scan122 --dsine-dir /scratch/glenain/DSINE
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DSINE_REPO  = "https://github.com/baegwangbin/DSINE.git"
DSINE_CKPT_URL = "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
CKPT_NAME   = "dsine.pt"


def get_dsine(dsine_dir: Path) -> Path:
    if not dsine_dir.exists():
        print(f"Cloning DSINE → {dsine_dir} …")
        subprocess.run(["git", "clone", DSINE_REPO, str(dsine_dir)], check=True)
    ckpt = dsine_dir / CKPT_NAME
    if not ckpt.exists():
        print(f"Downloading DSINE weights from {DSINE_CKPT_URL} …")
        import urllib.request
        urllib.request.urlretrieve(DSINE_CKPT_URL, ckpt)
    return dsine_dir


def load_dsine_model(dsine_dir: Path, device: str) -> torch.nn.Module:
    sys.path.insert(0, str(dsine_dir))
    from models.dsine import DSINE  # type: ignore
    model = DSINE()
    ckpt = torch.load(dsine_dir / CKPT_NAME, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def run_dsine(model, img_np: np.ndarray, K: np.ndarray, device: str) -> np.ndarray:
    """img_np: (H, W, 3) float32 [0,1].  K: (3,3) intrinsics.
    Returns normals (H, W, 3) float32 in camera space, z points into scene."""
    import torchvision.transforms.functional as TF
    H, W = img_np.shape[:2]
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    # DSINE expects intrinsics as (1,3,3)
    K_t = torch.from_numpy(K[None].astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(img_t, intrins=K_t)  # (1,3,H,W), unit normals in cam space
    normal = out[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    return normal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--dsine-dir", type=Path,
                    default=Path("/scratch/glenain/DSINE"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    scene: Path = args.scene
    device = args.device if torch.cuda.is_available() else "cpu"

    dsine_dir = get_dsine(args.dsine_dir)
    model = load_dsine_model(dsine_dir, device)

    cam_dict = np.load(scene / "cameras.npz")
    img_paths = sorted(p for p in (scene / "image").iterdir()
                       if p.suffix.lower() in {".png", ".jpg"}
                       and not p.name.startswith("._"))

    out_dir = scene / "normals_dsine"
    out_dir.mkdir(exist_ok=True)
    print(f"Running DSINE on {len(img_paths)} images → {out_dir}")

    from scipy.linalg import rq
    for i, img_path in enumerate(img_paths):
        img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        H, W = img_np.shape[:2]

        # extract K at original resolution from cameras.npz
        P = cam_dict[f"world_mat_{i}"][:3, :4].astype(np.float64)
        K_cam, R_cam = rq(P[:, :3])
        s = np.sign(np.diag(K_cam)); s[s == 0] = 1.0
        K_cam = K_cam @ np.diag(s)
        K_cam /= K_cam[2, 2]

        normal = run_dsine(model, img_np, K_cam.astype(np.float32), device)  # (H,W,3)

        stem = f"{i:06d}_normal"
        np.save(out_dir / f"{stem}.npy", normal)
        png = ((normal * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(png).save(out_dir / f"{stem}.png")

        if i % 8 == 0:
            print(f"  [{i}/{len(img_paths)}] {img_path.name}  shape={normal.shape}")

    print(f"Done — {len(img_paths)} normal maps saved to {out_dir}")


if __name__ == "__main__":
    main()
