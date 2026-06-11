"""Upsample 384x384 monocular normals to 1600x1200 for dtu_idr scenes.

Usage: python precompute_normals_idr.py --scan 122
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=122)
    ap.add_argument("--src", type=Path,
                    default=Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu"))
    ap.add_argument("--dst", type=Path,
                    default=Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr"))
    ap.add_argument("--size", nargs=2, type=int, default=[1600, 1200],
                    metavar=("W", "H"))
    args = ap.parse_args()

    src_dir = args.src / f"scan{args.scan}"
    dst_dir = args.dst / f"scan{args.scan}" / "normals"
    dst_dir.mkdir(exist_ok=True)

    W, H = args.size
    npy_files = sorted(src_dir.glob("*_normal.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No *_normal.npy found in {src_dir}")

    print(f"Upsampling {len(npy_files)} normal maps {src_dir.name} → {dst_dir} at {W}x{H}")
    for path in npy_files:
        n = np.load(path).astype(np.float32)   # (3, H, W) in [0, 1]
        if n.shape[0] == 3:
            n = n.transpose(1, 2, 0)           # (H, W, 3)
        n = n * 2.0 - 1.0                      # decode to [-1, 1]

        # upsample each channel via PIL BILINEAR
        n_up = np.stack([
            np.array(Image.fromarray(n[..., c]).resize((W, H), Image.BILINEAR))
            for c in range(3)
        ], axis=-1)                             # (H, W, 3)

        # renormalize
        norm = np.linalg.norm(n_up, axis=-1, keepdims=True).clip(min=1e-6)
        n_up = n_up / norm

        np.save(dst_dir / path.name, n_up.astype(np.float32))
        png = ((n_up * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(png).save(dst_dir / path.with_suffix(".png").name)

    print(f"Done — saved to {dst_dir}")

if __name__ == "__main__":
    main()
