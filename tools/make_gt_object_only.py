"""Render-style GT image: scan122 view 50, masked, white background, tight crop."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def make_object_only(image_path: Path, mask_path: Path, out_path: Path, pad: int = 16) -> None:
    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127

    out = img.copy()
    out[~mask] = 255

    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, out.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, out.shape[1])
    cropped = out[y0:y1, x0:x1]

    Image.fromarray(cropped).save(out_path)
    print(f"wrote {out_path}  ({cropped.shape[1]}x{cropped.shape[0]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scan", default="scan122")
    p.add_argument("--view", type=int, default=50)
    p.add_argument("--root", default="baselines/NeuS/public_data")
    p.add_argument("--out", default=None)
    p.add_argument("--pad", type=int, default=16)
    args = p.parse_args()

    root = Path(args.root) / args.scan
    img_path = root / "image" / f"{args.view:06d}.png"
    mask_path = root / "mask" / f"{args.view:03d}.png"
    out_path = Path(args.out) if args.out else Path(f"{args.scan}_v{args.view}_gt_object.png")
    make_object_only(img_path, mask_path, out_path, pad=args.pad)
