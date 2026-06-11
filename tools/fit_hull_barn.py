"""Reuse the DTU --init hull code path on Barn with our sky-seg masks, then
render the resulting SDF with the same Phong/Colour/Hit panel training uses.

Pipeline (all logic reused from lip_tracer):
  - load Barn images + sky-seg masks from data/tnt/Barn/mask/
  - carve voxels with _points_inside_masks  (same as carve() but for NSVF format)
  - fit_to_hull(occ, ...) -> FTheta         (identical to DTU --init hull)
  - _render_poses(f, views, step=0, ...)    (same render training writes periodically)

Output: <out_dir>/render/render_00000.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image as _PIL

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lip_tracer.visual_hull import _points_inside_masks, fit_to_hull   # noqa: E402
from lip_tracer.train import _render_poses                             # noqa: E402


def load_views_with_masks(scene: Path, down: int = 2) -> dict:
    K_raw = np.loadtxt(scene / "intrinsics.txt", dtype=np.float32)[:3, :3]
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float32)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale  = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    mask_dir = scene / "mask"
    pose_paths = sorted((scene / "pose").glob("0_*.txt"))
    imgs, c2ws, Ks, masks = [], [], [], []
    for pp in pose_paths:
        ip = scene / "rgb" / (pp.stem + ".png")
        mp = mask_dir / (pp.stem + ".png")
        if not (ip.exists() and mp.exists()):
            continue
        img = imageio.imread(ip).astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        msk = (np.asarray(imageio.imread(mp)) > 127).astype(np.float32)
        c2w = np.loadtxt(pp, dtype=np.float32).reshape(4, 4)
        c2w[:3, 3] = (c2w[:3, 3] - center) / scale
        K = K_raw.copy()
        if down > 1:
            H0, W0 = img.shape[:2]; H, W = H0 // down, W0 // down
            img = np.array(_PIL.fromarray((img * 255).astype(np.uint8)).resize(
                (W, H), _PIL.BILINEAR)).astype(np.float32) / 255.0
            msk = np.array(_PIL.fromarray((msk * 255).astype(np.uint8)).resize(
                (W, H), _PIL.NEAREST)).astype(np.float32) / 255.0
            K[0] /= down; K[1] /= down
        imgs.append(img); c2ws.append(c2w); Ks.append(K); masks.append(msk > 0.5)
    H, W = imgs[0].shape[:2]
    print(f"[load] {len(imgs)} views  ({H}x{W})  bbox_scale={scale:.3f}")
    return {
        "images": torch.from_numpy(np.stack(imgs)),
        "masks":  torch.from_numpy(np.stack(masks)),
        "c2w":    torch.from_numpy(np.stack(c2ws)),
        "K":      torch.from_numpy(np.stack(Ks)),
        "H": H, "W": W,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",   type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--res",     type=int,   default=192)
    ap.add_argument("--bound",   type=float, default=1.5)   # same default as DTU
    ap.add_argument("--down",    type=int,   default=2)
    ap.add_argument("--steps",   type=int,   default=2000)  # fit_to_hull default
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "render").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    views = load_views_with_masks(args.scene.resolve(), down=args.down)

    # --- carve ---
    lin = np.linspace(-args.bound, args.bound, args.res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    masks_np = views["masks"].numpy().astype(np.float32)
    print(f"[carve] {args.res}^3 = {pts.shape[0]} pts vs {len(masks_np)} masks ...")
    t0 = time.time()
    inside = _points_inside_masks(pts, masks_np,
                                  views["c2w"].numpy(), views["K"].numpy(),
                                  views["H"], views["W"])
    occ = inside.reshape(args.res, args.res, args.res)
    print(f"[carve] {time.time()-t0:.1f}s  occ={occ.mean():.1%}")
    if occ.sum() == 0:
        raise RuntimeError("empty hull")

    # --- fit (identical to DTU --init hull) ---
    cam_origins = views["c2w"][:, :3, 3].numpy()
    f = fit_to_hull(occ, bound=args.bound, steps=args.steps,
                    cam_origins=cam_origins)
    f = f.to(device)

    # save checkpoint so it can be used as --resume init for lip_tracer.train
    torch.save({"f": f.state_dict()}, args.out_dir / "init.pt")
    print(f"[ckpt] -> {args.out_dir/'init.pt'}")

    # --- render the training-style Phong / Colour / Hit panel ---
    _render_poses(f, views, step=0, run_dir=args.out_dir, device=device)


if __name__ == "__main__":
    main()
