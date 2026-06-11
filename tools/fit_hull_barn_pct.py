"""Visibility-aware percentile visual hull for Barn + DTU-style SDF fit + Phong.

Difference vs strict AND-intersection (`_points_inside_masks`):
  - For each voxel V, count how many cameras see it (in-FOV) -> n_view(V).
  - Count how many of those classify it foreground               -> n_fg(V).
  - Hull inclusion: n_view(V) >= min_views  AND  n_fg(V) / n_view(V) >= p.

This is the standard probabilistic / soft visual-hull remedy for many-view
noisy-mask scenarios (a small fraction of bad masks per voxel are tolerated).

Sweeps p in {1.0, 0.99, 0.95, 0.90} (cheap, all share the same agreement
grid), fits an FTheta SDF to each surviving hull via the existing
fit_to_hull, renders the training-style Phong/Colour/Hit panel for each, and
saves all four renders + a side-by-side comparison.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as _PIL

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lip_tracer.visual_hull import fit_to_hull                  # noqa: E402
from lip_tracer.train import _render_poses                      # noqa: E402


def load_views_with_masks(scene: Path, down: int = 2) -> dict:
    K_raw = np.loadtxt(scene / "intrinsics.txt", dtype=np.float32)[:3, :3]
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float32)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale  = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    pose_paths = sorted((scene / "pose").glob("0_*.txt"))
    imgs, c2ws, Ks, masks = [], [], [], []
    for pp in pose_paths:
        ip = scene / "rgb"  / (pp.stem + ".png")
        mp = scene / "mask" / (pp.stem + ".png")
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
            img = np.array(_PIL.fromarray((img*255).astype(np.uint8)).resize((W, H), _PIL.BILINEAR)).astype(np.float32) / 255.0
            msk = np.array(_PIL.fromarray((msk*255).astype(np.uint8)).resize((W, H), _PIL.NEAREST)).astype(np.float32) / 255.0
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


def visibility_aware_counts(pts: np.ndarray, masks: np.ndarray,
                            c2ws: np.ndarray, Ks: np.ndarray,
                            H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_view, n_fg) of shape (N,):
       n_view = how many cameras have V in front of them and inside FOV
       n_fg   = how many of those classify V as foreground."""
    N = len(pts)
    n_view = np.zeros(N, dtype=np.int32)
    n_fg   = np.zeros(N, dtype=np.int32)
    for mask, c2w, K in zip(masks, c2ws, Ks):
        R, t = c2w[:3, :3], c2w[:3, 3]
        cam = (pts - t[None]) @ R
        z = cam[:, 2]
        valid = z > 1e-3
        zz = np.where(valid, z, 1.0)
        px = (cam[:, 0] / zz) * K[0, 0] + K[0, 2]
        py = (cam[:, 1] / zz) * K[1, 1] + K[1, 2]
        xi = np.floor(px).astype(np.int32)
        yi = np.floor(py).astype(np.int32)
        in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & valid
        n_view += in_bounds.astype(np.int32)
        if in_bounds.any():
            is_fg = np.zeros(N, dtype=bool)
            is_fg[in_bounds] = mask[yi[in_bounds], xi[in_bounds]]
            n_fg += is_fg.astype(np.int32)
    return n_view, n_fg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",       type=Path, required=True)
    ap.add_argument("--out-dir",     type=Path, required=True)
    ap.add_argument("--res",         type=int,   default=192)
    ap.add_argument("--bound",       type=float, default=1.5)
    ap.add_argument("--down",        type=int,   default=2)
    ap.add_argument("--fit-steps",   type=int,   default=2000)
    ap.add_argument("--min-views",   type=int,   default=8,
                    help="discard voxels seen by fewer than this many cameras")
    ap.add_argument("--thresholds",  type=float, nargs="+",
                    default=[1.00, 0.99, 0.95, 0.90])
    ap.add_argument("--best-thresh", type=float, default=0.95,
                    help="threshold to use for the saved init.pt checkpoint")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "render").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    views = load_views_with_masks(args.scene.resolve(), down=args.down)
    V, H, W = views["c2w"].shape[0], views["H"], views["W"]

    # build grid
    lin = np.linspace(-args.bound, args.bound, args.res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    masks_np = views["masks"].numpy().astype(np.float32)
    c2ws_np  = views["c2w"].numpy()
    Ks_np    = views["K"].numpy()

    print(f"[count] {pts.shape[0]} voxels x {V} masks ...")
    t0 = time.time()
    n_view, n_fg = visibility_aware_counts(pts, masks_np, c2ws_np, Ks_np, H, W)
    print(f"[count] {time.time()-t0:.1f}s  "
          f"n_view (per voxel): min={n_view.min()} med={int(np.median(n_view))} "
          f"max={n_view.max()}")

    # avoid divide-by-zero; mark unseen voxels as agree=0
    ratio = np.where(n_view > 0, n_fg / np.maximum(n_view, 1), 0.0)

    results = []   # (p, occ_frac, render_path)
    for p in args.thresholds:
        inside = (n_view >= args.min_views) & (ratio >= p)
        occ = inside.reshape(args.res, args.res, args.res)
        frac = occ.mean()
        print(f"[hull] p>={p:.2f}  occ={frac:.2%}  voxels={int(occ.sum())}")
        if occ.sum() == 0:
            print("  empty — skipping fit")
            results.append((p, frac, None))
            continue
        f = fit_to_hull(occ, bound=args.bound, steps=args.fit_steps,
                        cam_origins=c2ws_np[:, :3, 3], w_cam_free=0.5)
        f = f.to(device)
        step_tag = int(round(p * 100))
        _render_poses(f, views, step=step_tag, run_dir=args.out_dir, device=device)
        results.append((p, frac, args.out_dir / "render" / f"render_{step_tag:05d}.png"))
        if abs(p - args.best_thresh) < 1e-6:
            torch.save({"f": f.state_dict()}, args.out_dir / "init.pt")
            print(f"[ckpt] -> {args.out_dir/'init.pt'}")

    # composite comparison: one column per threshold, top-row Phong only
    cols = [r for r in results if r[2] is not None]
    if cols:
        fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5),
                                 squeeze=False)
        for ax, (p, frac, rp) in zip(axes[0], cols):
            img = imageio.imread(rp)
            # take just the top row (Phong) of the 3-row training-style panel
            Hp, Wp = img.shape[:2]
            ax.imshow(img[: Hp // 3])
            ax.set_title(f"p ≥ {p:.2f}    occ = {frac:.2%}", fontsize=11)
            ax.axis("off")
        fig.suptitle("Visibility-aware percentile visual hull on Barn — Phong only",
                     fontsize=13)
        fig.tight_layout()
        out = args.out_dir / "compare.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
