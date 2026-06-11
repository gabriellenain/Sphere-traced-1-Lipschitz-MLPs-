"""Diagnose why the sky-mask visual hull on Barn looks loose / slab-like.

Steps:
  1. (Re)generate sky masks via SegFormer if scene/mask is missing or stale.
  2. Per-mask quality stats: fg fraction distribution, skyline-row distribution.
  3. Per-voxel "agreement count" = how many of the V masks include this voxel.
     Carve at multiple thresholds (100%, 99%, 95%, 90%, 80%) and report
     occupancy at each.  If occupancy rises sharply when relaxing from 100%
     to 95%, a small number of bad masks are over-carving.
  4. Render Phong from one view at each threshold so you can SEE the effect.
  5. Save a single composite diagnostic PNG.

Run via diag_hull_masks.slurm (needs GPU for SegFormer + sphere tracing).
"""
from __future__ import annotations

import argparse
import subprocess
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

from lip_tracer.visual_hull import fit_to_hull                      # noqa: E402
from lip_tracer.train import _render_poses                          # noqa: E402


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
        if img.ndim == 3 and img.shape[-1] == 4: img = img[..., :3]
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
    return {
        "images": torch.from_numpy(np.stack(imgs)),
        "masks":  torch.from_numpy(np.stack(masks)),
        "c2w":    torch.from_numpy(np.stack(c2ws)),
        "K":      torch.from_numpy(np.stack(Ks)),
        "H": H, "W": W,
    }


def count_agreement(pts: np.ndarray, masks: np.ndarray, c2ws: np.ndarray,
                    Ks: np.ndarray, H: int, W: int,
                    front_only: bool = True) -> np.ndarray:
    """For each 3D point, count how many masks classify it as foreground.
    Points behind a camera contribute 0 from that camera by default (=
    'no information' — but they're not counted as fg).  This matches
    visual-hull convention (a voxel must be IN every cone)."""
    n_pts = len(pts)
    agree = np.zeros(n_pts, dtype=np.int32)
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
        if not in_bounds.any():
            continue
        is_fg = np.zeros(n_pts, dtype=bool)
        is_fg[in_bounds] = mask[yi[in_bounds], xi[in_bounds]]
        agree += is_fg.astype(np.int32)
    return agree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",   type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--res",     type=int,   default=160)
    ap.add_argument("--bound",   type=float, default=1.5)
    ap.add_argument("--down",    type=int,   default=2)
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--n-mask-samples", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "render").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- (1) (re)gen sky masks if needed ----
    mask_dir = args.scene / "mask"
    needs_seg = not (mask_dir / "_overview.png").exists() or \
                any("sky" not in (mask_dir / f).read_text(errors="ignore")[:0]
                    for f in [])
    # simpler heuristic: trust if overview exists. We'll force re-gen via flag.
    if not (mask_dir / "_overview.png").exists():
        print("[seg] running SegFormer sky-segmentation")
        subprocess.run([sys.executable, str(ROOT / "tools" / "sky_segment.py"),
                        "--scene", str(args.scene)], check=True)

    # ---- (2) load views + masks ----
    views = load_views_with_masks(args.scene.resolve(), down=args.down)
    V = views["c2w"].shape[0]
    H, W = views["H"], views["W"]
    print(f"[diag] {V} views ({H}x{W})")

    # per-mask stats: fg fraction, skyline-row distribution
    masks_np = views["masks"].numpy()
    fg_fracs = masks_np.reshape(V, -1).mean(axis=1)
    # skyline row = first row from top where mask==1
    skyline_rows = []
    for m in masks_np:
        first = np.argmax(m.any(axis=1)) if m.any() else H
        skyline_rows.append(first)
    skyline_rows = np.array(skyline_rows)

    print(f"[masks] fg fraction: min={fg_fracs.min():.0%} med={np.median(fg_fracs):.0%} max={fg_fracs.max():.0%}")
    print(f"[masks] skyline row: min={skyline_rows.min()} med={int(np.median(skyline_rows))} max={skyline_rows.max()} (H={H})")

    # ---- (3) build agreement grid + threshold sweep ----
    lin = np.linspace(-args.bound, args.bound, args.res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    print(f"[agree] counting fg-agreement for {pts.shape[0]} voxels x {V} masks ...")
    t0 = time.time()
    agree = count_agreement(pts, masks_np.astype(np.float32),
                            views["c2w"].numpy(), views["K"].numpy(), H, W)
    print(f"[agree] {time.time()-t0:.1f}s   max={agree.max()}/V={V}")

    thresholds = [1.00, 0.99, 0.95, 0.90, 0.80, 0.50]
    occ_at_thresh = {}
    for thr in thresholds:
        cutoff = int(np.ceil(thr * V))
        occ = (agree >= cutoff).reshape(args.res, args.res, args.res)
        occ_at_thresh[thr] = (occ, occ.mean(), cutoff)
        print(f"[thresh] >= {thr:.2f} ({cutoff}/{V} masks): occ={occ.mean():.1%}")

    # ---- (4) Fit + render Phong at each threshold ----
    cam_origins = views["c2w"][:, :3, 3].numpy()
    panels = []   # (label, phong_image)
    for thr in thresholds:
        occ, occ_frac, cutoff = occ_at_thresh[thr]
        if occ.sum() == 0:
            panels.append((f"thr={thr:.2f}  EMPTY", None))
            continue
        print(f"[fit] threshold {thr:.2f}  occ={occ_frac:.1%}  fitting ...")
        f = fit_to_hull(occ, bound=args.bound, steps=args.fit_steps,
                        cam_origins=cam_origins, w_cam_free=0.5)
        f = f.to(device)
        # write phong only via _render_poses; rename after
        _render_poses(f, views, step=int(round(thr*100)), run_dir=args.out_dir, device=device)
        phong = imageio.imread(args.out_dir / "render" / f"render_{int(round(thr*100)):05d}.png")
        panels.append((f"thr≥{thr:.2f}  ({cutoff}/{V})  occ={occ_frac:.1%}", phong))

    # ---- (5) compose the diagnostic page ----
    n_mask_samples = args.n_mask_samples
    sample_ids = np.linspace(0, V - 1, n_mask_samples, dtype=int)
    n_th_with_render = sum(p[1] is not None for p in panels)

    fig = plt.figure(figsize=(22, 5 * (1 + n_th_with_render) + 4))
    nrows = 3 + n_th_with_render          # row0: hist, row1: mask samples top, row2: mask samples bottom, then phongs
    # row 0: histograms
    ax0a = fig.add_subplot(nrows, 3, 1)
    ax0a.hist(fg_fracs, bins=30); ax0a.set_title(f"per-mask fg fraction (mean={fg_fracs.mean():.0%})")
    ax0a.set_xlabel("fg / total"); ax0a.set_ylabel("# masks")
    ax0b = fig.add_subplot(nrows, 3, 2)
    ax0b.hist(skyline_rows, bins=30); ax0b.set_title(f"skyline row index (H={H})")
    ax0b.set_xlabel("row of highest fg pixel"); ax0b.set_ylabel("# masks")
    ax0c = fig.add_subplot(nrows, 3, 3)
    ax0c.plot([t*100 for t,_ in occ_at_thresh.items()],
              [d[1]*100 for _,d in occ_at_thresh.items()], marker="o")
    ax0c.set_xlabel("agreement threshold (%)"); ax0c.set_ylabel("hull occupancy (%)")
    ax0c.set_title("occupancy vs mask-agreement threshold")
    ax0c.invert_xaxis(); ax0c.grid(True)

    # rows 1-2: sample masks
    for col, vi in enumerate(sample_ids):
        ax = fig.add_subplot(nrows, n_mask_samples, n_mask_samples + col + 1)
        ax.imshow(views["images"][vi].numpy()); ax.axis("off")
        ax.set_title(f"view {vi}", fontsize=8)
        ax = fig.add_subplot(nrows, n_mask_samples, 2*n_mask_samples + col + 1)
        rgb = (views["images"][vi].numpy() * 255).astype(np.uint8)
        msk = masks_np[vi]
        ov = rgb.copy().astype(np.float32)
        ov[~msk] = ov[~msk] * 0.3 + np.array([20, 70, 220]) * 0.7
        ax.imshow(ov.clip(0,255).astype(np.uint8)); ax.axis("off")
        ax.set_title(f"sky tinted blue ({(~msk).mean():.0%} sky)", fontsize=8)

    # remaining rows: phong renders at each threshold
    for row_idx, (lbl, ph) in enumerate(p for p in panels if p[1] is not None):
        ax = fig.add_subplot(nrows, 1, 3 + row_idx + 1)
        ax.imshow(ph); ax.axis("off"); ax.set_title(lbl, fontsize=11)

    fig.suptitle("sky-mask visual-hull diagnostics — Barn", fontsize=13)
    fig.tight_layout()
    out = args.out_dir / "diag.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
