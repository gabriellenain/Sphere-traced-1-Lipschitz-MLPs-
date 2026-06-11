#!/usr/bin/env python3
"""First-layer activation-chart boundaries of h_theta, on the GT photograph.

The activation regions of h_theta = F_theta(gamma_L(x)) tile space into local
affine "Fourier charts". The *full* partition (all 8 ConvexPotentialLayers +
7 MaxMin layers) is sub-pixel even on a hard zoom -- ~1 chart per pixel -- so
its boundary map is uniformly black and uninformative. This script therefore
draws the *first-layer* partition: the chart structure induced by the ReLU
signs of the first ConvexPotentialLayer alone (256 boundaries g_i(x) =
(W gamma_L(x))_i + b_i = 0), which is coarse enough to read.

Two panels, both over the ground-truth photograph of a real DTU view:
  LEFT  -- the whole object; the crop of the right panel is boxed.
  RIGHT -- a zoom into a sub-window, charts resolved.

Faithful by construction: each pixel ray is the ray that formed that GT pixel,
so a boundary at a pixel is the first-layer chart of the surface point seen
through it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.patches import Rectangle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import lip_tracer.data as data_mod
from activation_charts import chunked, first_cpl_preact, load_model, sphere_trace


def crop_rays(K: np.ndarray, c2w: np.ndarray, x0: float, y0: float,
              side_w: float, side_h: float, res_w: int, res_h: int):
    """Camera rays for a res_h x res_w grid over the GT pixel box given."""
    us = x0 + (np.arange(res_w) + 0.5) * side_w / res_w
    vs = y0 + (np.arange(res_h) + 0.5) * side_h / res_h
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                      (vv - K[1, 2]) / K[1, 1],
                      np.ones_like(uu)], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape)
    return (torch.from_numpy(origins.reshape(-1, 3).astype(np.float32)),
            torch.from_numpy(dirs.reshape(-1, 3).astype(np.float32)),
            uu, vv)


def first_layer_boundary(f, origins, dirs, uu, vv, img, eps, t_far,
                         iters, chunk):
    """Trace the ray grid; return (GT crop, hit mask, boundary mask, n, depth).

    A boundary pixel is a surface hit whose 4-neighbour has a different
    first-CPL ReLU sign pattern (the 256-bit first-layer activation code).
    `depth` is ray depth |hit - camera|, NaN off-surface.
    """
    res_h, res_w = uu.shape
    H, W = img.shape[:2]
    hits, ok = sphere_trace(f, origins, dirs, eps, t_far, iters, chunk)
    ok_g = ok.numpy().reshape(res_h, res_w)
    depth = np.full(res_h * res_w, np.nan, np.float32)
    depth[ok.numpy()] = (hits[ok] - origins[ok]).norm(dim=-1).numpy()
    depth = depth.reshape(res_h, res_w)

    label = np.full(res_h * res_w, -1, np.int64)
    hpts = hits[ok]
    if len(hpts):
        code = chunked(lambda z: torch.from_numpy(
            np.packbits((first_cpl_preact(f, z) > 0).numpy(), axis=1)),
            hpts, chunk)
        _, lab = np.unique(code, axis=0, return_inverse=True)
        label[ok.numpy()] = lab
    label = label.reshape(res_h, res_w)

    bnd = np.zeros((res_h, res_w), bool)
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        diff = ok_g[a] & ok_g[b] & (label[a] != label[b])
        bnd[a] |= diff; bnd[b] |= diff

    ui = np.clip(np.floor(uu).astype(int), 0, W - 1)
    vj = np.clip(np.floor(vv).astype(int), 0, H - 1)
    n_charts = len(np.unique(label[ok_g])) if ok_g.any() else 0
    return np.clip(img[vj, ui], 0, 1), ok_g, bnd, n_charts, depth


def pick_clean_crop(ok: np.ndarray, depth: np.ndarray, win: int):
    """Grid centre of the flattest fully-converged window of side `win`.

    "Fully converged" = every ray in the window hit the surface (no holes,
    no silhouette); among those, the window with the smallest depth variance
    is the smoothest / nicest patch to zoom into.
    """
    from scipy.ndimage import uniform_filter
    okf = ok.astype(np.float64)
    frac = uniform_filter(okf, win, mode="constant", cval=0.0)
    t0 = np.where(ok, depth, 0.0).astype(np.float64)
    m1 = uniform_filter(t0, win, mode="constant", cval=0.0)
    m2 = uniform_filter(t0 * t0, win, mode="constant", cval=0.0)
    var = np.maximum(m2 - m1 * m1, 0.0)
    score = np.where(frac > 0.999, var, np.inf)
    r = win // 2 + 1                                  # keep the box in-frame
    score[:r] = score[-r:] = np.inf
    score[:, :r] = score[:, -r:] = np.inf
    if not np.isfinite(score).any():                  # fallback: most converged
        score = np.where(np.isfinite(score), score, -frac)
    gy, gx = np.unravel_index(np.argmin(score), score.shape)
    return int(gx), int(gy), float(frac[gy, gx])


def draw(ax, gt, bnd, title, lw_alpha=0.8):
    ax.imshow(gt, interpolation="bilinear")
    overlay = np.zeros((*bnd.shape, 4), np.float32)
    overlay[bnd] = (0.0, 0.0, 0.0, lw_alpha)
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title, fontsize=8.5)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, nargs="?",
                    default=Path("outputs/run_20260521_014634_scan65_4907464"))
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--view", type=int, default=16)
    ap.add_argument("--res-full", type=int, default=900,
                    help="traced grid width for the full-object panel")
    ap.add_argument("--res-zoom", type=int, default=1000,
                    help="traced grid side for the zoom panel")
    ap.add_argument("--crop-frac", type=float, default=0.15,
                    help="zoom crop side as a fraction of min(image H, W)")
    ap.add_argument("--crop-cx", type=float, default=-1.0,
                    help="crop centre x, fraction of width (-1 = mask centroid)")
    ap.add_argument("--crop-cy", type=float, default=-1.0,
                    help="crop centre y, fraction of height (-1 = mask centroid)")
    ap.add_argument("--iters", type=int, default=320)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--out", type=Path, default=Path("figs/skull_chart_boundaries.png"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    f, cfg = load_model(args.run_dir, args.ckpt, args.device)
    eps, t_far = float(cfg["trace"]["eps"]), float(cfg["trace"]["t_far"])

    scene = Path(cfg["scene"])
    data_mod.SCENE = scene
    views = data_mod.load_views(scene, down=1)        # full-resolution GT photo
    vi = args.view % int(views["images"].shape[0])
    img = views["images"][vi].numpy()
    mask = views["masks"][vi].numpy().astype(bool)
    H, W = img.shape[:2]
    K = views["K"][vi].numpy()
    c2w = views["c2w"][vi].numpy()

    side = args.crop_frac * min(H, W)
    print(f"scene={scene.name} view={vi}  GT image {H}x{W}")

    # --- LEFT: full object --------------------------------------------------
    res_h_full = int(round(args.res_full * H / W))
    o, d, uu, vv = crop_rays(K, c2w, 0.0, 0.0, W, H, args.res_full, res_h_full)
    gt_f, ok_f, bnd_f, nc_f, depth_f = first_layer_boundary(
        f, o, d, uu, vv, img, eps, t_far, args.iters, args.chunk)
    print(f"full panel: {ok_f.sum()} hits, {nc_f} first-layer charts")

    # --- crop placement: a fully-converged, flat patch ----------------------
    sx = args.res_full / W                            # grid px per GT px
    if args.crop_cx >= 0 and args.crop_cy >= 0:
        cx, cy = args.crop_cx * W, args.crop_cy * H
        print(f"zoom crop: user-specified centre ({cx:.0f},{cy:.0f})")
    else:
        gx, gy, frac = pick_clean_crop(ok_f, depth_f, int(round(side * sx)))
        cx, cy = (gx + 0.5) / sx, (gy + 0.5) / sx
        print(f"zoom crop: auto centre ({cx:.0f},{cy:.0f}), "
              f"window convergence {frac * 100:.1f}%")
    x0, y0 = cx - side / 2, cy - side / 2

    # --- RIGHT: zoom --------------------------------------------------------
    o, d, uu, vv = crop_rays(K, c2w, x0, y0, side, side,
                             args.res_zoom, args.res_zoom)
    gt_z, ok_z, bnd_z, nc_z, _ = first_layer_boundary(
        f, o, d, uu, vv, img, eps, t_far, args.iters, args.chunk)
    print(f"zoom panel: {ok_z.sum()} hits / {ok_z.size} "
          f"({100 * ok_z.sum() / ok_z.size:.1f}% converged), "
          f"{nc_z} first-layer charts")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    fig.patch.set_facecolor("white")
    draw(axes[0], gt_f, bnd_f,
         f"full object  |  {nc_f} first-layer charts", lw_alpha=0.7)
    sx, sy = args.res_full / W, res_h_full / H
    axes[0].add_patch(Rectangle((x0 * sx, y0 * sy), side * sx, side * sy,
                                fill=False, ec="#ff2d2d", lw=1.6))
    draw(axes[1], gt_z, bnd_z,
         f"zoom ({side / min(H, W) * 100:.0f}% crop)  |  {nc_z} first-layer charts",
         lw_alpha=0.85)
    fig.suptitle(
        r"First-layer Fourier-chart boundaries of $h_\theta=F_\theta(\gamma_L(x))$"
        f"  --  {scene.name} view {vi}, ground-truth photograph", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
