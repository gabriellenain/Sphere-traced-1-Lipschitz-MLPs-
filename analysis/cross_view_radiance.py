#!/usr/bin/env python3
"""Cross-view radiance-consistency probe: is the lighting/shading coherent?

Idea. The SFM points are real 3D correspondences. Project each one into every
view and sample its colour. For a Lambertian surface under a *world-fixed*
light, a given point's radiance is view-independent, so its colour should be
constant across all views that see it. Any systematic spread means the scene
is NOT photometrically coherent across views.

Two distinct kinds of incoherence, reported separately:

  * exposure   -- a per-view global brightness gain (auto-exposure / vignetting
                  drift). A nuisance for MVS but easily normalised out.
  * shading    -- the residual per-point intensity spread AFTER removing the
                  per-view gain. This is the physically meaningful part: it is
                  what a camera-mounted light produces (radiance follows the
                  view), and it is what breaks the brightness-constancy
                  assumption of photometric MVS.
  * colour     -- per-point chromaticity spread: illuminant-colour / white-
                  balance drift and strong speculars.

DTU's rig carries the light on the arm with the camera, so expect a real,
measurable shading term; this quantifies how large it is for a scan.

Robust stats throughout (median / MAD): occluded projections show up as
outliers, not as visibility we can cheaply verify.

Usage:
    python analysis/cross_view_radiance.py --scene <dtu_idr/scanXX> -o OUTDIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def luma(rgb):  # rgb in [0,1], (...,3)
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def bilinear(img, uv):
    """img HxWx3 float, uv (N,2) in pixel (x,y). Returns (N,3)."""
    coords = np.vstack([uv[:, 1], uv[:, 0]])  # (row, col)
    return np.stack(
        [map_coordinates(img[..., c], coords, order=1, mode="nearest")
         for c in range(3)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="dir with image/, cameras.npz, sparse_sfm_points.txt")
    ap.add_argument("-o", "--outdir", default="outputs/radiance")
    ap.add_argument("--max-points", type=int, default=4000)
    ap.add_argument("--min-views", type=int, default=6, help="min observations per point")
    ap.add_argument("--mask", action="store_true", help="restrict samples to foreground mask")
    args = ap.parse_args()

    scene = Path(args.scene)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    cams = np.load(scene / "cameras.npz")
    n_views = sum(1 for k in cams.files if k.startswith("world_mat_") and "inv" not in k)
    img_paths = sorted(p for p in (scene / "image").glob("*.png")
                       if not p.name.startswith("._"))[:n_views]

    pts = np.loadtxt(scene / "sparse_sfm_points.txt").astype(np.float64)
    if len(pts) > args.max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), args.max_points, replace=False)]
    ph = np.c_[pts, np.ones(len(pts))]
    P = len(pts)

    # observations[p] accumulates (view, rgb) -> build arrays of NaN, fill seen.
    I = np.full((P, n_views), np.nan, dtype=np.float32)          # luma
    C = np.full((P, n_views, 2), np.nan, dtype=np.float32)       # chromaticity r,g

    for i, ipath in enumerate(img_paths):
        img = np.asarray(Image.open(ipath).convert("RGB"), np.float32) / 255.0
        H, W = img.shape[:2]
        Pmat = (cams[f"world_mat_{i}"] @ cams[f"scale_mat_{i}"])[:3]
        uvw = (Pmat @ ph.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        vis = (uvw[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        if args.mask:
            mpath = scene / "mask" / f"{i:03d}.png"
            if mpath.exists():
                m = np.asarray(Image.open(mpath).convert("L"), np.float32) / 255.0
                mv = bilinear(np.repeat(m[..., None], 3, 2), np.clip(uv, 0, [W - 1, H - 1]))[:, 0]
                vis &= mv > 0.5
        idx = np.where(vis)[0]
        rgb = bilinear(img, uv[idx])
        lum = luma(rgb)
        s = rgb.sum(1) + 1e-6
        I[idx, i] = lum
        C[idx, i, 0] = rgb[:, 0] / s
        C[idx, i, 1] = rgb[:, 1] / s

    nobs = np.sum(~np.isnan(I), axis=1)
    keep = nobs >= args.min_views
    I, C, nobs = I[keep], C[keep], nobs[keep]
    print(f"views={n_views}  points kept={keep.sum()} / {P}  (>= {args.min_views} views)")

    # --- per-view exposure gain: align each point to its own median ---------
    pt_med = np.nanmedian(I, axis=1, keepdims=True)
    ratio = I / pt_med                                   # (P, V)
    gain = np.nanmedian(ratio, axis=0)                   # (V,)  per-view gain
    gain /= np.nanmedian(gain)
    Icorr = I / gain[None, :]

    def robust_cv(X):
        med = np.nanmedian(X, axis=1)
        mad = np.nanmedian(np.abs(X - med[:, None]), axis=1)
        return 1.4826 * mad / np.maximum(med, 1e-3)

    cv_raw = robust_cv(I)
    cv_corr = robust_cv(Icorr)
    chroma_std = np.sqrt(np.nansum(np.nanvar(C, axis=1), axis=1))  # combined r,g spread

    print(f"per-view exposure gain: min {gain.min():.3f}  max {gain.max():.3f}  "
          f"(spread {100*(gain.max()/gain.min()-1):.1f}%)")
    print(f"intensity CV  raw   median {np.median(cv_raw):.3f}  p90 {np.percentile(cv_raw,90):.3f}")
    print(f"intensity CV  exp-corrected (=shading) median {np.median(cv_corr):.3f}  "
          f"p90 {np.percentile(cv_corr,90):.3f}")
    print(f"chromaticity std  median {np.median(chroma_std):.4f}  p90 {np.percentile(chroma_std,90):.4f}")

    # --- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(cv_raw, 60, alpha=.5, label="raw")
    ax[0, 0].hist(cv_corr, 60, alpha=.5, label="exposure-corrected (shading)")
    ax[0, 0].set_title("per-point intensity CV"); ax[0, 0].set_xlabel("robust CV"); ax[0, 0].legend()
    ax[0, 1].bar(range(n_views), gain)
    ax[0, 1].axhline(1, color="k", lw=.7)
    ax[0, 1].set_title("per-view exposure gain"); ax[0, 1].set_xlabel("view")
    ax[1, 0].hist(chroma_std, 60, color="C2")
    ax[1, 0].set_title("per-point chromaticity std"); ax[1, 0].set_xlabel("std")
    # example traces: highest-observation points, intensity across views
    top = np.argsort(-nobs)[:12]
    for p in top:
        v = np.where(~np.isnan(Icorr[p]))[0]
        ax[1, 1].plot(v, Icorr[p, v], "-o", ms=2, lw=.7, alpha=.7)
    ax[1, 1].set_title("12 most-seen points: intensity vs view\n(flat = coherent)")
    ax[1, 1].set_xlabel("view"); ax[1, 1].set_ylabel("exp-corrected intensity")
    fig.suptitle(f"Cross-view radiance coherence  --  {scene.name}")
    fig.tight_layout()
    fp = out / f"{scene.name}_radiance_coherence.png"
    fig.savefig(fp, dpi=110)
    print("wrote", fp)


if __name__ == "__main__":
    main()
