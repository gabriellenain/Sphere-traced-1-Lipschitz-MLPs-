"""Per-view "how many views see this pixel" coverage map.

For each pixel of each view:
  1. cast its camera ray and intersect the VISUAL HULL (occupancy grid) → 3D surface pt
  2. count how many cameras "see" that point := in front of cam (z>0) AND in-frame
     AND inside the silhouette (mask foreground).
Saves one heatmap PNG per view + a montage. Pure geometry, no trained model.
Self-occlusion is ignored (silhouette/frustum visibility only).
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from lip_tracer.visual_hull import carve, _load_masked_views


def ray_hull_hit(occ, bound, origins, dirs, n_samples):
    """First-hit point of each ray with the occupancy grid (axes z,y,x).

    origins (R,3), dirs (R,3) unit. Returns hit (R,3) and mask (R,) bool.
    Samples t in [t0,t1] = ray/cube intersection, marks first occupied sample.
    """
    R = origins.shape[0]
    res = occ.shape[0]
    # ray vs cube [-bound,bound]^3 (slab method)
    lo, hi = -bound, bound
    inv = np.where(np.abs(dirs) < 1e-9, 1e9, 1.0 / dirs)
    t1 = (lo - origins) * inv
    t2 = (hi - origins) * inv
    tmin = np.maximum.reduce(np.minimum(t1, t2), axis=1)
    tmax = np.minimum.reduce(np.maximum(t1, t2), axis=1)
    tmin = np.clip(tmin, 0.0, None)
    valid = tmax > tmin
    hit = np.full((R, 3), np.nan, np.float32)
    if not valid.any():
        return hit, np.zeros(R, bool)
    ts = np.linspace(0, 1, n_samples)[None, :]                      # (1,S)
    seg = (tmin[:, None] + (tmax - tmin)[:, None] * ts)             # (R,S)
    pts = origins[:, None, :] + seg[:, :, None] * dirs[:, None, :]  # (R,S,3) xyz
    vox = 2 * bound / max(res - 1, 1)
    gi = np.round((pts + bound) / vox).astype(np.int64)
    ix, iy, iz = gi[..., 0], gi[..., 1], gi[..., 2]
    ok = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res) & (iz >= 0) & (iz < res)
    occ_s = np.zeros((R, n_samples), bool)
    occ_s[ok] = occ[iz[ok], iy[ok], ix[ok]]
    occ_s &= valid[:, None]
    first = np.argmax(occ_s, axis=1)                # first True (0 if none)
    has = occ_s.any(axis=1)
    rows = np.arange(R)
    hit[has] = pts[rows[has], first[has]]
    return hit, has


def count_views_seeing(pts, masks, c2ws, Ks, H, W):
    """For each 3D point, count views where it is in front + in-frame + foreground."""
    N = len(pts)
    cnt = np.zeros(N, np.int32)
    for mask, c2w, K in zip(masks, c2ws, Ks):
        R, t = c2w[:3, :3], c2w[:3, 3]
        cam = (pts - t[None]) @ R
        valid = cam[:, 2] > 1e-6
        z = np.where(valid, cam[:, 2], 1.0)
        px = (cam[:, 0] / z) * K[0, 0] + K[0, 2]
        py = (cam[:, 1] / z) * K[1, 1] + K[1, 2]
        xi = np.floor(px).astype(int)
        yi = np.floor(py).astype(int)
        ib = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & valid
        fg = np.zeros(N, bool)
        fg[ib] = mask[yi[ib], xi[ib]] > 0.5
        cnt += fg.astype(np.int32)
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24")
    ap.add_argument("--out", default="outputs/view_coverage_scan24")
    ap.add_argument("--res", type=int, default=256, help="hull carve resolution")
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--down", type=int, default=4, help="image downsample for the maps")
    ap.add_argument("--samples", type=int, default=512, help="ray-march samples")
    ap.add_argument("--border-aware", action="store_true", default=True)
    args = ap.parse_args()

    scene = Path(args.scene)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== carving hull (res={args.res}) ===")
    occ = carve(scene=scene, res=args.res, bound=args.bound, border_aware=args.border_aware)
    print(f"  occupied: {occ.sum()} / {occ.size} ({occ.mean():.4f})")

    v = _load_masked_views(scene)
    masks = v["masks"].numpy().astype(np.float32)
    c2ws = v["c2w"].numpy()
    Ks = v["K"].numpy()
    Hf, Wf = v["H"], v["W"]
    V = len(c2ws)
    d = args.down
    H, W = Hf // d, Wf // d

    maps = []
    vmax = 0
    for vi in range(V):
        c2w, K = c2ws[vi], Ks[vi].copy()
        K[:2] /= d                                  # scale intrinsics to downsampled grid
        R, t = c2w[:3, :3], c2w[:3, 3]
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        # pixel ray dirs in cam frame, then to world
        xc = (xs.ravel() + 0.5 - K[0, 2]) / K[0, 0]
        yc = (ys.ravel() + 0.5 - K[1, 2]) / K[1, 1]
        dcam = np.stack([xc, yc, np.ones_like(xc)], -1)
        dirs = dcam @ R.T
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        origins = np.broadcast_to(t, dirs.shape).astype(np.float32)
        hit, has = ray_hull_hit(occ, args.bound, origins.astype(np.float32),
                                dirs.astype(np.float32), args.samples)
        cov = np.zeros(H * W, np.int32)
        if has.any():
            cov[has] = count_views_seeing(hit[has], masks, c2ws, Ks, Hf, Wf)
        cov = cov.reshape(H, W).astype(np.float32)
        cov[~has.reshape(H, W)] = np.nan          # background = no surface
        maps.append(cov)
        vmax = max(vmax, np.nanmax(cov) if has.any() else 0)
        print(f"  view {vi:02d}: hull pixels={int(has.sum()):6d}  "
              f"cov mean={np.nanmean(cov):.1f} max={np.nanmax(cov) if has.any() else 0:.0f}")

    vmax = int(vmax) if vmax > 0 else V
    # per-view PNGs
    for vi, cov in enumerate(maps):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(cov, cmap="turbo", vmin=0, vmax=vmax)
        ax.set_title(f"scan24 view {vi:02d} — #views seeing each pixel")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="n views")
        fig.tight_layout()
        fig.savefig(out / f"coverage_view{vi:02d}.png", dpi=110)
        plt.close(fig)

    # montage
    ncol = 7
    nrow = int(np.ceil(V / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 1.8))
    for i, ax in enumerate(axes.ravel()):
        if i < V:
            im = ax.imshow(maps[i], cmap="turbo", vmin=0, vmax=vmax)
            ax.set_title(f"v{i:02d}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"scan24 view-coverage (n views seeing each hull pixel, vmax={vmax})")
    fig.tight_layout()
    fig.colorbar(im, ax=axes, fraction=0.02, label="n views")
    fig.savefig(out / "coverage_montage.png", dpi=130)
    plt.close(fig)
    print(f"saved → {out}/  ({V} views + montage)  vmax={vmax}")


if __name__ == "__main__":
    main()
