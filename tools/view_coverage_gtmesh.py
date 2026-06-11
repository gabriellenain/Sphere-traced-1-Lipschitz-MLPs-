"""Per-view "how many views see this pixel" coverage map — GT-mesh, occlusion-aware.

Domain = GT mask (clipped + overlaid on the RGB).
Depth   = ray-traced from the DTU GT watertight mesh (true surface, z-buffer occlusion).

For each GT-foreground pixel of a view:
  1. ray-trace the GT mesh → 3D surface point P (front-most hit).
  2. count cameras that SEE P := in front + in-frame + GT-foreground + NOT occluded,
     occlusion tested against each view's GT-mesh depth map (z-buffer).
Saves per-view overlays (coverage blended on the RGB) + a montage.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import trimesh

from lip_tracer.visual_hull import _load_masked_views


def load_gt_mesh_idr(mesh_path: Path, scene: Path):
    """Load GT mesh (DTU world mm) and map verts to the IDR-normalized frame."""
    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    sm = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
    s = float(sm[0, 0]); t = sm[:3, 3]
    v = (np.asarray(tm.vertices, np.float64) - t[None]) / s        # world(mm) -> normalized
    return trimesh.Trimesh(vertices=v, faces=tm.faces, process=False)


def pixel_rays(c2w, K, H, W):
    """World-space ray origins/dirs for every pixel center (down-sampled grid)."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xc = (xs.ravel() + 0.5 - K[0, 2]) / K[0, 0]
    yc = (ys.ravel() + 0.5 - K[1, 2]) / K[1, 1]
    dcam = np.stack([xc, yc, np.ones_like(xc)], -1)
    dirs = dcam @ R.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(t, dirs.shape).astype(np.float64).copy()
    return origins, dirs


def depth_map(mesh, c2w, K, H, W, fg):
    """Ray-trace front-most hit per FG pixel → (depth z in cam frame, 3D point).

    Returns z (H*W,) with inf where no hit / bg, and P (H*W,3)."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    O, D = pixel_rays(c2w, K, H, W)
    sel = fg.ravel()
    z = np.full(H * W, np.inf)
    P = np.full((H * W, 3), np.nan)
    if not sel.any():
        return z, P
    loc, idx_ray, _ = mesh.ray.intersects_location(O[sel], D[sel], multiple_hits=False)
    if len(idx_ray):
        sub = np.nonzero(sel)[0][idx_ray]
        P[sub] = loc
        cam = (loc - t[None]) @ R
        z[sub] = cam[:, 2]
    return z, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24")
    ap.add_argument("--mesh", default="outputs/dtu_scan024_gt_watertight.ply")
    ap.add_argument("--out", default="outputs/view_coverage_scan24_gtmesh")
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--occl-margin", type=float, default=5e-3, help="depth tol (normalized units) for occlusion")
    args = ap.parse_args()

    scene = Path(args.scene); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mesh = load_gt_mesh_idr(Path(args.mesh), scene)
    print(f"  mesh: {len(mesh.vertices):,} verts {len(mesh.faces):,} faces  "
          f"bbox={mesh.bounds[0].round(2)}..{mesh.bounds[1].round(2)}")

    v = _load_masked_views(scene)
    masks_f = v["masks"].numpy().astype(np.float32)      # full-res (V,Hf,Wf)
    imgs_f = v["images"].numpy() if "images" in v else None
    c2ws = v["c2w"].numpy(); Ks = v["K"].numpy(); Hf, Wf = v["H"], v["W"]
    V = len(c2ws); d = args.down; H, W = Hf // d, Wf // d

    # down-sampled masks + intrinsics
    masks = masks_f[:, ::d, ::d][:, :H, :W] > 0.5
    Kds = Ks.copy(); Kds[:, :2] /= d

    # 1. depth map + 3D points per view (front-most GT-mesh hit inside the mask)
    print("=== ray-tracing GT-mesh depth for all views ===")
    Z = np.full((V, H * W), np.inf); Pts = np.full((V, H * W, 3), np.nan)
    for vi in range(V):
        Z[vi], Pts[vi] = depth_map(mesh, c2ws[vi], Kds[vi], H, W, masks[vi])
        print(f"  view {vi:02d}: hits={int(np.isfinite(Z[vi]).sum()):6d} / fg {int(masks[vi].sum()):6d}")

    # 2. coverage: for each view's points, count cameras that SEE them (z-buffer occlusion)
    print("=== counting view coverage ===")
    maps = []
    for vi in range(V):
        P = Pts[vi]; valid = np.isfinite(P[:, 0])
        cnt = np.zeros(H * W, np.int32)
        for vj in range(V):
            R, t = c2ws[vj][:3, :3], c2ws[vj][:3, 3]
            K = Kds[vj]
            cam = (P - t[None]) @ R
            zc = cam[:, 2]
            front = valid & (zc > 1e-6)
            zz = np.where(front, zc, 1.0)
            u = (cam[:, 0] / zz) * K[0, 0] + K[0, 2]
            w_ = (cam[:, 1] / zz) * K[1, 1] + K[1, 2]
            ui = np.floor(u).astype(int); wi = np.floor(w_).astype(int)
            inb = front & (ui >= 0) & (ui < W) & (wi >= 0) & (wi < H)
            fg = np.zeros(H * W, bool)
            fg[inb] = masks[vj][wi[inb], ui[inb]]
            # occlusion: GT-mesh front depth at that pixel vs this point's depth
            zbuf = np.full(H * W, np.inf)
            zbuf[inb] = Z[vj].reshape(H, W)[wi[inb], ui[inb]]
            visible = fg & (zc <= zbuf + args.occl_margin)
            cnt += visible.astype(np.int32)
        cov = cnt.reshape(H, W).astype(np.float32)
        cov[~valid.reshape(H, W)] = np.nan
        maps.append(cov)
        print(f"  view {vi:02d}: cov mean={np.nanmean(cov):.1f} "
              f"max={np.nanmax(cov) if valid.any() else 0:.0f} min={np.nanmin(cov) if valid.any() else 0:.0f}")

    vmax = int(np.nanmax([np.nanmax(m) for m in maps]))

    # 3. overlay on RGB
    def rgb_of(vi):
        if imgs_f is not None:
            im = imgs_f[vi][::d, ::d][:H, :W]
            return np.clip(im, 0, 1) if im.max() <= 1.0 else im / 255.0
        return np.ones((H, W, 3)) * 0.15

    cmap = cm.get_cmap("turbo")
    for vi, cov in enumerate(maps):
        base = rgb_of(vi).copy()
        m = np.isfinite(cov)
        col = cmap(np.where(m, cov, 0) / max(vmax, 1))[..., :3]
        ov = base.copy(); a = 0.65
        ov[m] = (1 - a) * base[m] + a * col[m]
        fig, ax = plt.subplots(figsize=(6, 4.6))
        im = ax.imshow(ov); ax.axis("off")
        ax.set_title(f"scan24 v{vi:02d} — #views seeing each GT-mask pixel (GT mesh, occlusion-aware)")
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        fig.colorbar(sm, ax=ax, fraction=0.046, label="n views")
        fig.tight_layout(); fig.savefig(out / f"coverage_view{vi:02d}.png", dpi=110); plt.close(fig)

    ncol = 7; nrow = int(np.ceil(V / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 1.8))
    for i, ax in enumerate(axes.ravel()):
        if i < V:
            base = rgb_of(i); m = np.isfinite(maps[i])
            col = cmap(np.where(m, maps[i], 0) / max(vmax, 1))[..., :3]
            ov = base.copy(); ov[m] = 0.35 * base[m] + 0.65 * col[m]
            ax.imshow(ov); ax.set_title(f"v{i:02d}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"scan24 GT-mesh view-coverage (occlusion-aware, vmax={vmax})")
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axes, fraction=0.02, label="n views")
    fig.savefig(out / "coverage_montage.png", dpi=130); plt.close(fig)
    print(f"saved → {out}/  vmax={vmax}")


if __name__ == "__main__":
    main()
