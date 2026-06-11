"""DTU multi-view coverage of a region (nose / cheek / back-of-skull, …).

GT-based, no model. Uses the official DTU GT point cloud (stlXXX_total.ply) +
the scan's camera poses. A region is given as a sphere (read off the overview).
For every camera we test, via a splatted z-buffer over the GT cloud, whether
the region is the front-most surface (occlusion-correct), and measure the
grazing angle of the view onto the region's GT normal.

Tells you, per region, whether a persistent hole is:
  - under-observed   (few cameras see it)            → irrecoverable by photo
  - grazing-dominated (seen only at oblique angles)   → ill-conditioned (b≠0)
  - well-covered & frontal                            → cause is elsewhere

Step 1 (find region coords):
    python dtu_coverage.py --scene /…/dtu_idr/scan65 \
        --gt-ply "/…/MVS Data/Points/stl/stl065_total.ply"

Step 2 (per region):
    python dtu_coverage.py --scene … --gt-ply … \
        --name nose --center 0.1 -0.2 0.3 --radius 0.12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import rq

from compare_dtu_chamfer import _load_point_cloud


def load_dtu_cameras(scene: Path):
    """Camera centres / rotations / K in the *normalised* frame (no images)."""
    cam = np.load(scene / "cameras.npz")
    idxs = sorted(int(k.split("_")[-1]) for k in cam.files
                  if k.startswith("world_mat_") and "inv" not in k)
    Cs, Rs, Ks = [], [], []
    for i in idxs:
        P = cam[f"world_mat_{i}"][:3, :4].astype(np.float64)
        K, R = rq(P[:, :3])
        S = np.diag(np.sign(np.diag(K)));  S[S == 0] = 1.0
        K = K @ S;  R = S @ R
        if np.linalg.det(R) < 0:
            K[:, 2] *= -1.0;  R[2, :] *= -1.0
        K = (K / K[2, 2])
        t = np.linalg.solve(K, P[:, 3])
        c = -R.T @ t                              # camera centre, RAW DTU frame
        Cs.append(c);  Rs.append(R);  Ks.append(K)
    return (np.asarray(Cs, np.float64), np.asarray(Rs, np.float64),
            np.asarray(Ks, np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--gt-ply", type=Path, required=True)
    ap.add_argument("--name", default="region")
    ap.add_argument("--center", type=float, nargs=3, default=None)
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--ref-view", type=int, default=None,
                    help="pick the region as a pixel box in this camera's image")
    ap.add_argument("--ref-box", type=int, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--img-hw", type=int, nargs=2, default=[1200, 1600],
                    metavar=("H", "W"), help="DTU image size (for in-frame test)")
    ap.add_argument("--buf-down", type=int, default=8, help="z-buffer downsample")
    ap.add_argument("--vis-tol", type=float, default=0.01,
                    help="depth tolerance (normalised units) for front-most test")
    ap.add_argument("--max-pts", type=int, default=500000,
                    help="subsample GT cloud for the occlusion buffer")
    ap.add_argument("--out", type=Path, default=Path("dtu_coverage.png"))
    args = ap.parse_args()

    C, R, K = load_dtu_cameras(args.scene)
    gt = _load_point_cloud(args.gt_ply).astype(np.float64)   # RAW DTU frame (= world_mat)
    V = C.shape[0]
    H, W = args.img_hw
    lo, hi = gt.min(0), gt.max(0)
    scene_scale = float(np.linalg.norm(hi - lo))            # bbox diagonal (raw units)
    vis_abs = args.vis_tol * scene_scale                    # --vis-tol is a fraction
    print(f"DTU GT: {len(gt)} pts | {V} cams | bbox min={lo.round(2)} "
          f"max={hi.round(2)} diag={scene_scale:.1f}")

    # ---- reference-view pixel-box region picking (robust on the total cloud) ----
    if args.ref_view is not None:
        import imageio.v2 as imageio
        imgs = sorted(p for p in (args.scene / "image").glob("*.png")
                      if not p.name.startswith("._"))
        rgb = imageio.imread(imgs[args.ref_view])[..., :3]
        Hi, Wi = rgb.shape[:2]
        if args.ref_box is None:
            fig, ax = plt.subplots(figsize=(9, 7)); ax.imshow(rgb)
            ax.set_xticks(np.arange(0, Wi, max(1, Wi // 20)))
            ax.set_yticks(np.arange(0, Hi, max(1, Hi // 20)))
            ax.grid(True, color="cyan", alpha=.4)
            ax.set_title(f"view {args.ref_view}: read --ref-box X0 Y0 X1 Y1")
            fig.tight_layout(); fig.savefig("dtu_ref.png", dpi=120)
            print("→ dtu_ref.png  (read --ref-box off the grid, re-run)")
            return
        Rv, cv, Kv = R[args.ref_view], C[args.ref_view], K[args.ref_view]
        xc = (gt - cv) @ Rv.T
        z = xc[:, 2]
        uv = xc @ Kv.T
        uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)
        x0, y0, x1, y1 = args.ref_box
        inbox = (z > 0) & (uv[:, 0] >= x0) & (uv[:, 0] < x1) \
            & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        if inbox.sum() < 20:
            raise SystemExit(f"only {int(inbox.sum())} GT pts in box — widen it")
        # keep the front-most sheet in the box (drop platform/background behind)
        zb = z[inbox]
        zmin = np.percentile(zb, 5)
        keep = inbox.copy()
        keep[inbox] = zb < zmin + 0.10 * scene_scale  # front sheet only (skull, not platform)
        pw = gt[keep]
        args.center = np.median(pw, axis=0)
        args.radius = float(np.percentile(np.linalg.norm(pw - args.center, axis=1), 90) * 1.1)
        print(f"[{args.name}] from view {args.ref_view} box {args.ref_box}: "
              f"center={np.round(args.center,3).tolist()} radius={args.radius:.3f} "
              f"({int(keep.sum())} pts)")

    if args.center is None:
        s = gt[::max(1, len(gt) // 60000)]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (i, j, k, ti) in zip(axs, [
                (0, 1, 2, "TOP (X→,Y↑) c=Z"), (0, 2, 1, "FRONT (X→,Z↑) c=Y"),
                (1, 2, 0, "SIDE (Y→,Z↑) c=X")]):
            sc = ax.scatter(s[:, i], s[:, j], s=1, c=s[:, k], cmap="turbo")
            ax.set_xlabel("XYZ"[i]); ax.set_ylabel("XYZ"[j])
            ax.set_aspect("equal"); ax.grid(True, alpha=.3); ax.set_title(ti)
            plt.colorbar(sc, ax=ax, shrink=.7)
        fig.suptitle("DTU GT — read region (x,y,z) off the grids, then --center", y=1.02)
        fig.tight_layout(); fig.savefig("dtu_coverage_overview.png", dpi=120,
                                        bbox_inches="tight")
        print("→ dtu_coverage_overview.png  (pick --name --center --radius, re-run)")
        return

    c = np.asarray(args.center)
    reg = gt[np.linalg.norm(gt - c, axis=1) < args.radius]
    if len(reg) < 20:
        raise SystemExit(f"only {len(reg)} GT pts in the sphere — fix --center/--radius")
    # region GT normal via PCA (smallest-variance axis)
    d = reg - reg.mean(0)
    nrm = np.linalg.eigh(d.T @ d)[1][:, 0]
    nrm /= np.linalg.norm(nrm) + 1e-9

    gt_buf = gt if len(gt) <= args.max_pts else \
        gt[np.random.default_rng(0).choice(len(gt), args.max_pts, replace=False)]
    bH, bW = H // args.buf_down, W // args.buf_down

    n_vis = np.zeros(V, int)
    angs = []
    for v in range(V):
        Rv, cv, Kv = R[v], C[v], K[v]
        def proj(X):
            xc = (X - cv) @ Rv.T               # world→cam
            z = xc[:, 2]
            uv = (xc @ Kv.T)
            uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)
            return uv, z
        # z-buffer from full cloud
        uvb, zb = proj(gt_buf)
        infr = (zb > 0) & (uvb[:, 0] >= 0) & (uvb[:, 0] < W) \
            & (uvb[:, 1] >= 0) & (uvb[:, 1] < H)
        px = np.clip((uvb[infr, 0] / args.buf_down).astype(int), 0, bW - 1)
        py = np.clip((uvb[infr, 1] / args.buf_down).astype(int), 0, bH - 1)
        zf = zb[infr]
        buf = np.full(bH * bW, np.inf)
        flat = py * bW + px
        np.minimum.at(buf, flat, zf)
        buf = buf.reshape(bH, bW)
        # region points front-most?
        uvr, zr = proj(reg)
        inr = (zr > 0) & (uvr[:, 0] >= 0) & (uvr[:, 0] < W) \
            & (uvr[:, 1] >= 0) & (uvr[:, 1] < H)
        if not inr.any():
            continue
        rx = np.clip((uvr[inr, 0] / args.buf_down).astype(int), 0, bW - 1)
        ry = np.clip((uvr[inr, 1] / args.buf_down).astype(int), 0, bH - 1)
        front = zr[inr] <= buf[ry, rx] + vis_abs
        k = int(front.sum())
        n_vis[v] = k
        if k:
            view = cv - reg[inr][front]
            view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
            cosang = np.abs(np.clip(view @ nrm, -1, 1))
            angs.append(np.degrees(np.arccos(cosang)))

    seers = n_vis > 0
    n_see = int(seers.sum())
    if not n_see:
        raise SystemExit("NO camera sees the region front-most — check --center/--radius/--vis-tol")
    angs = np.concatenate(angs)
    frac60 = float((angs > 60).mean())
    print(f"[{args.name}] cams seeing region: {n_see}/{V} ({n_see/V:.0%}) | "
          f"region pts={len(reg)} | grazing deg med={np.median(angs):.0f} "
          f"p90={np.percentile(angs,90):.0f} | frac>60°={frac60:.0%}")
    verdict = ("UNDER-OBSERVED → irrecoverable by photometry"
               if n_see / V < 0.15 else
               "GRAZING-DOMINATED → ill-conditioned (b≠0)"
               if frac60 > 0.5 else
               "well-covered & frontal → cause is elsewhere (tracer/loss)")
    print(f"[{args.name}] VERDICT: {verdict}")

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    rs = reg[::max(1, len(reg) // 8000)]
    ax.scatter(rs[:, 0], rs[:, 1], rs[:, 2], s=3, c="tab:orange", alpha=.5)
    sc = ax.scatter(C[:, 0], C[:, 1], C[:, 2], c=n_vis, cmap="viridis", s=40)
    ax.scatter(*c, c="red", marker="*", s=220)
    plt.colorbar(sc, ax=ax, shrink=.6, label="# region pts seen / cam")
    ax.set_title(f"[{args.name}] {n_see}/{V} cams")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(range(V), n_vis, color="tab:blue")
    ax2.set_xlabel("camera idx"); ax2.set_ylabel("# region pts seen")
    ax2.set_title("per-camera coverage")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(angs, bins=40, range=(0, 90), color="tab:green")
    ax3.axvline(60, color="red", ls="--", lw=1)
    ax3.set_xlabel("ray vs GT-normal angle (deg)")
    ax3.set_title(f"grazing (>60°={frac60:.0%})")

    fig.suptitle(f"{args.name}: {verdict}", y=1.03)
    fig.tight_layout(); fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
