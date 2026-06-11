"""GT multi-view coverage of the benne (no model, no mesh, no checkpoint).

Uses the Blender ground-truth depth maps: GT depth IS exactly where each
camera's rays land on the true surface, and it is occlusion-correct by
construction (depth = first visible surface, so if the benne is hidden from a
camera that camera's pixels back-project onto the occluder, not the benne).

For every training camera we back-project its GT depth, keep the points that
fall inside a sphere placed on the benne, and report:
  (1) 3D landing points on the benne + camera centres coloured by #rays,
  (2) per-camera benne-ray count (the multi-view sparsity),
  (3) incidence-angle distribution of those rays (the viewpoint spread).

Step 1 — find the benne coords (run with no --center):
    python benne_ray_dist.py --scene /…/nerf_synthetic/lego
        → prints GT bbox/centroid, writes benne_ray_dist_overview.png

Step 2 — the coverage analysis:
    python benne_ray_dist.py --scene /…/nerf_synthetic/lego \
        --center 0.0 0.35 0.55 --radius 0.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _depth_path(scene: Path, frame_path: str) -> Path | None:
    for suffix in ("_depth_0001.png", "_depth_0029.png"):
        p = scene / f"{frame_path}{suffix}"
        if p.exists():
            return p
    return None


def backproject(scene: Path, frame: dict, fov_x: float, far: float, stride: int):
    """Return (pts (N,3), dirs (N,3), normals (N,3), cam_centre (3,)) world-space.

    Normals are estimated from the depth grid via cross-product of adjacent
    back-projected pixels (same construction as make_lego_gt_mesh faces).
    """
    dp = _depth_path(scene, frame["file_path"])
    if dp is None:
        return None
    rgba = imageio.imread(str(dp)).astype(np.float32) / 255.0
    h, w = rgba.shape[:2]
    depth = rgba[..., 0] * far
    valid = (rgba[..., 3] > 0.5) & (depth > 0.0)

    ys, xs = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing="ij")
    d_s = depth[::stride, ::stride]
    v_s = valid[::stride, ::stride]
    fx = 0.5 * w / np.tan(0.5 * fov_x)
    cx, cy = w / 2.0, h / 2.0

    dir_cam = np.stack([(xs - cx) / fx, (ys - cy) / fx, np.ones_like(xs, np.float32)], -1)
    pts_cam = dir_cam * d_s[..., None]

    c2w = np.asarray(frame["transform_matrix"], np.float32) @ np.diag([1, -1, -1, 1]).astype(np.float32)
    R, t = c2w[:3, :3], c2w[:3, 3]
    grid = pts_cam @ R.T + t                                   # (hs, ws, 3) world

    # per-pixel normal from finite differences on the world grid
    nrm = np.zeros_like(grid)
    dv = grid[2:, 1:-1] - grid[:-2, 1:-1]                       # ∂/∂y
    du = grid[1:-1, 2:] - grid[1:-1, :-2]                       # ∂/∂x
    n = np.cross(du, dv)
    nrm[1:-1, 1:-1] = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    v_s = v_s.copy()
    v_s[[0, -1], :] = False; v_s[:, [0, -1]] = False            # drop border (no normal)

    dirs = dir_cam @ R.T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-9
    return (grid[v_s].astype(np.float32), dirs[v_s].astype(np.float32),
            nrm[v_s].astype(np.float32), t.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--far", type=float, default=6.0,
                    help="depth-png R-channel scale (match archive/make_lego_gt_mesh.py: 6.0)")
    ap.add_argument("--stride", type=int, default=4, help="pixel stride for speed")
    ap.add_argument("--center", type=float, nargs=3, default=None,
                    help="benne centre (x y z). Omit → overview / use --ref-view.")
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--ref-view", type=int, default=None,
                    help="frame index: pick the benne as a pixel box in this view")
    ap.add_argument("--ref-box", type=int, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="pixel box of the benne in --ref-view (full-res coords)")
    ap.add_argument("--out", type=Path, default=Path("benne_ray_dist.png"))
    args = ap.parse_args()

    meta = json.loads((args.scene / f"transforms_{args.split}.json").read_text())
    fov_x = meta["camera_angle_x"]
    frames = meta["frames"]

    # ---- reference-view benne picking (no 3D guessing) ----
    if args.ref_view is not None:
        fr = frames[args.ref_view]
        rgb = imageio.imread(args.scene / (fr["file_path"] + ".png"))
        if args.ref_box is None:
            # show the image with a pixel grid so the box can be read off
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb)
            H, W = rgb.shape[:2]
            ax.set_xticks(np.arange(0, W, max(1, W // 20)))
            ax.set_yticks(np.arange(0, H, max(1, H // 20)))
            ax.grid(True, color="cyan", alpha=.4)
            ax.set_title(f"view {args.ref_view}: read benne pixel box (X0 Y0 X1 Y1)")
            fig.tight_layout(); fig.savefig("benne_ref.png", dpi=120)
            print("→ benne_ref.png  (read --ref-box X0 Y0 X1 Y1 off the grid, re-run)")
            return
        # back-project the box's GT depth → exact benne 3D region
        dp = _depth_path(args.scene, fr["file_path"])
        d_rgba = imageio.imread(str(dp)).astype(np.float32) / 255.0
        h, w = d_rgba.shape[:2]
        depth = d_rgba[..., 0] * args.far
        x0, y0, x1, y1 = args.ref_box
        m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True
        m &= (d_rgba[..., 3] > 0.5) & (depth > 0)
        ys, xs = np.where(m)
        fx = 0.5 * w / np.tan(0.5 * fov_x)
        pc = np.stack([(xs - w / 2) / fx * depth[ys, xs],
                       (ys - h / 2) / fx * depth[ys, xs],
                       depth[ys, xs]], -1)
        c2w = np.asarray(fr["transform_matrix"], np.float32) @ np.diag(
            [1, -1, -1, 1]).astype(np.float32)
        pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
        # robust centre/radius: median + 90th-pct distance (a few stray points
        # seen through gaps to the far body must not inflate the benne sphere)
        c = np.median(pw, axis=0)
        dists = np.linalg.norm(pw - c, axis=-1)
        r = float(np.percentile(dists, 90) * 1.1)
        args.center = c.tolist()
        args.radius = r
        print(f"benne from view {args.ref_view} box {args.ref_box}: "
              f"center={c.round(3).tolist()} radius={r:.3f} ({m.sum()} px)")

    per_cam = []  # (pts, dirs, cam_centre) per usable frame
    for fr in frames:
        out = backproject(args.scene, fr, fov_x, args.far, args.stride)
        if out is not None:
            per_cam.append(out)
    if not per_cam:
        raise SystemExit(f"No GT depth maps found under {args.scene} (split={args.split}).")

    all_pts = np.concatenate([p for p, _, _, _ in per_cam])
    lo, hi = all_pts.min(0), all_pts.max(0)
    print(f"GT surface: {len(per_cam)} cams with depth | "
          f"bbox min={lo.round(3)} max={hi.round(3)} centroid={all_pts.mean(0).round(3)}")

    if args.center is None:
        s = all_pts[::max(1, len(all_pts) // 60000)]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (i, j, k, ti) in zip(axs, [
                (0, 1, 2, "TOP  (X→, Y↑)  colour=Z"),
                (0, 2, 1, "FRONT (X→, Z↑)  colour=Y"),
                (1, 2, 0, "SIDE  (Y→, Z↑)  colour=X")]):
            sc = ax.scatter(s[:, i], s[:, j], s=1, c=s[:, k], cmap="turbo")
            ax.set_xlabel("XYZ"[i]); ax.set_ylabel("XYZ"[j])
            ax.set_title(ti); ax.set_aspect("equal"); ax.grid(True, alpha=.3)
            plt.colorbar(sc, ax=ax, shrink=.7, label="XYZ"[k])
        fig.suptitle("GT lego — read benne (x,y,z) off the grids, then --center", y=1.02)
        fig.tight_layout(); fig.savefig("benne_ray_dist_overview.png", dpi=120,
                                        bbox_inches="tight")
        print("→ benne_ray_dist_overview.png  (3 ortho views; pick --center, re-run)")
        return

    c = np.asarray(args.center, np.float32)
    V = len(per_cam)
    n_hit = np.zeros(V, int)
    cam_xyz = np.stack([cc for _, _, _, cc in per_cam])
    hits, angs = [], []
    for i, (pts, dirs, nrm, cc) in enumerate(per_cam):
        m = np.linalg.norm(pts - c, axis=-1) < args.radius
        n_hit[i] = int(m.sum())
        if n_hit[i]:
            hits.append(pts[m])
            # grazing angle: angle between the view ray and the GT surface
            # normal at the landing point. 0° = head-on (well-conditioned for
            # photometric matching), →90° = grazing (ZNCC degenerates).
            cosang = np.abs(np.clip((dirs[m] * nrm[m]).sum(-1), -1, 1))
            angs.append(np.degrees(np.arccos(cosang)))

    n_see = int((n_hit > 0).sum())
    if not n_see:
        raise SystemExit("NO camera lands in the benne sphere — widen --radius "
                          "or fix --center (see bbox/centroid above).")
    hits = np.concatenate(hits)
    angs = np.concatenate(angs)
    print(f"cameras reaching the benne: {n_see}/{V} | total benne rays={n_hit.sum()} | "
          f"mean among seers={n_hit[n_hit>0].mean():.0f} | "
          f"grazing angle deg: med={np.median(angs):.0f} "
          f"p90={np.percentile(angs,90):.0f} max={angs.max():.0f} | "
          f"frac>60°={(angs>60).mean():.0%}")

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    hs = hits[::max(1, len(hits) // 30000)]
    ax.scatter(hs[:, 0], hs[:, 1], hs[:, 2], s=2, c="tab:orange", alpha=.4)
    sc = ax.scatter(cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2],
                    c=n_hit, cmap="viridis", s=45)
    ax.scatter(*c, c="red", marker="*", s=220)
    plt.colorbar(sc, ax=ax, shrink=.6, label="# benne rays / cam")
    ax.set_title(f"{n_see}/{V} cams reach benne")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(range(V), n_hit, color="tab:blue")
    ax2.set_xlabel("camera idx"); ax2.set_ylabel("# benne rays")
    ax2.set_title("per-camera coverage (multi-view sparsity)")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(angs, bins=40, range=(0, 90), color="tab:green")
    ax3.axvline(60, color="red", ls="--", lw=1, label="60° (grazing)")
    ax3.set_xlabel("ray vs GT surface-normal angle (deg)")
    ax3.set_ylabel("# rays")
    ax3.legend()
    ax3.set_title(f"grazing distribution (n={len(angs)}, "
                  f">{60}°={(angs>60).mean():.0%})")

    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
