#!/usr/bin/env python3
"""Visualise the per-pixel filters the MVSFormer++ sphere-carve init applies.

For a few reference views it shows, side by side:
  RGB | depth | confidence (prob) | photo mask (prob>τ_conf) |
  geo votes (#consistent src views) | COHERENT depth (photo & votes>=N)

The "coherent" pixels are exactly the ones that survive the init's two filters:
  * photometric   : confidence > --conf-thr   (carve flag --conf-thr)
  * geometric vote : depth reprojects consistently in >= --votes-req source views
                     (carve flag --votes-req; classic MVSNet reprojection check
                      with --pix-thr px and --depth-thr relative-depth tolerance)

NOTE on confidence scale: MVSFormer++ saves confidence as (prob*255) uint8, so a
threshold of 0.5 against the raw array is a no-op (min ~40). This script
normalises conf to prob in [0,1] before thresholding, so --conf-thr is a real
probability. Pass --conf-thr 0 to reproduce the carve's effective behaviour
(photo filter disabled; geometric votes do all the work).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        color = header == "PF"
        w, h = map(int, f.readline().decode("utf-8").split())
        scale = float(f.readline().decode("utf-8").rstrip())
        data = np.fromfile(f, "<f" if scale < 0 else ">f")
        data = data.reshape(h, w, 3 if color else 1)
        data = np.flipud(data)
    return data[..., 0] if not color else data


def read_cam(path: Path):
    """Return (extrinsic 4x4 world->cam, intrinsic 3x3)."""
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", path.read_text())
    nums = [float(x) for x in nums]
    extr = np.array(nums[0:16], dtype=np.float64).reshape(4, 4)
    intr = np.array(nums[16:25], dtype=np.float64).reshape(3, 3)
    return extr, intr


def read_pairs(pair_txt: Path, n_src: int) -> dict[int, list[int]]:
    lines = pair_txt.read_text().strip().splitlines()
    n = int(lines[0])
    out: dict[int, list[int]] = {}
    for k in range(n):
        ref = int(lines[1 + 2 * k])
        toks = lines[2 + 2 * k].split()
        srcs = [int(toks[1 + 2 * j]) for j in range(int(toks[0]))]
        out[ref] = srcs[:n_src]
    return out


def reproject(depth_ref, K_ref, E_ref, depth_src, K_src, E_src):
    """Reproject ref pixels through src and back; return reproj depth + xy in ref."""
    h, w = depth_ref.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xr, yr = xx.reshape(-1), yy.reshape(-1)
    ones = np.ones_like(xr, dtype=np.float64)

    # ref pixel -> ref cam -> world
    xyz_ref = np.linalg.inv(K_ref) @ (np.vstack((xr, yr, ones)) * depth_ref.reshape(-1))
    xyz_w = np.linalg.inv(E_ref) @ np.vstack((xyz_ref, ones))
    # world -> src cam -> src pixel
    xyz_s = (E_src @ xyz_w)[:3]
    K_xyz_s = K_src @ xyz_s
    xy_s = K_xyz_s[:2] / K_xyz_s[2:3]
    x_s = xy_s[0].reshape(h, w)
    y_s = xy_s[1].reshape(h, w)
    # sample src depth at projected location
    d_s = map_coordinates(depth_src, [y_s.ravel(), x_s.ravel()], order=1,
                          mode="constant", cval=0.0).reshape(-1)
    # src pixel -> src cam -> world -> ref cam
    xyz_s2 = np.linalg.inv(K_src) @ (np.vstack((xy_s, ones)) * d_s)
    xyz_w2 = np.linalg.inv(E_src) @ np.vstack((xyz_s2, ones))
    xyz_r2 = (E_ref @ xyz_w2)[:3]
    depth_reproj = xyz_r2[2].reshape(h, w)
    K_xyz_r2 = K_ref @ xyz_r2
    xy_r2 = K_xyz_r2[:2] / K_xyz_r2[2:3]
    return depth_reproj, xy_r2[0].reshape(h, w), xy_r2[1].reshape(h, w)


def geo_votes(depth_ref, K_ref, E_ref, srcs, depths, Ks, Es, pix_thr, depth_thr):
    h, w = depth_ref.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    votes = np.zeros((h, w), dtype=np.int32)
    for s in srcs:
        dr, xr2, yr2 = reproject(depth_ref, K_ref, E_ref, depths[s], Ks[s], Es[s])
        dist = np.sqrt((xr2 - xx) ** 2 + (yr2 - yy) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(dr - depth_ref) / np.maximum(depth_ref, 1e-6)
        votes += ((dist < pix_thr) & (rel < depth_thr) & (dr > 0)).astype(np.int32)
    return votes


def depth_only_view(args):
    """One clean depth PNG per view, masked to valid (confident) pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dd = args.depth_dir
    n_total = len(list((dd / "depth_est").glob("*.pfm")))
    if args.views is None:
        args.views = np.linspace(0, n_total - 1, args.n_views, dtype=int).tolist()
    out_dir = args.out or (dd.parent.parent / "depth_valid_pixels")
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in args.views:
        depth = read_pfm(dd / "depth_est" / f"{v:08d}.pfm").astype(np.float64)
        conf = np.load(dd / "confidence" / f"{v:08d}.npy").astype(np.float64) / 255.0
        dmask = depth > 1e-3
        valid = dmask & (conf > args.conf_thr)
        vlo, vhi = np.nanpercentile(depth[dmask], [2, 98])
        full_depth = np.where(dmask, depth, np.nan)
        valid_depth = np.where(valid, depth, np.nan)
        kept = 100.0 * valid.sum() / depth.size

        h, w = depth.shape
        fig, axes = plt.subplots(1, 3, figsize=(3 * w / 130, h / 130),
                                 facecolor="none")
        panels = [
            (full_depth, "turbo", vlo, vhi, "depth (m)",
             "MVSFormer++ depth"),
            (conf, "turbo", 0.0, 1.0, "confidence (prob)",
             "confidence"),
            (valid_depth, "turbo", vlo, vhi, "depth (m)",
             f"valid (prob>{args.conf_thr:g}) · {kept:.0f}%"),
        ]
        for ax, (img, cmap, lo, hi, cblab, title) in zip(axes, panels):
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("none")
            for s in ax.spines.values():
                s.set_visible(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(cblab, fontsize=7); cb.ax.tick_params(labelsize=6)
            ax.set_title(title, fontsize=9)
        fig.suptitle(f"view {v}", fontsize=10, y=1.02)
        out = out_dir / f"depth_view{v:04d}_valid.png"
        fig.savefig(out, dpi=160, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"[viz] view {v}: {kept:.1f}% valid  -> {out}")


def confidence_view(args):
    """Per view: RGB | depth | confidence(prob) | confident pixels at thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    dd = args.depth_dir
    n_total = len(list((dd / "depth_est").glob("*.pfm")))
    if args.views is None:
        args.views = np.linspace(0, n_total - 1, args.n_views, dtype=int).tolist()
    thrs = [0.3, 0.5, 0.7]
    BG = "#0d0d0d"
    cols = ["RGB", "depth (m)", "confidence (prob)"] + [f"prob>{t:g}" for t in thrs]
    nr, nc = len(args.views), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.0 * nr),
                             facecolor=BG, squeeze=False)
    for r, v in enumerate(args.views):
        depth = read_pfm(dd / "depth_est" / f"{v:08d}.pfm").astype(np.float64)
        conf = np.load(dd / "confidence" / f"{v:08d}.npy").astype(np.float64) / 255.0
        valid = depth > 1e-3
        dvis = np.where(valid, depth, np.nan)
        vlo, vhi = np.nanpercentile(dvis, [2, 98])
        rgb = np.asarray(Image.open(dd / "images" / f"{v:08d}.jpg"))
        panels = [(rgb, None, None, None), (dvis, vlo, vhi, "turbo"),
                  (conf, 0.0, 1.0, "turbo")]
        panels += [(conf, t, None, "mask") for t in thrs]
        for c, (img, lo, hi, cmap) in enumerate(panels):
            ax = axes[r][c]; ax.set_facecolor(BG)
            if cmap is None:
                ax.imshow(img)
            elif cmap == "mask":
                ax.imshow(np.where(img > lo, depth, np.nan), cmap="turbo",
                          vmin=vlo, vmax=vhi)
            else:
                ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], color="white", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"view {v}", color="white", fontsize=9)
            if cmap == "mask":
                kept = 100.0 * ((img > lo) & valid).sum() / max(valid.sum(), 1)
                ax.set_xlabel(f"{kept:.0f}% conf", color="white", fontsize=8)
            if c == 2:
                ax.set_xlabel(f"median prob {np.median(conf[valid]):.2f}",
                              color="white", fontsize=8)
    fig.suptitle(f"MVSFormer++ confidence · {dd.parent.parent.name}  "
                 f"(depth masked to confident pixels)",
                 color="white", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (dd.parent.parent / "mvsformer_confidence_pixels.png")
    fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
    print(f"[viz] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    root = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-")
    ap.add_argument("--depth-dir", type=Path,
                    default=root / "_diagnostics/mvsformer_meetingroom/"
                                   "depths_1152x640/Meetingroom")
    ap.add_argument("--pair-txt", type=Path,
                    default=root / "_diagnostics/mvsformer_meetingroom/"
                                   "staging/Meetingroom/pair.txt")
    ap.add_argument("--views", type=int, nargs="*", default=None,
                    help="ref view indices; default = 4 evenly spaced")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--conf-thr", type=float, default=0.5, help="prob in [0,1]")
    ap.add_argument("--votes-req", type=int, default=3)
    ap.add_argument("--n-src", type=int, default=10)
    ap.add_argument("--pix-thr", type=float, default=1.0)
    ap.add_argument("--depth-thr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--confidence-only", action="store_true",
                    help="simple view: RGB | depth | confidence(prob) | "
                         "confident pixels at a few thresholds (no geo votes)")
    ap.add_argument("--depth-only", action="store_true",
                    help="write one clean depth PNG per view, masked to valid "
                         "(confident, prob>--conf-thr) pixels; invalid transparent")
    args = ap.parse_args()

    if args.depth_only:
        return depth_only_view(args)
    if args.confidence_only:
        return confidence_view(args)

    dd = args.depth_dir
    n_total = len(list((dd / "depth_est").glob("*.pfm")))
    pairs = read_pairs(args.pair_txt, args.n_src)
    if args.views is None:
        args.views = np.linspace(0, n_total - 1, args.n_views, dtype=int).tolist()

    # we only need to load depths/cams for the ref views and their sources
    needed = set(args.views)
    for v in args.views:
        needed.update(pairs.get(v, []))
    depths, Ks, Es = {}, {}, {}
    for i in sorted(needed):
        depths[i] = read_pfm(dd / "depth_est" / f"{i:08d}.pfm").astype(np.float64)
        E, K = read_cam(dd / "cams" / f"{i:08d}_cam.txt")
        Es[i], Ks[i] = E, K

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    BG = "#0d0d0d"
    cols = ["RGB", "depth (m)", "confidence (prob)",
            f"photo  prob>{args.conf_thr:g}",
            f"geo votes (/{args.n_src})",
            f"COHERENT  (photo & votes>={args.votes_req})"]
    nr, nc = len(args.views), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.0 * nr),
                             facecolor=BG, squeeze=False)

    for r, v in enumerate(args.views):
        depth = depths[v]
        conf = np.load(dd / "confidence" / f"{v:08d}.npy").astype(np.float64) / 255.0
        valid = depth > 1e-3
        photo = (conf > args.conf_thr) & valid
        votes = geo_votes(depth, Ks[v], Es[v], pairs.get(v, []),
                          depths, Ks, Es, args.pix_thr, args.depth_thr)
        geo = (votes >= args.votes_req) & valid
        coherent = photo & geo

        dvis = np.where(valid, depth, np.nan)
        vlo, vhi = np.nanpercentile(dvis, [2, 98])
        rgb = np.asarray(Image.open(dd / "images" / f"{v:08d}.jpg"))
        coh_depth = np.where(coherent, depth, np.nan)

        panels = [
            (rgb, None, None, "gray"),
            (dvis, vlo, vhi, "turbo"),
            (conf, 0, 1, "viridis"),
            (photo, 0, 1, "gray"),
            (votes, 0, args.n_src, "turbo"),
            (coh_depth, vlo, vhi, "turbo"),
        ]
        for c, (img, lo, hi, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.set_facecolor(BG)
            if img.ndim == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], color="white", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"view {v}", color="white", fontsize=9)
        kept = 100.0 * coherent.sum() / max(valid.sum(), 1)
        axes[r][nc - 1].set_xlabel(f"{kept:.1f}% of valid kept",
                                   color="white", fontsize=8)

    fig.suptitle(
        f"MVSFormer++ init filters · {dd.parent.parent.name}  "
        f"conf>{args.conf_thr:g}, votes>={args.votes_req}/{args.n_src} "
        f"(pix<{args.pix_thr:g}px, reldepth<{args.depth_thr:g})",
        color="white", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (dd.parent.parent / "init_filter_coherent_pixels.png")
    fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
    print(f"[viz] -> {out}")


if __name__ == "__main__":
    main()
