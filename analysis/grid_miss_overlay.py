#!/usr/bin/env python3
"""Grid of the miss-overlay panel (panel 0 of plot_miss_convergence_budget) across views.

For each view: trace the foreground rays with the run's REAL tracer at the run's
training budget (bracketing on, iters from config — i.e. exactly how the scene was
trained), take the misses, retrace them to --max-iters and colour each by fate:
  green  recovered with more iters
  amber  stalled — a root exists (tracer-fixable)
  red    no root — escapes to t_far (genuine tangent miss)
Only the image+overlay panel is drawn, tiled into one grid figure.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import lip_tracer.data as data_mod
from plot_miss_convergence_budget import (
    C_NOROOT, C_RECOV, C_ROOT, _dilate, bracket_hits, load_run,
    rays_for_view, trace_until,
)


def overlay_for_view(f, cfg, views, vi, short_iters, max_iters, chunk, device,
                     bracket, crop_fg):
    """Return (img, cls_map, counts) for one view — same logic as panel 0."""
    origins, dirs, img, mask, H, W = rays_for_view(views, vi, device)
    fg = mask.reshape(-1)

    if bracket:
        cfg_short = replace(cfg, iters=short_iters)
        hit_short = bracket_hits(f, origins, dirs, cfg_short, chunk)
        miss_idx = np.where(fg & ~hit_short)[0]
    else:
        short = trace_until(f, origins, dirs, cfg, short_iters, chunk)
        miss_idx = np.where(fg & (short["conv_iter"] < 0))[0]

    if len(miss_idx) == 0:
        cls_map = np.full(H * W, -1, dtype=np.int64).reshape(H, W)
        counts = (0, 0, 0, int(fg.sum()))
        return _maybe_crop(img, cls_map, mask, H, W, crop_fg) + (counts,)

    midx = torch.from_numpy(miss_idx).to(device)
    long = trace_until(f, origins[midx], dirs[midx], cfg, max_iters, chunk)
    still = long["conv_iter"] < 0
    has_root = still & long["ever_neg"]
    no_root = still & ~long["ever_neg"]

    miss_cls = np.full(len(miss_idx), 2, dtype=np.int64)  # default no-root
    miss_cls[has_root] = 1
    miss_cls[~still] = 0                                   # recovered
    cls_map = np.full(H * W, -1, dtype=np.int64)
    cls_map[miss_idx] = miss_cls
    cls_map = cls_map.reshape(H, W)

    counts = (int((~still).sum()), int(has_root.sum()), int(no_root.sum()),
              int(fg.sum()))
    return _maybe_crop(img, cls_map, mask, H, W, crop_fg) + (counts,)


def _maybe_crop(img, cls_map, mask, H, W, crop_fg):
    if not crop_fg:
        return img, cls_map
    ys, xs = np.where(mask)
    if not ys.size:
        return img, cls_map
    pad = 0.12
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    dr, dc = int((r1 - r0) * pad) + 1, int((c1 - c0) * pad) + 1
    r0, r1 = max(r0 - dr, 0), min(r1 + dr, H - 1) + 1
    c0, c1 = max(c0 - dc, 0), min(c1 + dc, W - 1) + 1
    return img[r0:r1, c0:c1], cls_map[r0:r1, c0:c1]


def draw_cell(ax, img, cls_map, title):
    ax.imshow(img * 0.45 + 0.04)
    overlay = np.zeros((*cls_map.shape, 4), dtype=np.float32)
    for cid, hexc in [(0, C_RECOV), (1, C_ROOT), (2, C_NOROOT)]:
        rgb = matplotlib.colors.to_rgb(hexc)
        overlay[_dilate(cls_map == cid)] = (*rgb, 0.95)
    ax.imshow(overlay)
    ax.set_title(title, fontsize=7.2)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--n-views", type=int, default=16)
    ap.add_argument("--views", default="",
                    help="comma-separated view indices; overrides --n-views")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--short-iters", type=int, default=-1,
                    help="default: run's train budget (trace.iters from config)")
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--no-bracket", action="store_true",
                    help="use a bare march instead of the run's real tracer")
    ap.add_argument("--crop-fg", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("figs/sphere_trace_steps"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    f, cfg, scene, ckpt_path = load_run(args.run_dir, args.ckpt, args.device)
    data_mod.SCENE = scene
    views = data_mod.load_views(scene, down=args.down)
    nv = int(views["images"].shape[0])
    short_iters = cfg.iters if args.short_iters < 0 else args.short_iters
    bracket = not args.no_bracket

    if args.views.strip():
        vlist = [int(v) % nv for v in args.views.split(",")]
    else:
        vlist = [int(round(x)) % nv for x in np.linspace(0, nv - 1, args.n_views)]
    print(f"run={args.run_dir.name} views={vlist} down={args.down} "
          f"short-iters={short_iters} max-iters={args.max_iters} "
          f"bracket={bracket}", flush=True)

    cols = args.cols
    rows = (len(vlist) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.7))
    axes = np.atleast_1d(axes).ravel()

    for ax, vi in zip(axes, vlist):
        img, cls_map, (rec, root, nor, fgn) = overlay_for_view(
            f, cfg, views, vi, short_iters, args.max_iters, args.chunk,
            args.device, bracket, args.crop_fg)
        draw_cell(ax, img, cls_map,
                  f"view {vi}  miss {rec + root + nor}/{fgn}\n"
                  f"recovered {rec} · stalled {root} · tangent {nor}")
        print(f"  view {vi:>2}: recovered={rec} stalled={root} "
              f"no_root={nor}  (fg={fgn})", flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()
    for ax in axes[len(vlist):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=9, color=C_RECOV,
                   label="recovered with more iters"),
        plt.Line2D([], [], marker="s", ls="", ms=9, color=C_ROOT,
                   label="stalled — root exists (tracer-fixable)"),
        plt.Line2D([], [], marker="s", ls="", ms=9, color=C_NOROOT,
                   label=r"no root — escapes to $t_{far}$ (tangent)"),
    ]
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"{args.run_dir.name}  ·  {ckpt_path.name}  ·  "
        f"bracketed run tracer, iters={short_iters}→{args.max_iters}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if bracket else "_nobracket"
    stem = (f"{args.run_dir.name}_grid_miss_overlay_down{args.down}"
            f"_to{args.max_iters}{tag}")
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
