"""Overlay COLMAP sfm points (filtered by --sfm-min-views) on scan24 views."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-overlay")
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views, load_colmap_points, colmap_visibility_counts

SCENE = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24")
MIN_VIEWS = 30
VIEW_IDS = [0, 24, 48]
OUT = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/sfm_min30_overlay_scan24.png")

views = load_views(SCENE)
pts = load_colmap_points(SCENE)                      # (P, 3)
counts = colmap_visibility_counts(pts, views)        # (P,)
keep = counts >= MIN_VIEWS
print(f"COLMAP points: {len(pts)} total, {int(keep.sum())} kept (>= {MIN_VIEWS} masked views)")
print(f"visibility counts: min={int(counts.min())} median={int(counts.median())} max={int(counts.max())}")

w2c = torch.linalg.inv(views["c2w"])
R, t, K = w2c[:, :3, :3], w2c[:, :3, 3], views["K"]
H, W = views["H"], views["W"]
imgs = views["images"]

def project(v, p):
    xc = p @ R[v].T + t[v]
    uvh = xc @ K[v].T
    z = xc[:, 2]
    uv = uvh[:, :2] / uvh[:, 2:3].clamp_min(1e-6)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 1e-4)
    return uv, inb

fig, axes = plt.subplots(1, len(VIEW_IDS), figsize=(6 * len(VIEW_IDS), 8))
if len(VIEW_IDS) == 1:
    axes = [axes]
for ax, v in zip(axes, VIEW_IDS):
    ax.imshow(imgs[v].numpy())
    uv_d, inb_d = project(v, pts[~keep])
    uv_k, inb_k = project(v, pts[keep])
    ax.scatter(uv_d[inb_d, 0], uv_d[inb_d, 1], s=4, c="red", alpha=0.35,
               label=f"discarded (<{MIN_VIEWS}): {int(inb_d.sum())}")
    ax.scatter(uv_k[inb_k, 0], uv_k[inb_k, 1], s=6, c="lime", alpha=0.9,
               label=f">= {MIN_VIEWS} views: {int(inb_k.sum())}")
    ax.set_title(f"view {v}")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    ax.axis("off")

fig.suptitle(f"scan24 COLMAP sfm points — min-views {MIN_VIEWS} filter "
             f"({int(keep.sum())}/{len(pts)} kept)", fontsize=13)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"saved → {OUT}")
