"""Top-down preview of the FULL extent of a COLMAP point cloud, with the
barn-building bbox highlighted so we can see what's inside vs outside."""
from __future__ import annotations
import argparse
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out",   required=True)
ap.add_argument("--n",     type=int, default=300_000)
args = ap.parse_args()

pcd = o3d.io.read_point_cloud(args.input)
pts = np.asarray(pcd.points)
if len(pcd.colors) == len(pcd.points):
    cols = np.clip(np.asarray(pcd.colors), 0, 1)
else:
    cols = None
print(f"loaded {len(pts):,} pts  bbox {pts.min(0)} .. {pts.max(0)}", flush=True)

if len(pts) > args.n:
    idx = np.random.default_rng(0).choice(len(pts), args.n, replace=False)
    pts = pts[idx]
    if cols is not None: cols = cols[idx]

bmn, bmx = pts.min(0), pts.max(0)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Top-down (XZ plane, colour by height Y so floor/roof/trees become visible)
ax = axes[0]
s = ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], s=0.6, cmap="viridis_r",
               linewidths=0)
ax.set_xlim(bmn[0] - 0.3, bmx[0] + 0.3)
ax.set_ylim(bmn[2] - 0.3, bmx[2] + 0.3)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("top-down  (coloured by height Y, dark = low / ground)", fontsize=10)
ax.set_xlabel("X");  ax.set_ylabel("Z")
fig.colorbar(s, ax=ax, fraction=0.04, label="height (Y, COLMAP units, Y-down)")

# Top-down with RGB colour (true look)
ax = axes[1]
ax.scatter(pts[:, 0], pts[:, 2], c=(cols if cols is not None else "gray"),
           s=0.6, linewidths=0)
ax.set_xlim(bmn[0] - 0.3, bmx[0] + 0.3)
ax.set_ylim(bmn[2] - 0.3, bmx[2] + 0.3)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("top-down  (RGB)", fontsize=10)
ax.set_xlabel("X");  ax.set_ylabel("Z")

# Side view (XY): roof vs ground
ax = axes[2]
ax.scatter(pts[:, 0], pts[:, 1], c=(cols if cols is not None else "gray"),
           s=0.6, linewidths=0)
ax.set_xlim(bmn[0] - 0.3, bmx[0] + 0.3)
ax.set_ylim(bmn[1] - 0.3, bmx[1] + 0.3)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("side (XY)  Y-down → trees/roof at top, ground at bottom", fontsize=10)
ax.set_xlabel("X");  ax.set_ylabel("Y")

fig.suptitle(f"{args.input}   total bbox = [{bmn[0]:.2f},{bmx[0]:.2f}] x "
             f"[{bmn[1]:.2f},{bmx[1]:.2f}] x [{bmn[2]:.2f},{bmx[2]:.2f}]",
             fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=140, bbox_inches="tight")
print(f"wrote {args.out}", flush=True)
