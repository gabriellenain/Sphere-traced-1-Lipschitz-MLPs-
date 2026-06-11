"""Quick PNG preview of a COLMAP point cloud / mesh — 2x2 matplotlib grid of
top / front / side / oblique views, point cloud sub-sampled to keep it fast.
No GPU needed; runs in seconds."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out",   required=True)
ap.add_argument("--n",     type=int, default=200_000, help="max points")
args = ap.parse_args()

# read as point cloud OR sample from mesh
mesh = o3d.io.read_triangle_mesh(args.input)
if len(mesh.triangles) == 0:
    pcd = o3d.io.read_point_cloud(args.input)
else:
    pcd = mesh.sample_points_uniformly(number_of_points=args.n)

pts = np.asarray(pcd.points)
if len(pcd.colors) == len(pcd.points):
    cols = np.clip(np.asarray(pcd.colors), 0.0, 1.0)
else:
    cols = None

if len(pts) > args.n:
    idx = np.random.default_rng(0).choice(len(pts), args.n, replace=False)
    pts = pts[idx]
    if cols is not None: cols = cols[idx]

print(f"[load] {args.input}  pts={len(pts):,}  bbox={pts.min(0)} .. {pts.max(0)}",
      flush=True)
center = pts.mean(0)
ext    = (pts.max(0) - pts.min(0)).max() * 0.55

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
views = [
    ("top  (xz)",  (0, 2), 1, "Y"),
    ("front (xy)", (0, 1), 2, "Z"),
    ("side  (zy)", (2, 1), 0, "X"),
]
for ax, (title, (a, b), depth_axis, depth_label) in zip(axes.flat[:3], views):
    s = ax.scatter(pts[:, a], pts[:, b], c=(cols if cols is not None else pts[:, depth_axis]),
                   s=0.4, marker=".", linewidths=0,
                   cmap=("viridis" if cols is None else None))
    ax.set_aspect("equal")
    ax.set_xlim(center[a] - ext, center[a] + ext)
    ax.set_ylim(center[b] - ext, center[b] + ext)
    ax.set_title(title, fontsize=10)
    if cols is None:
        fig.colorbar(s, ax=ax, fraction=0.04, label=depth_label)

# oblique 3D
ax3 = axes[1, 1]; ax3.remove()
ax3 = fig.add_subplot(2, 2, 4, projection="3d")
ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
            c=(cols if cols is not None else pts[:, 2]),
            s=0.4, marker=".", linewidths=0,
            cmap=("viridis" if cols is None else None))
ax3.set_box_aspect((1, 1, 1))
ax3.set_xlim(center[0] - ext, center[0] + ext)
ax3.set_ylim(center[1] - ext, center[1] + ext)
ax3.set_zlim(center[2] - ext, center[2] + ext)
ax3.set_title("oblique", fontsize=10)
ax3.view_init(elev=25, azim=45)

fig.suptitle(f"{args.input}   {len(pts):,} pts", fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=120, bbox_inches="tight")
print(f"[write] {args.out}", flush=True)
