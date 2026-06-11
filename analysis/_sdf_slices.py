"""Visualise the SDF derived from a triangle mesh, as orthogonal slices.

Uses Open3D's RaycastingScene.compute_signed_distance — same query you'd
use in an SDF-regression init. Lets you sanity-check the inside/outside
structure before committing to a fit."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out",   required=True)
ap.add_argument("--res",   type=int,  default=320, help="grid res per axis")
ap.add_argument("--pad",   type=float, default=0.15,
                help="extra padding around the mesh bbox, fraction of extent")
args = ap.parse_args()

mesh = o3d.io.read_triangle_mesh(args.input)
if len(mesh.triangles) == 0:
    raise SystemExit(f"{args.input} has no triangles")
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

verts = np.asarray(mesh.vertices)
bmn = verts.min(0); bmx = verts.max(0)
ext = (bmx - bmn).max()
pad = args.pad * ext
bmn = bmn - pad; bmx = bmx + pad
ctr = 0.5 * (bmn + bmx)
print(f"[bbox] {bmn} .. {bmx}", flush=True)

def sdf_slice(axis: int, level: float) -> np.ndarray:
    """SDF on a 2D grid orthogonal to `axis` at world coordinate `level`."""
    others = [i for i in range(3) if i != axis]
    a, b = others  # the two in-plane axes
    g0 = np.linspace(bmn[a], bmx[a], args.res, dtype=np.float32)
    g1 = np.linspace(bmn[b], bmx[b], args.res, dtype=np.float32)
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    pts = np.empty((args.res, args.res, 3), dtype=np.float32)
    pts[..., axis] = level
    pts[..., a]    = G0
    pts[..., b]    = G1
    flat = o3d.core.Tensor(pts.reshape(-1, 3), dtype=o3d.core.Dtype.Float32)
    sd = scene.compute_signed_distance(flat).numpy().reshape(args.res, args.res)
    return g0, g1, sd

# COLMAP frame: Y-down. y_min = top of barn, y_max = ground.
y_top, y_mid, y_floor = bmn[1] + 0.30 * (bmx[1] - bmn[1]), \
                        bmn[1] + 0.60 * (bmx[1] - bmn[1]), \
                        bmn[1] + 0.85 * (bmx[1] - bmn[1])

panels = [
    ("Y slice — high (near roof)",   1, y_top,   "xz"),
    ("Y slice — mid",                1, y_mid,   "xz"),
    ("Y slice — low (floor)",        1, y_floor, "xz"),
    ("Z slice — through centre",     2, ctr[2],  "xy"),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 14))
vmax = ext * 0.5  # ~ 1.5 m at this scale

for ax, (title, axis, level, plane_lbl) in zip(axes.flat, panels):
    g0, g1, sd = sdf_slice(axis, level)
    im = ax.pcolormesh(g0, g1, sd.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.contour(g0, g1, sd.T, levels=[0.0], colors="k", linewidths=0.7)
    ax.set_aspect("equal")
    if plane_lbl == "xy":
        ax.invert_yaxis()    # Y-down → put +Y at the bottom
    ax.set_title(title + f"   (axis={'xyz'[axis]}={level:+.2f})", fontsize=10)
    ax.set_xlabel("xyz"[axis == 0])  # not strictly correct but readable
    fig.colorbar(im, ax=ax, fraction=0.04,
                 label="signed distance  (red=inside, blue=outside)")

fig.suptitle(f"SDF from {args.input}   (RaycastingScene.compute_signed_distance,  "
             f"res={args.res})", fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=120, bbox_inches="tight")
print(f"[write] {args.out}", flush=True)
