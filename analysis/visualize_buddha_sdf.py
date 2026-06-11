"""Visualise the GT SDF of the normalised Happy Buddha at several Z slices.

Saves a PNG grid: each row = one Z level, columns = [SDF heatmap | zero-crossing contour].
Run this before fitting to verify the SDF computation captures fine surface detail.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

PLY_IN  = Path("artifacts/happy_recon.ply")
PLY_OUT = Path("data/gt_meshes/happy_buddha_norm.ply")
OUT_PNG = Path("artifacts/buddha_sdf_slices.png")
RES     = 512          # pixels per slice — high enough to see robe creases
BOUND   = 1.1
Z_LEVELS = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.85]   # normalised Z from base to head

# ── 1. Load & normalise ──────────────────────────────────────────────────────
print("Loading mesh …")
raw   = trimesh.load(str(PLY_IN), process=False)
verts = np.asarray(raw.vertices, np.float32)
verts -= verts.mean(0)
verts /= np.abs(verts).max()
verts = verts[:, [0, 2, 1]]   # Stanford Y-up → Z-up
mesh  = trimesh.Trimesh(vertices=verts, faces=raw.faces, process=True)
print(f"  verts={len(mesh.vertices):,}  faces={len(mesh.faces):,}  watertight={mesh.is_watertight}")

PLY_OUT.parent.mkdir(parents=True, exist_ok=True)
mesh.export(str(PLY_OUT))
print(f"  saved normalised PLY → {PLY_OUT}")

# ── 2. Build face kD-tree once ───────────────────────────────────────────────
print("Building face kD-tree …")
face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
face_tree      = cKDTree(face_centroids)
face_normals   = mesh.face_normals
triangles      = mesh.vertices[mesh.faces]

def sdf_slice(z: float) -> np.ndarray:
    xs  = np.linspace(-BOUND, BOUND, RES)
    ys  = np.linspace(-BOUND, BOUND, RES)
    xx, yy = np.meshgrid(xs, ys[::-1])
    pts = np.stack([xx.ravel(), yy.ravel(), np.full(RES * RES, z)], 1).astype(np.float32)
    _, fidx   = face_tree.query(pts, workers=-1)
    closest   = trimesh.triangles.closest_point(triangles[fidx], pts)
    offset    = pts - closest
    dist      = np.linalg.norm(offset, axis=-1)
    sign      = np.sign((offset * face_normals[fidx]).sum(axis=-1))
    sign[sign == 0] = 1.0
    return (sign * dist).reshape(RES, RES)

# ── 3. Render slices ─────────────────────────────────────────────────────────
n = len(Z_LEVELS)
fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
kw_heat = dict(cmap="RdBu_r", vmin=-0.3, vmax=0.3,
               extent=[-BOUND, BOUND, -BOUND, BOUND], origin="upper")

for row, z in enumerate(Z_LEVELS):
    print(f"  computing SDF slice z={z:+.2f} …", flush=True)
    sdf = sdf_slice(z)
    xs  = np.linspace(-BOUND, BOUND, RES)

    ax_h = axes[row, 0]
    im   = ax_h.imshow(sdf, **kw_heat)
    ax_h.contour(xs, xs[::-1], sdf, levels=[0.0], colors="k", linewidths=1.2)
    ax_h.set_title(f"GT SDF  z={z:+.2f}  (red=outside, blue=inside)")
    ax_h.set_xlabel("x"); ax_h.set_ylabel("y")
    plt.colorbar(im, ax=ax_h, fraction=0.046)

    # Contour-only panel — shows fine zero-crossing detail more clearly
    ax_c = axes[row, 1]
    ax_c.set_facecolor("white")
    # filled distance bands for readability
    ax_c.imshow(np.abs(sdf) < 0.05, cmap="Greys_r", vmin=0, vmax=1,
                extent=[-BOUND, BOUND, -BOUND, BOUND], origin="upper", alpha=0.35)
    ax_c.contour(xs, xs[::-1], sdf,
                 levels=[-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15],
                 colors=["#3a6bc4","#5a8be4","#a0c0ff","k","#ffb0a0","#e45a5a","#c43a3a"],
                 linewidths=[0.8, 0.8, 0.8, 1.5, 0.8, 0.8, 0.8])
    ax_c.set_title(f"Contours  z={z:+.2f}  (black = surface)")
    ax_c.set_xlabel("x"); ax_c.set_ylabel("y")
    ax_c.set_xlim(-BOUND, BOUND); ax_c.set_ylim(-BOUND, BOUND)

plt.suptitle("Happy Buddha — GT SDF cross-sections (normalised coords, Z-up)", fontsize=13)
plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
