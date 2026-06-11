"""Neuralangelo-Fig-5-style Phong render of a mesh, via Open3D's CPU
RaycastingScene (Embree). No GPU/EGL needed — runs on the login node.

Picks a few viewpoints around the mesh bbox, ray-casts at 800x600, shades
with one directional light + ambient. Result: a 2x2 grid PNG that looks
like the gray normal/Phong views in the Neuralangelo paper."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--input",   required=True)
ap.add_argument("--out",     required=True)
ap.add_argument("--res",     type=int, default=800)
ap.add_argument("--aspect",  type=float, default=4/3)
ap.add_argument("--fov-deg", type=float, default=45.0)
ap.add_argument("--mode",    choices=["phong", "normals", "both"], default="both",
                help="phong=gray shaded, normals=RGB normal map, both=side by side")
args = ap.parse_args()

H = args.res
W = int(round(H * args.aspect))

# --- load mesh -----------------------------------------------------------
mesh = o3d.io.read_triangle_mesh(args.input)
if len(mesh.triangles) == 0:
    raise SystemExit(f"{args.input} has no triangles (point cloud, not a mesh)")
mesh.compute_vertex_normals()
verts = np.asarray(mesh.vertices)
bbox_min = verts.min(0); bbox_max = verts.max(0)
center   = 0.5 * (bbox_min + bbox_max)
extent   = (bbox_max - bbox_min).max()
print(f"[load] {args.input}  verts={len(verts):,}  tris={len(mesh.triangles):,}",
      flush=True)
print(f"       bbox center={center}  extent={extent:.3f}", flush=True)

# --- build raycasting scene ---------------------------------------------
scene = o3d.t.geometry.RaycastingScene()
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene.add_triangles(mesh_t)

# --- view definitions ----------------------------------------------------
# 4 viewpoints orbiting the center, all looking inward, slightly above
# COLMAP uses Y-down world convention: world up = -Y axis.
WORLD_UP = np.array([0.0, -1.0, 0.0])
elev = -0.25      # above the barn (in -Y direction)
radius = extent * 1.4
views = []
for name, azim_deg in [
    ("front",  0.0),
    ("right", 90.0),
    ("back",  180.0),
    ("left",  270.0),
    ("oblique", 35.0),
]:
    a = np.deg2rad(azim_deg)
    cam = center + radius * np.array([np.cos(a), 0.0, np.sin(a)]) \
                 + np.array([0.0, elev * extent, 0.0])
    views.append((name, cam))
views.append(("top-down",
              center + np.array([0.0, -extent * 1.4, 0.0])))

# --- ray casting + Phong shading ----------------------------------------
def render(cam_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (phong_img, normal_img). Each is HxWx3 in [0,1]."""
    look = (center - cam_pos)
    look = look / np.linalg.norm(look)
    # if camera is nearly above the centre, WORLD_UP and look are parallel —
    # use world +Z as the reference up so the top-down view orients properly.
    ref_up = WORLD_UP if abs(np.dot(look, WORLD_UP)) < 0.95 else np.array([0., 0., 1.])
    right = np.cross(look, ref_up); right = right / np.linalg.norm(right)
    up = np.cross(right, look)

    f = 0.5 * H / np.tan(np.deg2rad(args.fov_deg) / 2.0)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dx = (xs - W / 2 + 0.5) / f
    dy = (ys - H / 2 + 0.5) / f   # y down
    d  = (dx[..., None] * right + dy[..., None] * up + look[None, None, :])
    d  = d / np.linalg.norm(d, axis=-1, keepdims=True)

    o = np.broadcast_to(cam_pos, d.shape).astype(np.float32)
    rays = np.concatenate([o, d.astype(np.float32)], axis=-1).reshape(-1, 6)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans  = scene.cast_rays(rays)

    t_hit  = ans["t_hit"].numpy().reshape(H, W)
    normal = ans["primitive_normals"].numpy().reshape(H, W, 3)
    hit    = np.isfinite(t_hit)

    # face the camera (front-side)
    n = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    flip = (n * d).sum(-1) > 0
    n[flip] = -n[flip]

    # Phong: warm light from upper-left + faint fill from below-right
    L = np.array([-0.3, -0.85, 0.45]); L = L / np.linalg.norm(L)
    diff = np.clip((n * L).sum(-1), 0.0, 1.0)
    phong = (0.30 + 0.70 * diff)[..., None] * np.array([0.78, 0.78, 0.82])
    phong[~hit] = 1.0

    # Normal map: classic Neuralangelo-Fig-1 right-half style (n → (n+1)/2)
    # Express normals in camera frame so the colouring matches the view.
    R = np.stack([right, up, -look], axis=0)   # world -> camera
    n_cam = n @ R.T
    nmap = (0.5 * (n_cam + 1.0)).clip(0, 1)
    nmap[~hit] = 1.0

    return np.clip(phong, 0, 1), nmap

n_rows = 1 if args.mode != "both" else 2
n_cols = len(views)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows))
if n_rows == 1:
    axes = np.array(axes).reshape(1, -1)

for col, (name, cam) in enumerate(views):
    phong, nmap = render(cam)
    if args.mode == "phong":
        axes[0, col].imshow(phong);  axes[0, col].set_title(name, fontsize=10)
    elif args.mode == "normals":
        axes[0, col].imshow(nmap);   axes[0, col].set_title(name, fontsize=10)
    else:
        axes[0, col].imshow(phong);  axes[0, col].set_title(name, fontsize=10)
        axes[1, col].imshow(nmap)
    for row in range(n_rows):
        axes[row, col].axis("off")
if args.mode == "both":
    axes[0, 0].set_ylabel("Phong",   rotation=90, fontsize=10, labelpad=8)
    axes[1, 0].set_ylabel("Normals", rotation=90, fontsize=10, labelpad=8)

fig.suptitle(args.input, fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=140, bbox_inches="tight")
print(f"[write] {args.out}", flush=True)
