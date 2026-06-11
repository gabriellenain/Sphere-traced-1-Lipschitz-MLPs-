"""Render the Happy Buddha PLY directly with the paper shading pipeline.

Loads artifacts/happy_recon.ply, normalises it, optionally smooths vertex
normals, and renders with hemisphere + AO shading. No voxelisation or
marching cubes — full original mesh detail is preserved.

Outputs:
  artifacts/happy_buddha_mc.png        — 4-view azimuth strip (0/90/180/270°)
  artifacts/happy_buddha_<az>deg.png   — individual views
"""
import io, sys, tarfile, time, urllib.request
t_start = time.time()
def _elapsed(): return time.time() - t_start
from pathlib import Path

import numpy as np
import torch
import trimesh
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).parent))
from render_paper import shade
from render_paper_marching import _make_intersector, _mesh_hits, _mesh_ao

OUT_DIR  = Path("artifacts")
PLY_PATH = OUT_DIR / "happy_recon.ply"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Download (if not cached) ──────────────────────────────────────────────
URL = "http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz"
if not PLY_PATH.exists():
    print("Downloading happy_recon.tar.gz …")
    data = urllib.request.urlopen(URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".ply"):
                PLY_PATH.write_bytes(tf.extractfile(m).read())
                print(f"  extracted → {PLY_PATH}")
                break
else:
    print("PLY cached.")

# ── 2. Load, normalise, light smooth ────────────────────────────────────────
print(f"[{_elapsed():6.1f}s] Loading mesh …", flush=True)
t0    = time.time()
raw   = trimesh.load(str(PLY_PATH), process=False)
verts = np.asarray(raw.vertices, np.float32)
verts -= verts.mean(0)
verts /= np.abs(verts).max()
verts = verts[:, [0, 2, 1]]   # Stanford Y-up → Z-up (matches shade()'s hemisphere)
mesh  = trimesh.Trimesh(vertices=verts, faces=raw.faces, process=True)
print(f"[{_elapsed():6.1f}s]   verts={len(mesh.vertices):,}  faces={len(mesh.faces):,}  ({time.time()-t0:.1f}s)", flush=True)


intersector = _make_intersector(mesh)

# ── 3. Camera helpers ────────────────────────────────────────────────────────
H, W  = 1600, 900         # high-res portrait
fov_y = 45.0
fy    = H / (2 * np.tan(np.radians(fov_y / 2)))
K     = np.array([[fy, 0, W/2], [0, fy, H/2], [0, 0, 1]], np.float64)

DIST  = 3.2               # camera distance from origin
ELEV  = 0.10              # slight upward tilt of the eye
up_g  = np.array([0.0, 0.0, 1.0])

def make_c2w(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    az = np.radians(azimuth_deg)
    eye    = np.array([DIST * np.sin(az), -DIST * np.cos(az), ELEV])
    target = np.array([0.0, 0.0, 0.0])
    forward = target - eye;  forward /= np.linalg.norm(forward)
    right   = np.cross(forward, up_g); right   /= np.linalg.norm(right)
    down    = np.cross(forward, right); down    /= np.linalg.norm(down)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w, eye

def render_view(c2w: np.ndarray, eye: np.ndarray, label: str) -> np.ndarray:
    print(f"[{_elapsed():6.1f}s] === {label} ===", flush=True)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d_cam  = np.stack([
        (xs + 0.5 - K[0, 2]) / K[0, 0],
        (ys + 0.5 - K[1, 2]) / K[1, 1],
        np.ones((H, W), np.float64),
    ], axis=-1)
    dirs      = d_cam @ c2w[:3, :3].T
    dirs     /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins   = np.broadcast_to(eye, dirs.shape).reshape(-1, 3).astype(np.float64)
    dirs_flat = dirs.reshape(-1, 3).astype(np.float64)

    t0 = time.time()
    hit, x_hit, normals, _ = _mesh_hits(mesh, intersector, origins, dirs_flat)
    print(f"[{_elapsed():6.1f}s]   hits {hit.sum():,}/{len(hit):,} ({hit.mean()*100:.1f}%)  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    ao = _mesh_ao(mesh, intersector, x_hit, normals, hit, rays=32, radius=0.08)
    print(f"[{_elapsed():6.1f}s]   AO done ({time.time()-t0:.1f}s)", flush=True)

    n_t    = torch.from_numpy(normals.astype(np.float32))
    d_t    = torch.from_numpy(dirs_flat.astype(np.float32))
    ao_t   = torch.from_numpy(ao.astype(np.float32)).unsqueeze(-1)
    shaded = shade(n_t, d_t, ao_t, "cpu").numpy()

    # Contrast boost: undo sRGB gamma → stretch levels → re-encode.
    # shade() clamps AO floor at 0.35 which leaves shadows pale;
    # pulling the black point up to 0.12 (linear) recovers crevice depth.
    lin = np.clip(shaded.astype(np.float64) ** 2.2, 0, 1)
    lin = np.clip((lin - 0.12) / (1.0 - 0.12), 0, 1)
    shaded = lin ** (1.0 / 2.2)

    alpha   = hit[:, None].astype(np.float64)
    img     = (shaded * alpha + 1.0 * (1.0 - alpha)).reshape(H, W, 3)
    nmap    = ((0.5 * (normals + 1.0)).clip(0, 1) * alpha + 0.5 * (1.0 - alpha)).reshape(H, W, 3)
    return img, nmap

# ── 4. Render 4 azimuths ────────────────────────────────────────────────────
AZIMUTHS = [0, 90, 180, 270]
views, nmaps = [], []
for az in AZIMUTHS:
    c2w, eye = make_c2w(az)
    img, nmap = render_view(c2w, eye, f"azimuth {az}°")
    u8 = lambda x: np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
    imageio.imwrite(str(OUT_DIR / f"happy_buddha_{az}deg.png"),        u8(img))
    imageio.imwrite(str(OUT_DIR / f"happy_buddha_{az}deg_normals.png"), u8(nmap))
    print(f"[{_elapsed():6.1f}s]   saved → {az}deg shaded + normals", flush=True)
    views.append(img)
    nmaps.append(nmap)

# ── 5. Save strips (shaded row + normal-map row) ─────────────────────────────
def u8(x): return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
shaded_strip = np.concatenate(views, axis=1)
normal_strip = np.concatenate(nmaps, axis=1)
grid = np.concatenate([shaded_strip, normal_strip], axis=0)  # shaded on top, normals below

imageio.imwrite(str(OUT_DIR / "happy_buddha_mc.png"),      u8(shaded_strip))
imageio.imwrite(str(OUT_DIR / "happy_buddha_normals.png"), u8(normal_strip))
imageio.imwrite(str(OUT_DIR / "happy_buddha_grid.png"),    u8(grid))
print(f"[{_elapsed():6.1f}s] Saved strips → happy_buddha_mc.png / _normals.png / _grid.png  (total {_elapsed():.1f}s)", flush=True)
print("Columns: 0° | 90° | 180° | 270°  —  top row: shaded, bottom row: normals", flush=True)
