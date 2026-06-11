"""Overlay the border-aware hull onto real scan24 photos from the SAME cameras.

Ray-marches the hull's distance-transform SDF through each chosen DTU camera
(exact K + pose), shades it, and composites it over that camera's RGB so the hull
silhouette can be compared against the real object. Left = photo, right = overlay.
"""
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt
from lip_tracer.visual_hull import carve
from lip_tracer.data import load_views, load_colmap_points

SCENE = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan24")
OUT = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_fix")
RES, BOUND = 256, 1.5
DOWN, STEPS, EPS = 4, 96, 0.004

# ---- matched hull SDF ----
sfm = load_colmap_points(SCENE).numpy()
lo, hi = sfm.min(0), sfm.max(0); pad = np.maximum(0.15, 0.15 * (hi - lo))
roi = (np.maximum(lo - pad, -BOUND), np.minimum(hi + pad, BOUND))
occ = carve(SCENE, RES, BOUND, roi_bounds=roi, border_aware=True)
vox = 2 * BOUND / (RES - 1)
sdf_np = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * vox  # (z,y,x)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV}")
sdf = torch.from_numpy(sdf_np.astype(np.float32))[None, None].to(DEV)        # (1,1,D=z,H=y,W=x)

def sdf_at(p):                       # p: (N,3) world xyz
    g = (p / BOUND).view(1, 1, 1, -1, 3)          # last dim order (x,y,z)=(W,H,D)
    return F.grid_sample(sdf, g, align_corners=True, padding_mode="border").view(-1)

def ray_box(o, d):                   # entry/exit t for [-BOUND,BOUND]^3
    inv = 1.0 / d
    t0 = (-BOUND - o) * inv; t1 = (BOUND - o) * inv
    tmin = torch.minimum(t0, t1).amax(1); tmax = torch.maximum(t0, t1).amin(1)
    return tmin, tmax

views = load_views(SCENE)
V = views["c2w"].shape[0]
idx = np.linspace(0, V - 1, 4, dtype=int)
H, W = views["H"], views["W"]
Hd, Wd = H // DOWN, W // DOWN
light = torch.tensor([0.4, 0.5, 0.8], device=DEV); light /= light.norm()

rows = []
for vi in idx:
    c2w = views["c2w"][vi].to(DEV); K = views["K"][vi].clone().to(DEV)
    K[0] /= DOWN; K[1] /= DOWN
    ys, xs = torch.meshgrid(torch.arange(Hd, device=DEV), torch.arange(Wd, device=DEV), indexing="ij")
    dcam = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1],
                        torch.ones_like(xs)], -1).reshape(-1, 3).float()
    d = dcam @ c2w[:3, :3].T; d = d / d.norm(dim=1, keepdim=True)
    o = c2w[:3, 3].expand_as(d).contiguous()
    tmin, tmax = ray_box(o, d)
    alive = tmin < tmax
    t = torch.where(alive, tmin + 1e-3, torch.full_like(tmin, 1e9))
    hit = torch.zeros(len(d), dtype=torch.bool, device=DEV)
    for _ in range(STEPS):
        run = alive & ~hit & (t < tmax)
        if not run.any(): break
        s = sdf_at(o[run] + t[run, None] * d[run])
        newhit = s < EPS
        ridx = run.nonzero(as_tuple=True)[0]
        hit[ridx[newhit]] = True
        t[ridx] = t[ridx] + torch.clamp(s, min=vox)
    # normals at hit
    ph = o + t[:, None] * d
    shade = torch.zeros(len(d), device=DEV)
    if hit.any():
        e = vox
        p = ph[hit]
        def s_(off): return sdf_at(p + torch.tensor(off, dtype=torch.float32, device=DEV))
        nx = s_([e, 0, 0]) - s_([-e, 0, 0])
        ny = s_([0, e, 0]) - s_([0, -e, 0])
        nz = s_([0, 0, e]) - s_([0, 0, -e])
        n = torch.stack([nx, ny, nz], 1); n = n / (n.norm(dim=1, keepdim=True) + 1e-9)
        shade[hit] = (0.3 + 0.7 * (n @ light).abs()).clamp(0, 1)
    hitm = hit.view(Hd, Wd).cpu().numpy()
    sh = shade.view(Hd, Wd).cpu().numpy()

    photo = views["images"][vi].numpy()
    photo = np.array(Image.fromarray((photo * 255).astype(np.uint8)).resize((Wd, Hd), Image.BILINEAR)) / 255.0
    overlay = photo.copy()
    a = 0.55
    hull_rgb = np.stack([sh * 1.0, sh * 0.55, sh * 0.1], -1)   # orange-ish shaded hull
    overlay[hitm] = (1 - a) * photo[hitm] + a * hull_rgb[hitm]
    # green = hull silhouette boundary, blue = mask boundary used for carving
    from scipy.ndimage import binary_erosion
    edge = hitm & ~binary_erosion(hitm, iterations=2)
    mask = views["masks"][vi].numpy()
    mask = np.array(Image.fromarray(mask).resize((Wd, Hd), Image.NEAREST)).astype(bool)
    medge = mask & ~binary_erosion(mask, iterations=2)
    overlay[medge] = [0.1, 0.5, 1.0]    # mask outline (blue)
    overlay[edge] = [0.1, 1.0, 0.2]     # hull outline (green)
    rows.append(np.concatenate([photo, overlay], axis=1))

grid = np.concatenate(rows, axis=0)
p = OUT / "hull_scan24_photo_overlay.png"
Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8)).save(p)
print(f"views={idx.tolist()}  saved → {p}")
