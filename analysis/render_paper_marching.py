"""Paper-figure renderer using a marching-cubes mesh.

This mirrors analysis/render_paper.py's view selection and output layout, but renders an
extracted zero level set with mesh ray casting instead of sphere tracing. The
result is meant for clean static figures that visually match the geometry used
by Chamfer evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from lip_tracer.data import load_blender_views, load_views
from lip_tracer.model import make_model
from lip_tracer.visual_hull import carve
from render_paper import reproject_color, shade


def _resolve_trained_scene(ckpt_path: Path) -> Path | None:
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return None
    scene = cfg.get("scene")
    return Path(scene) if scene else None


def _infer_dataset(scene: Path) -> str:
    parts = str(scene).lower()
    if "nerf_synthetic" in parts or "/lego" in parts:
        return "lego"
    return "dtu"


def _checkpoint_model_kwargs(ckpt: dict) -> dict:
    state = ckpt["f"]
    enc = ckpt.get("input_encoding", "identity")
    if enc == "neus":
        enc = "pe"

    hidden = ckpt.get("hidden")
    if hidden is None:
        if "head_weight" in state:
            hidden = state["head_weight"].shape[0]
        elif "layers.0.weight" in state:
            hidden = state["layers.0.weight"].shape[0]
        else:
            hidden = 256

    depth = ckpt.get("depth")
    if depth is None:
        depth = sum(
            1 for key in state
            if key.startswith("net.") and key.endswith(".weight") and "_u" not in key
        ) or 8

    return {
        "hidden": hidden,
        "depth": depth,
        "group_size": ckpt.get("group_size", 2),
        "activation": ckpt.get("activation", "groupsort"),
        "input_encoding": enc,
        "multires": ckpt.get("multires", 6),
        "architecture": ckpt.get("architecture", "cpl"),
    }


def _level_tag(level: float) -> str:
    return f"{level:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _label_bottom(img: np.ndarray, text: str, band_frac: float = 0.09) -> np.ndarray:
    """Append a white strip with centred black text below a float[0,1] HxWx3 image."""
    from PIL import Image, ImageDraw

    H, W = img.shape[:2]
    band = max(20, int(round(H * band_frac)))
    canvas = np.ones((H + band, W, 3), dtype=img.dtype)
    canvas[:H] = img
    pil = Image.fromarray(np.clip(canvas * 255 + 0.5, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, band - 8))
    except Exception:
        font = None
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        oy = bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
        oy = 0
    draw.text(((W - tw) / 2.0, H + (band - th) / 2.0 - oy),
              text, fill=(0, 0, 0), font=font)
    return np.asarray(pil).astype(img.dtype) / 255.0


def _mesh_from_volume(vol: np.ndarray, bound: float, res: int, level: float) -> trimesh.Trimesh:
    import mcubes

    if vol.min() > level or vol.max() < level:
        raise ValueError(
            f"No {level:+.4f} crossing in SDF grid "
            f"(range=[{vol.min():.4f}, {vol.max():.4f}]). Try adjusting --bound."
        )

    spacing = 2 * bound / (res - 1)
    verts, faces = mcubes.marching_cubes(vol, level)
    verts = (verts * spacing - bound).astype(np.float32)
    # PyMCubes winds faces opposite to our SDF sign convention (outside > 0),
    # which would make trimesh vertex_normals point inward — opposite of the
    # SDF gradient used by sphere-trace and the analytic-normals path. Flip
    # winding so vertex normals and ∇f agree out-of-the-box.
    faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=False)
    print(
        f"[mesh] level {level:+.4f}: "
        f"{len(mesh.vertices):,} verts, {len(mesh.faces):,} faces",
        flush=True,
    )
    return mesh


def _eval_sdf_on(f, points: torch.Tensor, device: str, chunk: int,
                 tag: str = "eval") -> torch.Tensor:
    """Chunked SDF evaluation on a (N, 3) CPU tensor; returns (N,) CPU tensor."""
    n_total = len(points)
    n_chunks = max(1, (n_total + chunk - 1) // chunk)
    log_every = max(1, n_chunks // 10)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for ci, i in enumerate(range(0, n_total, chunk)):
            out.append(f(points[i:i + chunk].to(device)).detach().cpu())
            if (ci + 1) % log_every == 0 or ci + 1 == n_chunks:
                done = min(i + chunk, n_total)
                elapsed = time.time() - t0
                eta = elapsed * (n_total / done - 1) if done else 0.0
                print(
                    f"       [{tag}] [{done:>12,} / {n_total:,}] "
                    f"{100.0 * done / n_total:5.1f}%  "
                    f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s",
                    flush=True,
                )
    return torch.cat(out) if out else torch.zeros(0)


def extract_mesh(ckpt_path: Path, bound: float, res: int, device: str,
                 level: float = 0.0,
                 chunk: int | None = None,
                 keep_model: bool = False,
                 coarse_res: int = 128,
                 band_k: float = 2.5):
    """Narrow-band marching cubes.

    1. Dense coarse SDF eval at coarse_res^3.
    2. Trilinear-upsample to res^3 as a baseline far-from-surface field.
    3. Re-evaluate the true SDF only at fine cells with |sdf|<band, where
       band = band_k * coarse_cell_size. For a ~1-Lipschitz SDF this band
       provably contains every iso-level crossing.

    Cuts SDF evals by ~10-20x vs dense res^3 with identical mesh.
    """
    import torch.nn.functional as F

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    f = make_model(**_checkpoint_model_kwargs(ckpt)).to(device).eval()
    f.load_state_dict(ckpt["f"], strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))

    if chunk is None:
        chunk = 65536 if device == "cuda" else 65536

    print(
        f"[mesh] coarse pass {coarse_res}^3 = {coarse_res ** 3:,} points "
        f"(device={device}, chunk={chunk})",
        flush=True,
    )
    vox_c = torch.linspace(-bound, bound, coarse_res)
    grid_c = torch.stack(torch.meshgrid(vox_c, vox_c, vox_c, indexing="ij"), dim=-1).reshape(-1, 3)
    t0 = time.time()
    sdf_c = _eval_sdf_on(f, grid_c, device, chunk, tag="coarse").reshape(
        coarse_res, coarse_res, coarse_res
    )
    print(f"[mesh] coarse pass done in {time.time() - t0:.1f}s; "
          f"range=[{sdf_c.min().item():.4f}, {sdf_c.max().item():.4f}]",
          flush=True)
    del grid_c

    # Trilinear upsample as baseline for cells outside the band.
    vol = F.interpolate(
        sdf_c[None, None], size=(res, res, res),
        mode="trilinear", align_corners=True,
    ).squeeze(0).squeeze(0).numpy().astype(np.float32)
    del sdf_c

    coarse_cell = 2.0 * bound / (coarse_res - 1)
    band = float(band_k * coarse_cell)
    near_f = np.abs(vol) < band
    n_eval = int(near_f.sum())
    print(
        f"[mesh] narrow band |sdf|<{band:.4f} (band_k={band_k}): "
        f"{n_eval:,}/{near_f.size:,} fine cells "
        f"({100.0 * n_eval / near_f.size:.3f}%, "
        f"vs {res ** 3:,} for dense — {res ** 3 / max(n_eval, 1):.1f}x fewer)",
        flush=True,
    )

    # Fine pass: evaluate true SDF only on flagged cells.
    vox_f = torch.linspace(-bound, bound, res)
    ix, iy, iz = np.where(near_f)
    ix_t = torch.from_numpy(ix.astype(np.int64))
    iy_t = torch.from_numpy(iy.astype(np.int64))
    iz_t = torch.from_numpy(iz.astype(np.int64))
    pts_f = torch.stack([vox_f[ix_t], vox_f[iy_t], vox_f[iz_t]], dim=-1)
    t0 = time.time()
    sdf_f = _eval_sdf_on(f, pts_f, device, chunk, tag="fine").numpy().astype(np.float32)
    print(f"[mesh] fine pass done in {time.time() - t0:.1f}s", flush=True)
    vol[ix, iy, iz] = sdf_f
    del pts_f, sdf_f

    if not keep_model:
        del f
        f = None
    if device != "cpu":
        torch.cuda.empty_cache()
    print(f"[mesh] final vol range=[{vol.min():.4f}, {vol.max():.4f}]", flush=True)
    mesh = _mesh_from_volume(vol, bound, res, level)
    return mesh, f, vol


def extract_hull_mesh(scene: Path, res: int, bound: float) -> trimesh.Trimesh:
    """Carve visual hull from masks and extract its zero level set via marching cubes."""
    import mcubes

    print(f"[hull] carving visual hull  res={res}  bound={bound}", flush=True)
    t0 = time.time()
    occ = carve(scene=scene, res=res, bound=bound)
    print(f"[hull] occupied voxels: {occ.sum():,} / {occ.size:,}  ({time.time() - t0:.1f}s)", flush=True)

    spacing = 2 * bound / (res - 1)
    verts, faces = mcubes.marching_cubes(occ.astype(np.float64), 0.5)
    verts = (verts * spacing - bound).astype(np.float32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=False)
    print(f"[hull] extracted {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces", flush=True)
    return mesh


def _make_intersector(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        print("[mesh] using pyembree ray intersector", flush=True)
        return RayMeshIntersector(mesh)
    except Exception as exc:
        print(f"[mesh] using trimesh ray intersector ({exc})", flush=True)
        return mesh.ray


def _camera_rays(c2w: np.ndarray, K: np.ndarray, H: int, W: int, ss: int):
    Hs, Ws = H * ss, W * ss
    Ks = K.copy()
    Ks[0, 0] *= ss
    Ks[1, 1] *= ss
    Ks[0, 2] *= ss
    Ks[1, 2] *= ss

    ys, xs = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing="ij")
    d_cam = np.stack(
        [
            (xs + 0.5 - Ks[0, 2]) / Ks[0, 0],
            (ys + 0.5 - Ks[1, 2]) / Ks[1, 1],
            np.ones_like(xs, dtype=np.float64),
        ],
        axis=-1,
    )
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).astype(np.float64)
    return origins, dirs.reshape(-1, 3).astype(np.float64), Hs, Ws


def _mesh_hits(mesh: trimesh.Trimesh, intersector, origins: np.ndarray, dirs: np.ndarray):
    n_rays = len(origins)
    t0 = time.time()
    locs, ray_idx, tri_idx = intersector.intersects_location(origins, dirs, multiple_hits=False)
    print(f"[mesh] camera rays: {len(ray_idx):,}/{n_rays:,} hits ({time.time() - t0:.1f}s)", flush=True)

    hit = np.zeros(n_rays, dtype=bool)
    x_hit = np.zeros((n_rays, 3), dtype=np.float64)
    normals = np.zeros((n_rays, 3), dtype=np.float64)
    depth = np.zeros(n_rays, dtype=np.float64)
    if len(ray_idx) == 0:
        return hit, x_hit, normals, depth

    hit[ray_idx] = True
    x_hit[ray_idx] = locs
    depth[ray_idx] = np.einsum("ij,ij->i", locs - origins[ray_idx], dirs[ray_idx])

    from trimesh.triangles import points_to_barycentric

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)[tri_idx]
    tri = np.stack([verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]], axis=1)
    bary = points_to_barycentric(tri, locs)
    n_hit = (
        vertex_normals[faces[:, 0]] * bary[:, 0:1]
        + vertex_normals[faces[:, 1]] * bary[:, 1:2]
        + vertex_normals[faces[:, 2]] * bary[:, 2:3]
    )
    n_hit /= np.linalg.norm(n_hit, axis=-1, keepdims=True) + 1e-9
    normals[ray_idx] = n_hit
    return hit, x_hit, normals, depth


def _mesh_ao(mesh: trimesh.Trimesh, intersector, x_hit: np.ndarray, normals: np.ndarray,
             hit: np.ndarray, rays: int, radius: float, seed: int = 0) -> np.ndarray:
    ao = np.ones(len(hit), dtype=np.float64)
    if rays <= 0 or not hit.any():
        return ao

    t0 = time.time()
    nh = normals[hit]
    xh = x_hit[hit]
    n_hit = len(xh)
    helper = np.where(
        np.abs(nh[:, 0:1]) < 0.9,
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
    )
    tangent = np.cross(nh, helper)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-9
    bitangent = np.cross(nh, tangent)

    rng = np.random.default_rng(seed)
    u1 = rng.random((rays, n_hit))
    u2 = rng.random((rays, n_hit))
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    sx = (r * np.cos(phi))[..., None]
    sy = (r * np.sin(phi))[..., None]
    sz = np.sqrt(np.maximum(0.0, 1.0 - u1))[..., None]
    ao_dirs = sx * tangent + sy * bitangent + sz * nh
    ao_origins = np.broadcast_to(xh + nh * 1e-4, ao_dirs.shape).copy()

    locs, ray_idx, _ = intersector.intersects_location(
        ao_origins.reshape(-1, 3), ao_dirs.reshape(-1, 3), multiple_hits=False
    )
    dist = np.full(rays * n_hit, np.inf, dtype=np.float64)
    if len(ray_idx):
        flat_origins = ao_origins.reshape(-1, 3)
        dist[ray_idx] = np.linalg.norm(locs - flat_origins[ray_idx], axis=-1)
    occlusion = (dist.reshape(rays, n_hit) < radius).mean(axis=0)
    ao[hit] = 1.0 - occlusion
    print(f"[mesh] AO {rays} rays in {time.time() - t0:.1f}s", flush=True)
    return ao


def _analytic_normals(x_hit: np.ndarray, hit: np.ndarray, model,
                      device: str, chunk: int = 8192) -> np.ndarray:
    """Replace mesh vertex normals with SDF gradient at exact hit points."""
    normals = np.zeros_like(x_hit)
    if not hit.any():
        return normals
    pts = torch.from_numpy(x_hit[hit].astype(np.float32))
    grads = []
    t0 = time.time()
    for i in range(0, len(pts), chunk):
        x = pts[i:i + chunk].to(device).requires_grad_(True)
        g = torch.autograd.grad(model(x).sum(), x)[0]
        grads.append(g.detach().cpu())
    grads_np = torch.cat(grads, dim=0).numpy()
    grads_np /= np.linalg.norm(grads_np, axis=-1, keepdims=True) + 1e-9
    normals[hit] = grads_np
    print(f"[mesh] analytic normals {hit.sum():,} pts in {time.time() - t0:.1f}s", flush=True)
    return normals


def _composite_rgba(rgba: np.ndarray, H: int, W: int, ss: int) -> np.ndarray:
    rgba = rgba.reshape(H, ss, W, ss, 4).mean(axis=(1, 3))
    a = rgba[..., 3:4]
    rgb = rgba[..., :3] / np.clip(a, 1e-6, 1.0)
    return rgb * a + (1.0 - a)


def render_normals_only(mesh: trimesh.Trimesh, intersector, c2w: np.ndarray, K: np.ndarray,
                        H: int, W: int, ss: int,
                        sdf_model=None, sdf_device: str = "cpu"):
    origins, dirs, Hs, Ws = _camera_rays(c2w, K, H, W, ss)
    hit, x_hit, normals, _ = _mesh_hits(mesh, intersector, origins, dirs)
    if sdf_model is not None:
        normals = _analytic_normals(x_hit, hit, sdf_model, sdf_device)
    normal_color = (0.5 * (normals + 1.0)).clip(0, 1)
    alpha = hit.astype(np.float64)[:, None]
    rgba_normals = np.concatenate([normal_color * alpha, alpha], axis=-1).reshape(Hs, Ws, 4)
    return _composite_rgba(rgba_normals, H, W, ss)


def render_one(mesh: trimesh.Trimesh, intersector, c2w: np.ndarray, K: np.ndarray,
               H: int, W: int, ss: int, ao_rays: int, ao_radius: float,
               src_K=None, src_c2w=None, src_imgs=None, color_neighbors: int = 3,
               sdf_model=None, sdf_device: str = "cpu"):
    origins, dirs, Hs, Ws = _camera_rays(c2w, K, H, W, ss)
    hit, x_hit, normals, depth = _mesh_hits(mesh, intersector, origins, dirs)
    if sdf_model is not None:
        normals = _analytic_normals(x_hit, hit, sdf_model, sdf_device)
    ao = _mesh_ao(mesh, intersector, x_hit, normals, hit, ao_rays, ao_radius)

    device = "cpu"
    n_t = torch.from_numpy(normals.astype(np.float32))
    d_t = torch.from_numpy(dirs.astype(np.float32))
    ao_t = torch.from_numpy(ao.astype(np.float32)).unsqueeze(-1)
    shaded = shade(n_t, d_t, ao_t, device).numpy()
    normal_color = (0.5 * (normals + 1.0)).clip(0, 1)

    if src_imgs is not None:
        color = reproject_color(
            torch.from_numpy(x_hit.astype(np.float32)),
            torch.from_numpy(hit),
            torch.from_numpy(normals.astype(np.float32)),
            None,
            src_K,
            src_c2w,
            src_imgs,
            "cpu",
            k_neighbors=color_neighbors,
            ref_center=torch.from_numpy(c2w[:3, 3].astype(np.float32)),
        ).numpy()
    else:
        color = np.ones_like(shaded)

    alpha = hit.astype(np.float64)[:, None]
    rgba_shaded = np.concatenate([shaded * alpha, alpha], axis=-1).reshape(Hs, Ws, 4)
    rgba_normals = np.concatenate([normal_color * alpha, alpha], axis=-1).reshape(Hs, Ws, 4)
    rgba_color = np.concatenate([color * alpha, alpha], axis=-1).reshape(Hs, Ws, 4)

    # Hit-masked SS reduction for depth: only average over rays that actually hit.
    # Non-hit subpixels contributed 0 to the previous mean, creating a dark fringe
    # at silhouettes. depth_avg holds raw along-ray distance in scene units;
    # NaN marks pixels with zero hit coverage so global normalization can ignore them.
    depth_ss = depth.reshape(Hs, Ws)
    hit_ss = hit.reshape(Hs, Ws).astype(np.float64)
    hit_count = hit_ss.reshape(H, ss, W, ss).sum(axis=(1, 3))
    depth_sum = (depth_ss * hit_ss).reshape(H, ss, W, ss).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_avg = np.where(hit_count > 0, depth_sum / hit_count, np.nan)
    hit_alpha = hit_count / float(ss * ss)
    return (
        _composite_rgba(rgba_shaded, H, W, ss),
        _composite_rgba(rgba_normals, H, W, ss),
        depth_avg,
        _composite_rgba(rgba_color, H, W, ss),
        hit_alpha,
    )


def _depth_viz(depth_avg: np.ndarray, hit_alpha: np.ndarray,
               lo: float, hi: float) -> np.ndarray:
    """Inverted-gray depth composited over white using global (lo, hi)."""
    H, W = depth_avg.shape
    norm = np.zeros((H, W), dtype=np.float64)
    valid = np.isfinite(depth_avg)
    if valid.any() and hi > lo:
        norm[valid] = 1.0 - (depth_avg[valid] - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    a = hit_alpha[..., None]
    gray = np.repeat(norm[..., None], 3, axis=-1)
    return gray * a + (1.0 - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="path to a 1-Lip-tracer checkpoint (.pt) — drives MC extraction.")
    ap.add_argument("--mesh", type=Path, default=None,
                    help="alternative to --ckpt: render a precomputed mesh (NeuS/GeoNeuS .ply). "
                         "Skips SDF extraction; --scene is then required.")
    ap.add_argument("--mesh_unscale", action=argparse.BooleanOptionalAction, default=True,
                    help="apply scale_mat_inv_0 from cameras.npz to bring DTU meshes "
                         "(saved in world frame by NeuS/GeoNeuS) into the normalized frame "
                         "that load_views() uses. Default on for DTU.")
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--dataset", choices=["dtu", "lego"], default=None)
    ap.add_argument("--views", type=str, default=None)
    ap.add_argument("--n_views", type=int, default=6)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--down", type=int, default=1)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--mc_res", type=int, default=1024)
    ap.add_argument("--mc_coarse", type=int, default=128,
                    help="coarse grid for narrow-band MC; band is masked from this pass")
    ap.add_argument("--band_k", type=float, default=2.5,
                    help="narrow-band radius in coarse cells (>=2 safe for 1-Lipschitz SDFs)")
    ap.add_argument("--level", type=float, default=0.0,
                    help="marching-cubes iso-level used for rendering")
    ap.add_argument("--mesh_levels", type=str, default=None,
                    help="comma-separated iso-levels to export as meshes, e.g. -0.005,0,0.005,0.010")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--ao_rays", type=int, default=32)
    ap.add_argument("--ao_radius", type=float, default=0.06)
    ap.add_argument("--color_neighbors", type=int, default=3)
    ap.add_argument(
        "--layout",
        choices=["full", "gt-depth-normal-mesh"],
        default="full",
        help="grid layout: full = GT|mesh|normals|depth|color|missing; "
             "gt-depth-normal-mesh = GT|depth|normal|mesh|missing",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("paper_render_marching"))
    ap.add_argument("--save_mesh", action="store_true")
    ap.add_argument("--hull", action="store_true",
                    help="also render visual hull mesh as an extra column")
    ap.add_argument("--hull_res", type=int, default=256,
                    help="voxel carving resolution for the hull")
    ap.add_argument("--hull_bound", type=float, default=None,
                    help="carving bound (default: same as --bound)")
    ap.add_argument("--analytic_normals", action="store_true",
                    help="compute normals from SDF gradient instead of mesh vertex normals")
    args = ap.parse_args()

    if (args.ckpt is None) == (args.mesh is None):
        ap.error("provide exactly one of --ckpt or --mesh")
    if args.mesh is not None:
        if args.scene is None:
            ap.error("--scene is required when using --mesh (no checkpoint config to read from)")
        if args.mesh_levels:
            ap.error("--mesh_levels is only meaningful with --ckpt (needs the SDF volume)")
        if args.analytic_normals:
            print("[info] --analytic_normals ignored: no SDF model when rendering --mesh", flush=True)
            args.analytic_normals = False
    else:
        trained_scene = _resolve_trained_scene(args.ckpt)
        if trained_scene is not None:
            if args.scene is None:
                print(f"[info] scene auto-resolved from checkpoint config: {trained_scene}", flush=True)
                args.scene = trained_scene
            elif trained_scene.resolve() != args.scene.resolve():
                print("[info] using trained scene (overrides --scene):", flush=True)
                print(f"       trained: {trained_scene}", flush=True)
                print(f"       ignored: {args.scene}", flush=True)
                args.scene = trained_scene
        elif args.scene is None:
            ap.error("--scene is required: checkpoint has no config.json with a scene path.")

    if args.dataset is None:
        args.dataset = _infer_dataset(args.scene)
        print(f"[info] dataset inferred from scene path: {args.dataset}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[render] marching-cubes backend  device={device}  "
        f"mc_res={args.mc_res} bound={args.bound} level={args.level:+.4f}",
        flush=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mesh is not None:
        print(f"[mesh] loading precomputed mesh: {args.mesh}", flush=True)
        mesh = trimesh.load(str(args.mesh), process=False, force="mesh")
        print(f"[mesh] {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces", flush=True)
        if args.mesh_unscale and args.dataset == "dtu":
            cam_npz = args.scene / "cameras.npz"
            if cam_npz.exists():
                cn = np.load(cam_npz)
                if "scale_mat_inv_0" in cn:
                    S_inv = cn["scale_mat_inv_0"].astype(np.float64)
                else:
                    S_inv = np.linalg.inv(cn["scale_mat_0"].astype(np.float64))
                v = np.asarray(mesh.vertices, dtype=np.float64)
                v_h = np.concatenate([v, np.ones((len(v), 1))], axis=1)
                v_norm = (v_h @ S_inv.T)[:, :3]
                mesh.vertices = v_norm.astype(np.float32)
                aabb = mesh.bounds
                print(f"[mesh] applied scale_mat_inv_0  AABB now: "
                      f"[{aabb[0]} .. {aabb[1]}]", flush=True)
            else:
                print(f"[mesh] WARN: --mesh_unscale set but {cam_npz} missing — skipping", flush=True)
        sdf_model = None
        sdf_vol = None
    else:
        mesh, sdf_model, sdf_vol = extract_mesh(
            args.ckpt, args.bound, args.mc_res, device,
            level=args.level, chunk=args.chunk, keep_model=args.analytic_normals,
            coarse_res=args.mc_coarse, band_k=args.band_k,
        )
    level_meshes = []
    if args.save_mesh and args.mesh is None:
        mesh.export(args.out_dir / "mesh.ply")
        print(f"[mesh] saved {args.out_dir / 'mesh.ply'}", flush=True)
    if args.mesh_levels:
        for level in [float(x) for x in args.mesh_levels.split(",") if x.strip()]:
            if abs(level - args.level) < 1e-12:
                level_mesh = mesh
            else:
                level_mesh = _mesh_from_volume(sdf_vol, args.bound, args.mc_res, level)
            level_path = args.out_dir / f"mesh_level_{_level_tag(level)}.ply"
            level_mesh.export(level_path)
            print(f"[mesh] saved {level_path}", flush=True)
            level_meshes.append((level, level_mesh))
    del sdf_vol
    intersector = _make_intersector(mesh)
    level_intersectors = [
        (level, level_mesh, intersector if level_mesh is mesh else _make_intersector(level_mesh))
        for level, level_mesh in level_meshes
    ]
    if level_intersectors:
        level_names = " | ".join(f"{level:+.3f}" for level, _, _ in level_intersectors)
        print(f"[mesh] level-normal columns: {level_names}", flush=True)

    hull_mesh = None
    hull_intersector = None
    if args.hull:
        hull_bound = args.hull_bound if args.hull_bound is not None else args.bound
        hull_mesh = extract_hull_mesh(args.scene, args.hull_res, hull_bound)
        if args.save_mesh:
            hull_mesh.export(args.out_dir / "hull.ply")
            print(f"[hull] saved {args.out_dir / 'hull.ply'}", flush=True)
        hull_intersector = _make_intersector(hull_mesh)

    views = (
        load_blender_views(scene=args.scene, split="train", down=1)
        if args.dataset == "lego"
        else load_views(scene=args.scene, down=args.down)
    )
    n_total = views["c2w"].shape[0]
    if args.views:
        view_ids = [int(v) for v in args.views.split(",")]
    else:
        k = min(args.n_views, n_total)
        view_ids = np.linspace(0, n_total - 1, k).round().astype(int).tolist()
    print(
        f"[render] views={view_ids} H={views['H']} W={views['W']} "
        f"ss={args.ss} zoom={args.zoom}",
        flush=True,
    )

    gt_imgs = views.get("images")
    src_K = views["K"].float()
    src_c2w = views["c2w"].float()
    if args.zoom != 1.0:
        views["K"][:, 0, 0] *= args.zoom
        views["K"][:, 1, 1] *= args.zoom

    rows = []
    level_rows = []
    per_view = []
    t_total = time.time()
    for vi in view_ids:
        print(f"[render] === view {vi} ===", flush=True)
        mask = torch.ones(src_c2w.shape[0], dtype=torch.bool)
        mask[vi] = False
        shaded, normal_color, depth, color, hit_alpha = render_one(
            mesh,
            intersector,
            views["c2w"][vi].numpy(),
            views["K"][vi].numpy(),
            views["H"],
            views["W"],
            args.ss,
            args.ao_rays,
            args.ao_radius,
            src_K=src_K[mask],
            src_c2w=src_c2w[mask],
            src_imgs=gt_imgs[mask] if gt_imgs is not None else None,
            color_neighbors=args.color_neighbors,
            sdf_model=sdf_model,
            sdf_device=device,
        )
        if level_intersectors:
            level_normals = []
            for level, level_mesh, level_intersector in level_intersectors:
                print(f"[mesh] rendering level {level:+.4f} normals for view {vi}", flush=True)
                _nrm = render_normals_only(
                    level_mesh,
                    level_intersector,
                    views["c2w"][vi].numpy(),
                    views["K"][vi].numpy(),
                    views["H"],
                    views["W"],
                    args.ss,
                    sdf_model=sdf_model,
                    sdf_device=device,
                )
                level_normals.append(_label_bottom(_nrm, f"level {level:+.4f}"))
            level_row = np.concatenate(level_normals, axis=1)
            level_rows.append(level_row)
        hull_shaded = None
        if hull_mesh is not None:
            print(f"[hull] rendering view {vi}", flush=True)
            hull_shaded, _, _, _, _ = render_one(
                hull_mesh,
                hull_intersector,
                views["c2w"][vi].numpy(),
                views["K"][vi].numpy(),
                views["H"],
                views["W"],
                args.ss,
                ao_rays=0,
                ao_radius=args.ao_radius,
            )
        if gt_imgs is not None:
            gt_raw = gt_imgs[vi].numpy()
            black = gt_raw.sum(-1, keepdims=True) < 1e-6
            gt = np.where(black, 1.0, gt_raw)
        else:
            gt_raw = np.ones_like(shaded)
            gt = gt_raw

        # Missing: GT image has object (non-zero in raw, before black→white conversion)
        # but mesh has zero coverage. gt_raw has background=0 from load_views.
        # When zoom != 1, warp gt_raw to match the rendered K (scale around image centre).
        if args.zoom != 1.0 and gt_imgs is not None:
            from scipy.ndimage import map_coordinates
            H_r, W_r = gt_raw.shape[:2]
            cy, cx = H_r / 2.0, W_r / 2.0
            ys, xs = np.mgrid[0:H_r, 0:W_r].astype(np.float64)
            xs_src = cx + (xs - cx) / args.zoom
            ys_src = cy + (ys - cy) / args.zoom
            gt_raw_warped = np.stack([
                map_coordinates(gt_raw[..., c], [ys_src, xs_src], order=1, mode="constant", cval=0.0)
                for c in range(gt_raw.shape[-1])
            ], axis=-1)
            gt_fg = gt_raw_warped.mean(axis=-1) > 1e-3
        else:
            gt_fg = gt_raw.mean(axis=-1) > 1e-3
        missing = gt_fg & (hit_alpha == 0.0)             # (H, W)
        n_missing = int(missing.sum())
        n_gt_fg = int(gt_fg.sum())
        print(
            f"  missing pixels: {n_missing}/{n_gt_fg} fg "
            f"({100.0 * n_missing / max(n_gt_fg, 1):.1f}% of GT foreground)",
            flush=True,
        )

        # Build missing-area visualization: shaded render + bright red overlay
        missing_vis = shaded.copy()
        missing_vis[missing] = [1.0, 0.12, 0.08]

        per_view.append({
            "vi": vi,
            "gt": gt,
            "shaded": shaded,
            "normal_color": normal_color,
            "depth_raw": depth,
            "hit_alpha": hit_alpha,
            "color": color,
            "missing_vis": missing_vis,
            "hull_shaded": hull_shaded,
        })

    # Global depth normalization across all rendered views so columns are
    # comparable. Use finite hit depths only; non-hit pixels are NaN.
    all_depths = np.concatenate([v["depth_raw"][np.isfinite(v["depth_raw"])]
                                 for v in per_view]) if per_view else np.array([])
    if all_depths.size:
        depth_lo = float(all_depths.min())
        depth_hi = float(all_depths.max())
    else:
        depth_lo, depth_hi = 0.0, 1.0
    print(f"[render] global depth range: [{depth_lo:.4f}, {depth_hi:.4f}] (scene units)", flush=True)

    def u8(x: np.ndarray) -> np.ndarray:
        return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)

    for v, level_row in zip(per_view, level_rows if level_rows else [None] * len(per_view)):
        vi = v["vi"]
        depth_viz = _depth_viz(v["depth_raw"], v["hit_alpha"], depth_lo, depth_hi)
        imageio.imwrite(args.out_dir / f"view{vi:03d}_shaded.png", u8(v["shaded"]))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_normals.png", u8(v["normal_color"]))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_depth.png", u8(depth_viz))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_color.png", u8(v["color"]))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_gt.png", u8(v["gt"]))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_missing.png", u8(v["missing_vis"]))
        if level_row is not None:
            imageio.imwrite(args.out_dir / f"view{vi:03d}_level_normals.png", u8(level_row))
        if v["hull_shaded"] is not None:
            imageio.imwrite(args.out_dir / f"view{vi:03d}_hull.png", u8(v["hull_shaded"]))

        if args.layout == "gt-depth-normal-mesh":
            row = [v["gt"], depth_viz, v["normal_color"], v["shaded"], v["missing_vis"]]
            if v["hull_shaded"] is not None:
                row.insert(4, v["hull_shaded"])  # GT | depth | normal | mesh | hull | missing
        else:
            row = [v["gt"], v["shaded"], v["normal_color"], depth_viz, v["color"], v["missing_vis"]]
            if v["hull_shaded"] is not None:
                row.insert(2, v["hull_shaded"])  # GT | mesh | hull | normals | depth | color | missing
        rows.append(np.concatenate(row, axis=1))

    grid = np.concatenate(rows, axis=0)
    imageio.imwrite(args.out_dir / "grid.png", np.clip(grid * 255 + 0.5, 0, 255).astype(np.uint8))
    if level_rows:
        level_grid = np.concatenate(level_rows, axis=0)
        imageio.imwrite(args.out_dir / "level_normals_grid.png", np.clip(level_grid * 255 + 0.5, 0, 255).astype(np.uint8))
    print(f"[render] total {time.time() - t_total:.1f}s", flush=True)
    if args.layout == "gt-depth-normal-mesh":
        columns = "GT | depth | normal | mesh | hull | missing" if args.hull else "GT | depth | normal | mesh | missing"
    else:
        columns = "GT | mesh | hull | normals | depth | color | missing" if args.hull else "GT | mesh | normals | depth | color | missing"
    print(f"wrote {args.out_dir}/grid.png  ({len(view_ids)} views: {columns})")
    if level_intersectors:
        print(f"wrote {args.out_dir}/level_normals_grid.png  (columns: {level_names})")


if __name__ == "__main__":
    main()
