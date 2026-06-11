#!/usr/bin/env python3
"""Fit the 1-Lipschitz FTheta directly to a GT mesh signed-distance field.

This is an expressivity/optimization diagnostic: no cameras, no tracing, no
photometric loss. If FTheta cannot fit this target, the architecture or
optimization is the bottleneck rather than the reconstruction pipeline.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

from lip_tracer.config import ModelConfig
from lip_tracer.model import FTheta, make_model


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="scene")
    if isinstance(mesh, trimesh.Scene):
        parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise ValueError(f"no triangle mesh found in {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"invalid/empty mesh: {path}")
    mesh.remove_unreferenced_vertices()
    return mesh


def _rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def signed_distance_repo_convention(mesh, pts: np.ndarray, chunk: int,
                                    label: str = "") -> np.ndarray:
    """Return SDF: positive outside, negative inside.

    Uses face-centroid cKDTree to find the nearest face per point, then computes
    the exact closest point on that triangle (one-to-one, O(n_pts) memory).
    Sign from face normal — reliable everywhere, no ray casting.
    """
    import time
    import trimesh
    from scipy.spatial import cKDTree

    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)  # (F, 3)
    face_tree = cKDTree(face_centroids)
    face_normals = mesh.face_normals  # (F, 3)
    triangles = mesh.vertices[mesh.faces]  # (F, 3, 3)

    sdf = np.empty(len(pts), dtype=np.float32)
    n_chunks = (len(pts) + chunk - 1) // chunk
    t0 = time.time()
    for ci, i in enumerate(range(0, len(pts), chunk)):
        batch = pts[i:i + chunk]
        print(f"  [{label}] chunk {ci+1}/{n_chunks} starting  mem={_rss_gb():.2f} GB", flush=True)
        _, fidx = face_tree.query(batch, workers=-1)
        # exact closest point on each nearest triangle — one-to-one, O(n_pts)
        closest = trimesh.triangles.closest_point(triangles[fidx], batch)
        offset = batch - closest
        dist = np.linalg.norm(offset, axis=-1)
        sign = np.sign((offset * face_normals[fidx]).sum(axis=-1))
        sign[sign == 0] = 1.0
        sdf[i:i + chunk] = (sign * dist).astype(np.float32)
        elapsed = time.time() - t0
        done = ci + 1
        eta = elapsed / done * (n_chunks - done)
        print(f"  [{label}] chunk {done}/{n_chunks} done  elapsed {elapsed:.0f}s  ETA {eta:.0f}s  mem={_rss_gb():.2f} GB",
              flush=True)
    return sdf


def sample_dataset(mesh, bound: float, n_near: int, n_vol: int,
                   near_std: float, sdf_chunk: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    surf, face_idx = mesh.sample(n_near, return_index=True)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-8)
    offsets = rng.normal(0.0, near_std, size=(n_near, 1)).astype(np.float32)
    near = surf.astype(np.float32) + offsets * normals
    near_sdf = signed_distance_repo_convention(mesh, near, sdf_chunk, label="near")

    vol = rng.uniform(-bound, bound, size=(n_vol, 3)).astype(np.float32)
    vol_sdf = signed_distance_repo_convention(mesh, vol, sdf_chunk, label="vol")

    return {"near": near, "near_sdf": near_sdf, "vol": vol, "vol_sdf": vol_sdf}


def save_loss_plot(history: list[tuple[int, float, float, float]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(history, dtype=np.float32)
    plt.figure(figsize=(8, 5))
    plt.plot(arr[:, 0], arr[:, 1], label="total")
    plt.plot(arr[:, 0], arr[:, 2], label="near L1")
    plt.plot(arr[:, 0], arr[:, 3], label="volume L1")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def compute_chamfer(gt_mesh, pred_mesh_path: Path, n_points: int = 100_000,
                    seed: int = 0) -> dict:
    """Bidirectional Chamfer + Hausdorff between GT mesh and predicted mesh.

    Returns accuracy (pred→gt), completeness (gt→pred), chamfer, hausdorff — all in
    mesh units (same coordinate system as the SDF fitting).

    The predicted mesh is reduced to its largest connected component (by surface
    area) before sampling, so disconnected MC floaters do not inflate Chamfer.
    """
    import trimesh
    from scipy.spatial import cKDTree

    if not pred_mesh_path.exists():
        return {}
    pred_mesh = trimesh.load(str(pred_mesh_path), force="mesh")
    if len(pred_mesh.faces) == 0:
        return {}
    parts = pred_mesh.split(only_watertight=False)
    if len(parts) > 1:
        pred_mesh = max(parts, key=lambda m: m.area)

    rng = np.random.default_rng(seed)
    gt_pts,   _ = gt_mesh.sample(n_points,   return_index=True)
    pred_pts, _ = pred_mesh.sample(n_points, return_index=True)
    gt_pts   = gt_pts.astype(np.float32)
    pred_pts = pred_pts.astype(np.float32)

    gt_tree   = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    acc_d,  _ = pred_tree.query(gt_pts,   workers=-1)   # completeness: gt → pred
    comp_d, _ = gt_tree.query(pred_pts,   workers=-1)   # accuracy:     pred → gt

    accuracy     = float(comp_d.mean())
    completeness = float(acc_d.mean())
    chamfer      = 0.5 * (accuracy + completeness)
    hausdorff    = float(max(comp_d.max(), acc_d.max()))

    return {
        "accuracy":     accuracy,
        "completeness": completeness,
        "chamfer":      chamfer,
        "hausdorff":    hausdorff,
        "n_points":     n_points,
    }


def save_hq_renders(mesh_path: Path, out: Path, res: int = 1024) -> None:
    """Four-view high-quality render using open3d OffscreenRenderer (PBR shading)."""
    import open3d as o3d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not mesh_path.exists():
        return
    mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh_o3d.has_triangles():
        return
    mesh_o3d.compute_vertex_normals()

    # Normalize to unit sphere for consistent camera placement
    verts = np.asarray(mesh_o3d.vertices)
    center = verts.mean(axis=0)
    scale  = np.abs(verts - center).max()
    mesh_o3d.translate(-center)
    mesh_o3d.scale(1.0 / max(scale, 1e-6), center=[0, 0, 0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader            = "defaultLit"
    mat.base_color        = [0.85, 0.80, 0.74, 1.0]
    mat.base_roughness    = 0.5
    mat.base_reflectance  = 0.3

    # Camera positions: front, right, back, top
    cam_eyes = np.array([
        [ 0.0,  0.0,  3.0],   # front
        [ 3.0,  0.0,  0.0],   # right
        [ 0.0,  0.0, -3.0],   # back
        [ 0.0,  3.0,  0.5],   # top-ish
    ], dtype=np.float64)
    view_labels = ["front", "right", "back", "top"]

    renderer = o3d.visualization.rendering.OffscreenRenderer(res, res)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.add_geometry("mesh", mesh_o3d, mat)
    renderer.scene.scene.set_sun_light(
        direction=[-0.3, -0.8, -0.5],
        color=[1.0, 1.0, 1.0],
        intensity=75000,
    )
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_indirect_light_intensity(25000)

    fov   = 40.0
    up    = np.array([0.0, 1.0, 0.0])
    imgs  = []
    for eye in cam_eyes:
        # Recompute up if looking straight down
        fwd = -eye / (np.linalg.norm(eye) + 1e-8)
        if abs(fwd @ up) > 0.95:
            up_v = np.array([0.0, 0.0, 1.0])
        else:
            up_v = up
        renderer.setup_camera(fov, [0.0, 0.0, 0.0], eye.tolist(), up_v.tolist())
        img = np.asarray(renderer.render_to_image())
        imgs.append(img)

    renderer.scene.remove_geometry("mesh")

    fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 4))
    for ax, img, label in zip(axes, imgs, view_labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()


def save_render_png(mesh_path: Path, out: Path, res: int = 512) -> None:
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not mesh_path.exists():
        return
    m = trimesh.load(str(mesh_path), force="mesh")
    if len(m.faces) == 0:
        return
    v, f = m.vertices.astype(np.float32), m.faces
    lim = max(np.abs(v).max(), 1e-3)
    xs = np.linspace(-lim, lim, res)
    ys = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(xs, ys[::-1])
    origins = np.stack([xx.ravel(), yy.ravel(), np.full(res * res, -lim * 5)], 1).astype(np.float32)
    dirs = np.zeros_like(origins); dirs[:, 2] = 1.0
    locs, idx_ray, idx_tri = m.ray.intersects_location(origins, dirs, multiple_hits=False)
    img = np.ones((res * res, 3))
    if len(locs) > 0:
        fn = m.face_normals[idx_tri]
        fn[fn[:, 2] > 0] *= -1
        key = np.array([-0.3, 0.5, -1.0]); key /= np.linalg.norm(key)
        fill = np.array([0.5, 0.2, -0.5]); fill /= np.linalg.norm(fill)
        shade = (0.75 * (fn @ key).clip(0, 1) + 0.25 * (fn @ fill).clip(0, 1) + 0.1).clip(0, 1)
        img[idx_ray] = shade[:, None] * np.array([0.95, 0.90, 0.82])
    plt.imsave(str(out), img.reshape(res, res, 3).clip(0, 1))


def save_sdf_comparison(mesh, f: FTheta, out: Path, bound: float, device: str,
                        res: int = 256) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    face_tree = cKDTree(face_centroids)
    face_normals = mesh.face_normals
    import trimesh as _trimesh
    triangles = mesh.vertices[mesh.faces]

    xs = np.linspace(-bound, bound, res)
    ys = np.linspace(-bound, bound, res)
    xx, yy = np.meshgrid(xs, ys[::-1])
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(res * res)], 1).astype(np.float32)

    _, fidx = face_tree.query(pts, workers=-1)
    closest = _trimesh.triangles.closest_point(triangles[fidx], pts)
    offset = pts - closest
    dist = np.linalg.norm(offset, axis=-1)
    sign = np.sign((offset * face_normals[fidx]).sum(-1)); sign[sign == 0] = 1.0
    gt_sdf = (sign * dist).reshape(res, res)

    with torch.no_grad():
        pred_sdf = f.sdf(torch.from_numpy(pts).to(device)).float().cpu().numpy().reshape(res, res)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    kw = dict(cmap="RdBu_r", vmin=-0.3, vmax=0.3, extent=[-bound, bound, -bound, bound])
    for ax, data, title in [
        (axes[0], gt_sdf,          "GT SDF (z=0)"),
        (axes[1], pred_sdf,        "Predicted SDF (z=0)"),
        (axes[2], pred_sdf - gt_sdf, "Error"),
    ]:
        im = ax.imshow(data, **kw)
        ax.contour(xs, ys, (gt_sdf if "Error" not in title else pred_sdf),
                   levels=[0.0], colors="k", linewidths=1)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close()


def render_buddha_paper_views(mesh_path: Path, out_dir: Path) -> None:
    """Render pred mesh with the same pipeline as analysis/happy_buddha_mc.py.

    Outputs: shaded strip, normal-map strip, combined grid, and per-azimuth PNGs.
    """
    import sys
    import trimesh
    sys.path.insert(0, str(Path(__file__).parent))
    from render_paper import shade
    from render_paper_marching import _make_intersector, _mesh_hits, _mesh_ao

    if not mesh_path.exists():
        print(f"render_buddha_paper_views: {mesh_path} not found, skipping")
        return

    print("loading predicted mesh for paper render …", flush=True)
    raw  = trimesh.load(str(mesh_path), process=True)
    mesh = raw if isinstance(raw, trimesh.Trimesh) else trimesh.util.concatenate(list(raw.geometry.values()))
    intersector = _make_intersector(mesh)

    H, W  = 1600, 900
    fov_y = 45.0
    fy    = H / (2 * np.tan(np.radians(fov_y / 2)))
    K     = np.array([[fy, 0, W/2], [0, fy, H/2], [0, 0, 1]], np.float64)
    DIST  = 3.2
    up_g  = np.array([0.0, 0.0, 1.0])

    def _c2w(az_deg):
        az  = np.radians(az_deg)
        eye = np.array([DIST * np.sin(az), -DIST * np.cos(az), 0.10])
        fwd = -eye / np.linalg.norm(eye)
        r   = np.cross(fwd, up_g); r /= np.linalg.norm(r)
        d   = np.cross(fwd, r);    d /= np.linalg.norm(d)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = r; c2w[:3, 1] = d; c2w[:3, 2] = fwd; c2w[:3, 3] = eye
        return c2w, eye

    def _render_view(az_deg):
        c2w, eye = _c2w(az_deg)
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        d_cam  = np.stack([(xs + 0.5 - K[0,2]) / K[0,0],
                           (ys + 0.5 - K[1,2]) / K[1,1],
                           np.ones((H, W), np.float64)], axis=-1)
        dirs      = d_cam @ c2w[:3, :3].T
        dirs     /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        origins   = np.broadcast_to(eye, dirs.shape).reshape(-1, 3).astype(np.float64)
        dirs_flat = dirs.reshape(-1, 3).astype(np.float64)

        hit, x_hit, normals, _ = _mesh_hits(mesh, intersector, origins, dirs_flat)
        print(f"  az={az_deg}°  hits {hit.sum():,}/{len(hit):,} ({hit.mean()*100:.1f}%)", flush=True)
        ao = _mesh_ao(mesh, intersector, x_hit, normals, hit, rays=32, radius=0.08)

        n_t    = torch.from_numpy(normals.astype(np.float32))
        d_t    = torch.from_numpy(dirs_flat.astype(np.float32))
        ao_t   = torch.from_numpy(ao.astype(np.float32)).unsqueeze(-1)
        shaded = shade(n_t, d_t, ao_t, "cpu").numpy()

        # same contrast boost as analysis/happy_buddha_mc.py
        lin    = np.clip(shaded.astype(np.float64) ** 2.2, 0, 1)
        lin    = np.clip((lin - 0.12) / (1.0 - 0.12), 0, 1)
        shaded = lin ** (1.0 / 2.2)

        alpha  = hit[:, None].astype(np.float64)
        img    = (shaded * alpha + 1.0 * (1.0 - alpha)).reshape(H, W, 3)
        nmap   = ((0.5 * (normals + 1.0)).clip(0, 1) * alpha + 0.5 * (1.0 - alpha)).reshape(H, W, 3)
        return img, nmap

    def _u8(x): return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)

    views, nmaps = [], []
    for az in [0, 90, 180, 270]:
        img, nmap = _render_view(az)
        imageio.imwrite(str(out_dir / f"pred_buddha_{az}deg.png"),         _u8(img))
        imageio.imwrite(str(out_dir / f"pred_buddha_{az}deg_normals.png"), _u8(nmap))
        views.append(img); nmaps.append(nmap)

    shaded_strip = np.concatenate(views, axis=1)
    normal_strip = np.concatenate(nmaps, axis=1)
    imageio.imwrite(str(out_dir / "pred_buddha_shaded.png"), _u8(shaded_strip))
    imageio.imwrite(str(out_dir / "pred_buddha_normals.png"), _u8(normal_strip))
    imageio.imwrite(str(out_dir / "pred_buddha_grid.png"),
                    _u8(np.concatenate([shaded_strip, normal_strip], axis=0)))
    print(f"saved paper renders → {out_dir}/pred_buddha_*.png", flush=True)


def save_mc_mesh(f: FTheta, out: Path, bound: float, res: int,
                 device: str) -> tuple[int, int]:
    from skimage.measure import marching_cubes
    import trimesh

    xs = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing="ij"), dim=-1).reshape(-1, 3)
    vals = []
    with torch.no_grad():
        for i in range(0, len(grid), 65536):
            vals.append(f(grid[i:i + 65536]).float().cpu())
    vol = torch.cat(vals).reshape(res, res, res).numpy()
    if vol.min() > 0 or vol.max() < 0:
        print("marching cubes skipped: no zero crossing in grid")
        return 0, 0
    spacing = (2 * bound / (res - 1),) * 3
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    verts = verts.astype(np.float32) - bound
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(out)
    print(f"saved MC mesh -> {out}")
    return len(verts), len(faces)


def append_results_row(csv_path: Path, row: dict) -> None:
    """Append one sweep result row to a shared CSV, writing the header once."""
    fields = ["encoding", "architecture", "width", "depth", "group_size",
              "multires", "steps", "near_l1", "vol_l1", "chamfer", "accuracy",
              "completeness", "hausdorff", "n_verts", "n_faces", "out_dir"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"appended results row -> {csv_path}")


def main() -> None:
    mc = ModelConfig()
    ap = argparse.ArgumentParser(description="Regress FTheta to GT mesh SDF.")
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gt_sdf_fit"))
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--near-std", type=float, default=0.02)
    ap.add_argument("--n-near", type=int, default=200000)
    ap.add_argument("--n-vol", type=int, default=200000)
    ap.add_argument("--near-frac", type=float, default=0.7)
    ap.add_argument("--sdf-chunk", type=int, default=20000)
    ap.add_argument("--mc-res", type=int, default=256)
    ap.add_argument("--load-ckpt", type=Path, default=None, help="load checkpoint and skip training")
    ap.add_argument("--hidden", type=int, default=mc.hidden)
    ap.add_argument("--depth", type=int, default=mc.depth)
    ap.add_argument("--group-size", type=int, default=mc.group_size)
    ap.add_argument("--architecture", choices=["cpl", "neus"], default="cpl")
    ap.add_argument("--activation", choices=["groupsort", "nact"], default=mc.activation)
    ap.add_argument("--input-encoding", choices=["identity", "pe"], default=mc.input_encoding)
    ap.add_argument("--multires", type=int, default=mc.multires)
    ap.add_argument("--lipschitz-mode", choices=["none", "uniform", "per_band"],
                    default=mc.lipschitz_mode,
                    help="rescale PE so γ is 1-Lipschitz; 'per_band' recommended for SDFs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-csv", type=Path, default=None,
                    help="append a one-line config+metrics row to this CSV")
    ap.add_argument("--sweep", action="store_true",
                    help="skip the paper-grade renders/SDF plots (fast sweep mode)")
    ap.add_argument("--ckpt-every", type=int, default=200_000,
                    help="write in-training checkpoint every N steps (0 disables)")
    ap.add_argument("--resume", action="store_true",
                    help="if checkpoint_gt_sdf.pt exists in --out-dir, resume training from it")
    ap.add_argument("--amp", action="store_true",
                    help="bfloat16 autocast for forward+loss (no GradScaler needed)")
    ap.add_argument("--compile", action="store_true", dest="compile_model",
                    help="torch.compile the model (large speedup for small MLPs)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.mesh)
    print(f"mesh: {args.mesh}")
    print(f"verts={len(mesh.vertices):,} faces={len(mesh.faces):,} watertight={mesh.is_watertight}")
    print(f"bounds: {mesh.bounds.tolist()}")
    if not mesh.is_watertight:
        print("warning: non-watertight mesh can make signed distance signs unreliable")

    cfg = vars(args).copy()
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.load_ckpt is not None:
        ckpt_data = torch.load(args.load_ckpt, map_location=device)
        f = make_model(
            hidden=args.hidden,
            depth=ckpt_data.get("depth", args.depth),
            group_size=ckpt_data.get("group_size", args.group_size),
            activation=ckpt_data.get("activation", args.activation),
            input_encoding=ckpt_data.get("input_encoding", args.input_encoding),
            multires=ckpt_data.get("multires", args.multires),
            architecture=ckpt_data.get("architecture", args.architecture),
            lipschitz_mode=ckpt_data.get("lipschitz_mode", args.lipschitz_mode),
        ).to(device)
        f.load_state_dict(ckpt_data["f"])
        print(f"loaded checkpoint: {args.load_ckpt}")
        pred_mesh_path = args.out_dir / "pred_mesh.ply"
        n_verts, n_faces = save_mc_mesh(f, pred_mesh_path, args.bound, args.mc_res, device)
        if not args.sweep:
            save_hq_renders(pred_mesh_path, args.out_dir / "pred_mesh_hq.png")
            print("saved HQ render -> pred_mesh_hq.png")
            save_render_png(pred_mesh_path, args.out_dir / "pred_mesh_render.png")
            render_buddha_paper_views(pred_mesh_path, args.out_dir)
            save_sdf_comparison(mesh, f, args.out_dir / "sdf_comparison.png", args.bound, device)
            print("saved SDF comparison -> sdf_comparison.png")
        chamfer = compute_chamfer(mesh, pred_mesh_path, n_points=100_000, seed=0)
        if chamfer:
            import json as _json
            (args.out_dir / "chamfer.json").write_text(_json.dumps(chamfer, indent=2))
            print(
                f"Chamfer: {chamfer['chamfer']:.6f}  "
                f"(acc={chamfer['accuracy']:.6f}  compl={chamfer['completeness']:.6f}  "
                f"hausdorff={chamfer['hausdorff']:.6f})"
            )
        if args.results_csv is not None:
            append_results_row(args.results_csv, {
                "encoding": "pe" if args.input_encoding == "pe" else "none",
                "architecture": f.architecture, "width": args.hidden,
                "depth": f.depth, "group_size": f.group_size,
                "multires": f.multires, "steps": 0,
                "chamfer": chamfer.get("chamfer", ""),
                "accuracy": chamfer.get("accuracy", ""),
                "completeness": chamfer.get("completeness", ""),
                "hausdorff": chamfer.get("hausdorff", ""),
                "n_verts": n_verts, "n_faces": n_faces,
                "out_dir": str(args.out_dir),
            })
        return

    print("sampling SDF targets ...")
    data = sample_dataset(mesh, args.bound, args.n_near, args.n_vol,
                          args.near_std, args.sdf_chunk, args.seed)
    # Degenerate triangles near scan holes produce NaN SDF values — drop them.
    ok_near = np.isfinite(data["near_sdf"])
    ok_vol  = np.isfinite(data["vol_sdf"])
    n_bad = (~ok_near).sum() + (~ok_vol).sum()
    if n_bad:
        print(f"dropping {n_bad} NaN SDF samples from degenerate mesh triangles")
    near     = torch.from_numpy(data["near"][ok_near]).to(device)
    near_sdf = torch.from_numpy(data["near_sdf"][ok_near]).to(device)
    vol      = torch.from_numpy(data["vol"][ok_vol]).to(device)
    vol_sdf  = torch.from_numpy(data["vol_sdf"][ok_vol]).to(device)
    print(f"near={len(near):,} vol={len(vol):,} device={device}")

    f = make_model(hidden=args.hidden, depth=args.depth, group_size=args.group_size,
                   activation=args.activation, input_encoding=args.input_encoding,
                   multires=args.multires, architecture=args.architecture,
                   lipschitz_mode=args.lipschitz_mode).to(device)

    if f.encoder is not None:
        with torch.no_grad():
            x_probe = near[:4096].to(device)
            enc = f.encoder(x_probe)
            d = f.encoder.input_dims
            L = f.encoder.multires
            print("encoder freq stats (mean |feature| over 4096 near-surface pts):")
            print(f"  identity : {enc[:, :d].abs().mean():.5f}  (dims 0–{d-1})")
            for k in range(L):
                base = d + k * 2 * d
                chunk = enc[:, base:base + 2 * d]
                print(f"  k={k} freq=2^{k}={2**k:4d} : {chunk.abs().mean():.5f}  (dims {base}–{base+2*d-1})")

    opt = torch.optim.Adam(f.parameters(), lr=args.lr)
    n_near_b = max(1, int(args.batch * args.near_frac))
    n_vol_b = max(1, args.batch - n_near_b)
    history: list[tuple[int, float, float, float]] = []

    ckpt_path = args.out_dir / "checkpoint_gt_sdf.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        ckpt_data = torch.load(ckpt_path, map_location=device)
        f.load_state_dict(ckpt_data["f"])
        if "opt" in ckpt_data:
            opt.load_state_dict(ckpt_data["opt"])
        start_step = int(ckpt_data.get("step", 0))
        history = list(ckpt_data.get("history", []))
        print(f"resumed from {ckpt_path} at step {start_step}")

    def save_ckpt(step: int) -> None:
        payload = {
            "f": f.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "history": history,
            "architecture": f.architecture,
            "group_size": f.group_size,
            "depth": f.depth,
            "activation": f.activation,
            "input_encoding": f.input_encoding,
            "multires": f.multires,
            "lipschitz_mode": f.lipschitz_mode,
            "mesh": str(args.mesh),
            "bound": args.bound,
        }
        torch.save(payload, ckpt_path)
        torch.save(payload, args.out_dir / f"checkpoint_step{step:09d}.pt")

    f_train = torch.compile(f) if args.compile_model else f
    if args.amp and device == "cuda":
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_ctx = contextlib.nullcontext

    for step in range(start_step, args.steps + 1):
        ni = torch.randint(0, len(near), (n_near_b,), device=device)
        vi = torch.randint(0, len(vol), (n_vol_b,), device=device)
        x = torch.cat([near[ni], vol[vi]], dim=0)
        y = torch.cat([near_sdf[ni], vol_sdf[vi]], dim=0)

        with amp_ctx():
            pred = f_train(x)     # train f.forward directly; f.sdf = f/K_PE used only in sphere tracing
            loss = F.l1_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                ni_eval = torch.randint(0, len(near), (min(8192, len(near)),), device=device)
                vi_eval = torch.randint(0, len(vol), (min(8192, len(vol)),), device=device)
                pred_near = f(near[ni_eval])
                gt_near = near_sdf[ni_eval]
                pred_vol = f(vol[vi_eval])
                gt_vol = vol_sdf[vi_eval]
                near_l1 = (pred_near - gt_near).abs().mean().item()
                vol_l1 = (pred_vol - gt_vol).abs().mean().item()
            history.append((step, loss.item(), near_l1, vol_l1))
            print(f"step {step:6d} loss={loss.item():.6f} near_l1={near_l1:.6f} vol_l1={vol_l1:.6f}", flush=True)
            if step > 0 and step % 2000 == 0:
                save_loss_plot(history, args.out_dir / "loss.png")

        if args.ckpt_every > 0 and step > start_step and step % args.ckpt_every == 0:
            save_ckpt(step)

    save_ckpt(args.steps)
    print(f"saved checkpoint -> {ckpt_path}")
    save_loss_plot(history, args.out_dir / "loss.png")
    pred_mesh_path = args.out_dir / "pred_mesh.ply"
    n_verts, n_faces = save_mc_mesh(f, pred_mesh_path, args.bound, args.mc_res, device)
    if not args.sweep:
        save_hq_renders(pred_mesh_path, args.out_dir / "pred_mesh_hq.png")
        print("saved HQ render -> pred_mesh_hq.png")
        save_render_png(pred_mesh_path, args.out_dir / "pred_mesh_render.png")
        render_buddha_paper_views(pred_mesh_path, args.out_dir)
        save_sdf_comparison(mesh, f, args.out_dir / "sdf_comparison.png", args.bound, device)
        print("saved SDF comparison -> sdf_comparison.png")

    chamfer = compute_chamfer(mesh, pred_mesh_path, n_points=100_000, seed=args.seed)
    if chamfer:
        import json as _json
        (args.out_dir / "chamfer.json").write_text(_json.dumps(chamfer, indent=2))
        print(
            f"Chamfer: {chamfer['chamfer']:.6f}  "
            f"(acc={chamfer['accuracy']:.6f}  compl={chamfer['completeness']:.6f}  "
            f"hausdorff={chamfer['hausdorff']:.6f})"
        )

    if args.results_csv is not None:
        final_near_l1 = history[-1][2] if history else ""
        final_vol_l1 = history[-1][3] if history else ""
        append_results_row(args.results_csv, {
            "encoding": "pe" if args.input_encoding == "pe" else "none",
            "architecture": f.architecture, "width": args.hidden,
            "depth": f.depth, "group_size": f.group_size,
            "multires": f.multires, "steps": args.steps,
            "near_l1": final_near_l1, "vol_l1": final_vol_l1,
            "chamfer": chamfer.get("chamfer", ""),
            "accuracy": chamfer.get("accuracy", ""),
            "completeness": chamfer.get("completeness", ""),
            "hausdorff": chamfer.get("hausdorff", ""),
            "n_verts": n_verts, "n_faces": n_faces,
            "out_dir": str(args.out_dir),
        })


if __name__ == "__main__":
    main()
