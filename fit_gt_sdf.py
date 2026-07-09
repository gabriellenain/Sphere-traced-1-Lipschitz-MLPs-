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


def _pred_grad(f_train, x, output_div, create_graph):
    """pred(x)=f(x)/output_div and its input-space gradient g=grad_x pred."""
    x = x.detach().requires_grad_(True)
    pred = f_train(x) / output_div
    g, = torch.autograd.grad(pred.sum(), x, create_graph=create_graph)
    return pred, g, x


def mc_surface_points(f_train, output_div, bound, res, device):
    """Marching-cubes vertices of the current zero set {pred=0}. This is the
    'mesh M_k' DiffCD re-extracts every K_mesh iterations to draw surface samples
    from. Returns None if the field has no zero crossing yet."""
    from skimage.measure import marching_cubes
    xs = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing="ij"), -1).reshape(-1, 3)
    vals = []
    with torch.no_grad():
        for i in range(0, len(grid), 65536):
            vals.append((f_train(grid[i:i + 65536]) / output_div).float().cpu())
    vol = torch.cat(vals).reshape(res, res, res).numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None
    verts, _, _, _ = marching_cubes(vol, level=0.0,
                                    spacing=(2 * bound / (res - 1),) * 3)
    return (verts.astype(np.float32) - bound)


def diffcd_loss(f_train, output_div, cloud_pts, cloud_tree, cloud_idx_batch,
                surf_bank, eik_pts, proj_iters, grad_eps, tau, conv_eps,
                eik_weight):
    """DiffCD (Härenstam-Nielsen et al., ECCV 2024), faithful form:

        L = 1/2 ( mean_i |pred(x~_i)|                      # points -> surface
                + mean_i min_j ||x_i(θ) - x~_j|| )          # surface -> points
          + eik_weight * mean_s (||grad pred(x_s)|| - 1)^2  # eikonal

    pred = f(x)/output_div. Points->surface is the plain |pred| (0th-order distance,
    valid because the eikonal term drives ||grad pred||->1). Surface points x_i(θ)
    come from marching-cubes vertices `surf_bank`, filtered by |pred|<tau and then
    SDF-descent-projected M=proj_iters times; non-converged (|pred|>conv_eps) are
    dropped. NN distance to the cloud is NOT squared (paper uses ||.||).
    """
    # --- Term A: points -> surface = mean |pred| over a cloud minibatch ---
    pred_p = f_train(cloud_pts[cloud_idx_batch]) / output_div
    loss_a = pred_p.abs().mean()

    # --- Term B: surface -> points ---
    loss_b = torch.zeros((), device=cloud_pts.device)
    n_used = 0
    if surf_bank is not None and len(surf_bank) > 0:
        with torch.no_grad():
            cand = surf_bank
            pv = (f_train(cand) / output_div).abs()
            cand = cand[pv < tau]
        if len(cand) > 0:
            # SDF-descent projection: x <- x - pred * grad/||grad||  (M iters).
            x = cand
            for _ in range(proj_iters):
                x = x.detach().requires_grad_(True)
                pred = f_train(x) / output_div
                g, = torch.autograd.grad(pred.sum(), x, create_graph=True)
                gn = g.norm(dim=-1, keepdim=True).clamp_min(grad_eps)
                x = x - pred.unsqueeze(-1) * g / gn
            s = x  # differentiable through the final descent step
            with torch.no_grad():
                converged = (f_train(s) / output_div).abs() < conv_eps
            s = s[converged]
            n_used = len(s)
            if n_used > 0:
                with torch.no_grad():
                    _, idx = cloud_tree.query(s.detach().cpu().numpy(), k=1)
                nn_pts = cloud_pts[torch.as_tensor(idx, device=s.device)]
                loss_b = (s - nn_pts).norm(dim=-1).mean()

    # --- Eikonal: ||grad pred|| -> 1 on sampled points ---
    _, g_e, _ = _pred_grad(f_train, eik_pts, output_div, create_graph=True)
    loss_eik = ((g_e.norm(dim=-1) - 1.0) ** 2).mean()

    loss = 0.5 * (loss_a + loss_b) + eik_weight * loss_eik
    return loss, loss_a.detach(), loss_b.detach(), loss_eik.detach(), n_used


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
    ap.add_argument("--target-scale", type=float, default=1.0,
                    help="fit f ~= target_scale * SDF_gt (f/target_scale ~= SDF_gt). "
                         ">1 demands |grad f|>1 from a 1-Lipschitz net -> should fail.")
    ap.add_argument("--output-div", type=float, default=1.0,
                    help="fit model(gamma(x))/output_div ~= SDF_gt. With raw PE this "
                         "is the (f.gamma)/lambda 1-Lipschitz construction; set "
                         "output_div = lambda_L = sqrt((4^L+2)/3). Target stays plain d.")
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
    ap.add_argument("--loss", choices=["l1", "diffcd"], default="l1",
                    help="l1 = regress pred to GT SDF everywhere (default). "
                         "diffcd = symmetric differentiable Chamfer to an unoriented "
                         "point cloud (Härenstam-Nielsen et al., ECCV 2024).")
    # DiffCD hyperparameters (paper defaults).
    ap.add_argument("--diffcd-eik-weight", type=float, default=0.1,
                    help="eikonal weight lambda (paper: 0.1 clean / 0.5 / 1.0 noisy)")
    ap.add_argument("--diffcd-proj-iters", type=int, default=4,
                    help="SDF-descent projection steps M (paper: 4)")
    ap.add_argument("--diffcd-tau", type=float, default=0.01,
                    help="reject MC candidates with |pred|>tau before projecting")
    ap.add_argument("--diffcd-conv-eps", type=float, default=0.001,
                    help="drop projected points with |pred|>conv_eps (paper: 0.001)")
    ap.add_argument("--diffcd-grad-eps", type=float, default=1e-6,
                    help="numerical floor for ||grad|| in the projection step")
    ap.add_argument("--diffcd-mesh-every", type=int, default=1000,
                    help="re-extract the surface mesh every K iters (paper: 1000)")
    ap.add_argument("--diffcd-mesh-res", type=int, default=256,
                    help="marching-cubes res for surface sampling (paper: 512)")
    ap.add_argument("--diffcd-n-cloud", type=int, default=200000,
                    help="size of the unoriented point cloud P")
    ap.add_argument("--diffcd-n-surf", type=int, default=8192,
                    help="surface-sample candidates drawn from the mesh per step")
    ap.add_argument("--diffcd-n-eik", type=int, default=8192,
                    help="points for the eikonal term (half uniform, half near-cloud)")
    ap.add_argument("--geometric-init", action="store_true",
                    help="NeuS MLP: init f as a sphere SDF (IGR/SAL/IDR) to break the "
                         "unoriented sign ambiguity. Required for DiffCD to converge.")
    ap.add_argument("--geometric-init-radius", type=float, default=0.5,
                    help="radius of the sphere used for geometric init")
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
    # --target-scale c asks the net to fit c * SDF_gt (i.e. f/c ~= SDF_gt). The zero
    # set is unchanged (c*0 = 0, so MC level stays 0), but the slope demand becomes
    # |grad f| = c: a probe of whether the 1-Lipschitz net can represent a c-Lip
    # field. c > 1 should be impossible by construction.
    if args.target_scale != 1.0:
        near_sdf = near_sdf * args.target_scale
        vol_sdf  = vol_sdf  * args.target_scale
        print(f"target-scale: fitting f ~= {args.target_scale:g} * SDF_gt "
              f"(demanded |grad f| = {args.target_scale:g}; net is 1-Lipschitz)")
    print(f"near={len(near):,} vol={len(vol):,} device={device}")

    cloud_pts = cloud_tree = cloud_np = None
    if args.loss == "diffcd":
        from scipy.spatial import cKDTree
        # Unoriented point cloud P (positions only — no normals, no signs).
        cloud_np = mesh.sample(args.diffcd_n_cloud).astype(np.float32)
        cloud_pts = torch.from_numpy(cloud_np).to(device)
        cloud_tree = cKDTree(cloud_np)
        print(f"diffcd: unoriented cloud P = {len(cloud_np):,} pts, "
              f"eik_weight={args.diffcd_eik_weight} proj_iters={args.diffcd_proj_iters} "
              f"mesh_every={args.diffcd_mesh_every} mesh_res={args.diffcd_mesh_res}")

    f = make_model(hidden=args.hidden, depth=args.depth, group_size=args.group_size,
                   activation=args.activation, input_encoding=args.input_encoding,
                   multires=args.multires, architecture=args.architecture,
                   lipschitz_mode=args.lipschitz_mode).to(device)

    if args.geometric_init:
        if not hasattr(f, "geometric_init"):
            raise SystemExit(f"--geometric-init not supported for architecture {args.architecture!r}")
        f.geometric_init(radius=args.geometric_init_radius)
        with torch.no_grad():
            r = args.geometric_init_radius
            probe = torch.tensor([[0., 0., 0.], [r, 0., 0.], [2 * r, 0., 0.]], device=device)
            fp = (f(probe) / args.output_div).tolist()
            print(f"geometric init (r={r}): f(0)={fp[0]:.3f} f(r)={fp[1]:.3f} "
                  f"f(2r)={fp[2]:.3f}  (expect ~ -r, ~0, ~+r)")

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

    surf_bank = None  # marching-cubes vertices of {pred=0}, refreshed periodically
    loss_a = loss_b = loss_eik = torch.zeros((), device=device)
    n_used = 0
    for step in range(start_step, args.steps + 1):
        ni = torch.randint(0, len(near), (n_near_b,), device=device)
        vi = torch.randint(0, len(vol), (n_vol_b,), device=device)
        x = torch.cat([near[ni], vol[vi]], dim=0)
        y = torch.cat([near_sdf[ni], vol_sdf[vi]], dim=0)

        if args.loss == "diffcd":
            # DiffCD (paper-faithful): symmetric Chamfer between the unoriented
            # cloud P and {pred=0}, plus eikonal. No amp (needs double-backward).
            if step % args.diffcd_mesh_every == 0:
                verts = mc_surface_points(f_train, args.output_div, args.bound,
                                          args.diffcd_mesh_res, device)
                surf_bank = (torch.from_numpy(verts).to(device)
                             if verts is not None else None)
            # cloud minibatch for term A
            ci = torch.randint(0, len(cloud_pts), (args.batch,), device=device)
            # surface candidates: random subset of the current MC mesh bank
            bank = None
            if surf_bank is not None and len(surf_bank) > 0:
                bi = torch.randint(0, len(surf_bank),
                                   (min(args.diffcd_n_surf, len(surf_bank)),), device=device)
                bank = surf_bank[bi]
            # eikonal points: half uniform in the domain, half near the cloud
            ne = args.diffcd_n_eik
            unif = (torch.rand(ne // 2, 3, device=device) * 2 - 1) * args.bound
            cj = torch.randint(0, len(cloud_pts), (ne - ne // 2,), device=device)
            near_c = cloud_pts[cj] + torch.randn(ne - ne // 2, 3, device=device) * args.near_std
            eik_pts = torch.cat([unif, near_c], dim=0)
            loss, loss_a, loss_b, loss_eik, n_used = diffcd_loss(
                f_train, args.output_div, cloud_pts, cloud_tree, ci,
                bank, eik_pts, args.diffcd_proj_iters, args.diffcd_grad_eps,
                args.diffcd_tau, args.diffcd_conv_eps, args.diffcd_eik_weight)
        else:
            with amp_ctx():
                # --output-div L: pred = model(gamma(x)) / L. With raw PE,
                # model(gamma(x)) is L-Lipschitz in world space, so pred is
                # 1-Lipschitz and is fit to the plain SDF d. Tests whether a
                # PE-normalized (output-divided) 1-Lipschitz field can represent d.
                # Zero set {pred=0} = {model=0}, unchanged by /L.
                pred = f_train(x) / args.output_div
                loss = F.l1_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                ni_eval = torch.randint(0, len(near), (min(8192, len(near)),), device=device)
                vi_eval = torch.randint(0, len(vol), (min(8192, len(vol)),), device=device)
                pred_near = f(near[ni_eval]) / args.output_div
                gt_near = near_sdf[ni_eval]
                pred_vol = f(vol[vi_eval]) / args.output_div
                gt_vol = vol_sdf[vi_eval]
                near_l1 = (pred_near - gt_near).abs().mean().item()
                vol_l1 = (pred_vol - gt_vol).abs().mean().item()
            history.append((step, loss.item(), near_l1, vol_l1))
            extra = (f" A(p->s)={loss_a.item():.5f} B(s->p)={loss_b.item():.5f} "
                     f"eik={loss_eik.item():.5f} nsurf={n_used}"
                     if args.loss == "diffcd" else "")
            print(f"step {step:6d} loss={loss.item():.6f} near_l1={near_l1:.6f} "
                  f"vol_l1={vol_l1:.6f}{extra}", flush=True)
            if step > 0 and step % 2000 == 0:
                save_loss_plot(history, args.out_dir / "loss.png")

        if args.ckpt_every > 0 and step > start_step and step % args.ckpt_every == 0:
            save_ckpt(step)

    save_ckpt(args.steps)
    print(f"saved checkpoint -> {ckpt_path}")
    save_loss_plot(history, args.out_dir / "loss.png")
    # Final L1 metrics (in SDF units): average the last few logged evals to denoise.
    import json as _json
    tail = history[-5:] if len(history) >= 5 else history
    final_metrics = {
        "final_step": history[-1][0] if history else 0,
        "near_l1": float(np.mean([h[2] for h in tail])),
        "vol_l1":  float(np.mean([h[3] for h in tail])),
        "total_l1": float(np.mean([h[1] for h in tail])),
        "output_div": args.output_div,
        "target_scale": args.target_scale,
    }
    (args.out_dir / "metrics.json").write_text(_json.dumps(final_metrics, indent=2))
    print(f"final L1  near={final_metrics['near_l1']:.6f}  "
          f"vol={final_metrics['vol_l1']:.6f}  total={final_metrics['total_l1']:.6f}")
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
