"""Mesh-based paper renderer.

Extracts a mesh from an SDF checkpoint via marching cubes, then ray-casts the
mesh and shades it with the same `shade()` used by analysis/render_paper.py. AO is
computed from the mesh (cosine-weighted hemisphere occlusion), so the result
does not depend on |grad f| and is fair across ablations.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from lip_tracer.data import load_blender_views, load_views
from render_paper import shade


def extract_mesh(ckpt_path: Path, bound: float, res: int, device: str) -> trimesh.Trimesh:
    from skimage.measure import marching_cubes
    from lip_tracer.model import make_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    enc = ckpt.get("input_encoding", "identity")
    if enc == "neus":
        enc = "pe"
    f = make_model(
        hidden=ckpt.get("hidden", 256),
        depth=ckpt.get("depth", 8),
        group_size=ckpt.get("group_size", 2),
        activation=ckpt.get("activation", "groupsort"),
        input_encoding=enc,
        multires=ckpt.get("multires", 6),
        architecture=ckpt.get("architecture", "cpl"),
    ).to(device).eval()
    f.load_state_dict(state, strict=False)

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    chunk = 4096 if device == "cuda" else 65536
    out_chunks = []
    t0 = time.time()
    n_total = grid.shape[0]
    with torch.no_grad():
        for i in range(0, n_total, chunk):
            out_chunks.append(f(grid[i:i + chunk]).cpu())
    vol = torch.cat(out_chunks).reshape(res, res, res).numpy()
    print(f"[mesh] SDF eval {time.time()-t0:.1f}s  range=[{vol.min():.3f},{vol.max():.3f}]", flush=True)
    if vol.min() > 0 or vol.max() < 0:
        raise ValueError(f"No zero-crossing in SDF grid; try a different --bound.")

    spacing = 2 * bound / (res - 1)
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3,
                                        gradient_direction="ascent")
    verts = (verts - bound).astype(np.float32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=False)
    print(f"[mesh] verts={len(mesh.vertices)} faces={len(mesh.faces)}", flush=True)
    return mesh


def _make_intersector(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector(mesh)
    except Exception:
        return mesh.ray


def render_one(mesh: trimesh.Trimesh, c2w: np.ndarray, K: np.ndarray, H: int, W: int,
               ss: int, ao_rays: int, ao_radius: float):
    inter = _make_intersector(mesh)
    Hs, Ws = H * ss, W * ss
    Ks = K.copy()
    Ks[0, 0] *= ss; Ks[1, 1] *= ss
    Ks[0, 2] *= ss; Ks[1, 2] *= ss

    ys, xs = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing="ij")
    d_cam = np.stack([(xs + 0.5 - Ks[0, 2]) / Ks[0, 0],
                      (ys + 0.5 - Ks[1, 2]) / Ks[1, 1],
                      np.ones_like(xs, dtype=np.float64)], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    o = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).astype(np.float64)
    d = dirs.reshape(-1, 3).astype(np.float64)
    N = o.shape[0]

    t0 = time.time()
    locs, ridx, tidx = inter.intersects_location(o, d, multiple_hits=False)
    print(f"[mesh] camera rays: {len(ridx)}/{N} hits  ({time.time()-t0:.1f}s)", flush=True)

    hit = np.zeros(N, dtype=bool)
    x_hit = np.zeros((N, 3), dtype=np.float64)
    n = np.zeros((N, 3), dtype=np.float64)
    if len(ridx):
        hit[ridx] = True
        x_hit[ridx] = locs
        from trimesh.triangles import points_to_barycentric
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
        faces = np.asarray(mesh.faces)[tidx]
        tri = np.stack([verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]], axis=1)
        bary = points_to_barycentric(tri, locs)
        nn = (vn[faces[:, 0]] * bary[:, 0:1]
              + vn[faces[:, 1]] * bary[:, 1:2]
              + vn[faces[:, 2]] * bary[:, 2:3])
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-9
        n[ridx] = nn

    ao = np.ones(N, dtype=np.float64)
    if ao_rays > 0 and hit.any():
        t1 = time.time()
        Hh = int(hit.sum())
        nh = n[hit]
        helper = np.where(np.abs(nh[:, 0:1]) < 0.9,
                          np.array([[1.0, 0.0, 0.0]]),
                          np.array([[0.0, 1.0, 0.0]]))
        ta = np.cross(nh, helper); ta /= np.linalg.norm(ta, axis=-1, keepdims=True) + 1e-9
        tb = np.cross(nh, ta)
        rng = np.random.default_rng(0)
        u1 = rng.random((ao_rays, Hh)); u2 = rng.random((ao_rays, Hh))
        r = np.sqrt(u1); phi = 2 * np.pi * u2
        sx = (r * np.cos(phi))[..., None]
        sy = (r * np.sin(phi))[..., None]
        sz = np.sqrt(np.maximum(0.0, 1.0 - u1))[..., None]
        ao_d = sx * ta + sy * tb + sz * nh        # (R, Hh, 3)
        ao_o = np.broadcast_to(x_hit[hit] + nh * 1e-4, ao_d.shape)
        ao_d = ao_d.reshape(-1, 3); ao_o = ao_o.reshape(-1, 3).copy()
        loc_ao, ridx_ao, _ = inter.intersects_location(ao_o, ao_d, multiple_hits=False)
        dist = np.full(ao_o.shape[0], np.inf)
        if len(ridx_ao):
            dist[ridx_ao] = np.linalg.norm(loc_ao - ao_o[ridx_ao], axis=-1)
        occl = (dist < ao_radius).reshape(ao_rays, Hh).mean(0)
        ao_full = np.ones(N); ao_full[hit] = 1.0 - occl
        ao = ao_full
        print(f"[mesh] AO {ao_rays} rays  ({time.time()-t1:.1f}s)", flush=True)

    nt = torch.from_numpy(n.astype(np.float32))
    vt = torch.from_numpy(d.astype(np.float32))
    aot = torch.from_numpy(ao.astype(np.float32)).unsqueeze(-1)
    rgb = shade(nt, vt, aot, "cpu").numpy()
    n_color = (0.5 * (n + 1.0)).clip(0, 1)

    alpha = hit.astype(np.float64)[:, None]
    rgba_s = np.concatenate([rgb * alpha, alpha], axis=-1).reshape(Hs, Ws, 4)
    rgba_n = np.concatenate([n_color * alpha, alpha], axis=-1).reshape(Hs, Ws, 4)

    def _comp(rgba):
        rgba = rgba.reshape(H, ss, W, ss, 4).mean(axis=(1, 3))
        a = rgba[..., 3:4]
        rgb_c = rgba[..., :3] / np.clip(a, 1e-6, 1.0)
        return rgb_c * a + (1.0 - a)

    return _comp(rgba_s), _comp(rgba_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--dataset", choices=["dtu", "lego"], default="lego")
    ap.add_argument("--views", type=str, default=None)
    ap.add_argument("--n_views", type=int, default=6)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--mc_res", type=int, default=512)
    ap.add_argument("--ao_rays", type=int, default=24)
    ap.add_argument("--ao_radius", type=float, default=0.05)
    ap.add_argument("--out_dir", type=Path, default=Path("paper_render_mesh"))
    ap.add_argument("--save_mesh", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mesh] device={device}", flush=True)

    mesh = extract_mesh(args.ckpt, args.bound, args.mc_res, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mesh:
        mesh.export(args.out_dir / "mesh.ply")

    views = (load_blender_views(scene=args.scene, split="train", down=1)
             if args.dataset == "lego" else load_views(scene=args.scene))
    n_total = views["c2w"].shape[0]
    if args.views:
        view_ids = [int(v) for v in args.views.split(",")]
    else:
        k = min(args.n_views, n_total)
        view_ids = np.linspace(0, n_total - 1, k).round().astype(int).tolist()
    print(f"[mesh] views={view_ids} H={views['H']} W={views['W']} ss={args.ss}", flush=True)

    gt_imgs = views.get("images")
    rows = []
    for vi in view_ids:
        print(f"[mesh] === view {vi} ===", flush=True)
        shaded, ncolor = render_one(
            mesh, views["c2w"][vi].numpy(), views["K"][vi].numpy(),
            views["H"], views["W"], args.ss, args.ao_rays, args.ao_radius,
        )
        if gt_imgs is not None:
            gt = gt_imgs[vi].numpy()
            black = gt.sum(-1, keepdims=True) < 1e-6
            gt = np.where(black, 1.0, gt)
        else:
            gt = np.ones_like(shaded)

        def _u8(x): return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
        imageio.imwrite(args.out_dir / f"view{vi:03d}_shaded.png", _u8(shaded))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_normals.png", _u8(ncolor))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_gt.png", _u8(gt))
        rows.append(np.concatenate([gt, shaded, ncolor], axis=1))

    grid = np.concatenate(rows, axis=0)
    imageio.imwrite(args.out_dir / "grid.png",
                    np.clip(grid * 255 + 0.5, 0, 255).astype(np.uint8))
    print(f"wrote {args.out_dir}/grid.png", flush=True)


if __name__ == "__main__":
    main()
