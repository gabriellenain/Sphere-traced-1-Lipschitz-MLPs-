#!/usr/bin/env python3
"""Fit the 1-Lipschitz FTheta directly to a GT mesh signed-distance field.

This is an expressivity/optimization diagnostic: no cameras, no tracing, no
photometric loss. If FTheta cannot fit this target, the architecture or
optimization is the bottleneck rather than the reconstruction pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lip_tracer.config import ModelConfig
from lip_tracer.model import FTheta


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


def save_mc_mesh(f: FTheta, out: Path, bound: float, res: int, device: str) -> None:
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
        return
    spacing = (2 * bound / (res - 1),) * 3
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=spacing)
    verts = verts.astype(np.float32) - bound
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(out)
    print(f"saved MC mesh -> {out}")


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
    ap.add_argument("--activation", choices=["groupsort", "nact"], default=mc.activation)
    ap.add_argument("--input-encoding", choices=["identity", "pe"], default=mc.input_encoding)
    ap.add_argument("--multires", type=int, default=mc.multires)
        ap.add_argument("--seed", type=int, default=0)
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
    cfg["mesh"] = str(args.mesh)
    cfg["out_dir"] = str(args.out_dir)
    if cfg.get("load_ckpt") is not None:
        cfg["load_ckpt"] = str(args.load_ckpt)
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.load_ckpt is not None:
        ckpt_data = torch.load(args.load_ckpt, map_location=device)
        f = FTheta(hidden=args.hidden, depth=ckpt_data.get("depth", args.depth),
                   group_size=ckpt_data.get("group_size", args.group_size),
                   activation=ckpt_data.get("activation", args.activation),
                   input_encoding=ckpt_data.get("input_encoding", args.input_encoding),
                   multires=ckpt_data.get("multires", args.multires),
                   ).to(device)
        f.load_state_dict(ckpt_data["f"])
        print(f"loaded checkpoint: {args.load_ckpt}")
        pred_mesh_path = args.out_dir / "pred_mesh.ply"
        save_mc_mesh(f, pred_mesh_path, args.bound, args.mc_res, device)
        save_render_png(pred_mesh_path, args.out_dir / "pred_mesh_render.png")
        print("saved render -> pred_mesh_render.png")
        save_sdf_comparison(mesh, f, args.out_dir / "sdf_comparison.png", args.bound, device)
        print("saved SDF comparison -> sdf_comparison.png")
        return

    print("sampling SDF targets ...")
    data = sample_dataset(mesh, args.bound, args.n_near, args.n_vol,
                          args.near_std, args.sdf_chunk, args.seed)
    near = torch.from_numpy(data["near"]).to(device)
    near_sdf = torch.from_numpy(data["near_sdf"]).to(device)
    vol = torch.from_numpy(data["vol"]).to(device)
    vol_sdf = torch.from_numpy(data["vol_sdf"]).to(device)
    print(f"near={len(near):,} vol={len(vol):,} device={device}")

    f = FTheta(hidden=args.hidden, depth=args.depth, group_size=args.group_size,
               activation=args.activation, input_encoding=args.input_encoding,
               multires=args.multires).to(device)

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

    for step in range(args.steps + 1):
        ni = torch.randint(0, len(near), (n_near_b,), device=device)
        vi = torch.randint(0, len(vol), (n_vol_b,), device=device)
        x = torch.cat([near[ni], vol[vi]], dim=0)
        y = torch.cat([near_sdf[ni], vol_sdf[vi]], dim=0)

        pred = f(x)           # train f.forward directly; f.sdf = f/K_PE used only in sphere tracing
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
            print(f"step {step:6d} loss={loss.item():.6f} near_l1={near_l1:.6f} vol_l1={vol_l1:.6f}")
            if step > 0 and step % 2000 == 0:
                save_loss_plot(history, args.out_dir / "loss.png")

    ckpt = args.out_dir / "checkpoint_gt_sdf.pt"
    torch.save({
        "f": f.state_dict(),
        "group_size": f.group_size,
        "depth": f.depth,
        "activation": f.activation,
        "input_encoding": f.input_encoding,
        "multires": f.multires,
                "mesh": str(args.mesh),
        "bound": args.bound,
    }, ckpt)
    print(f"saved checkpoint -> {ckpt}")
    save_loss_plot(history, args.out_dir / "loss.png")
    pred_mesh_path = args.out_dir / "pred_mesh.ply"
    save_mc_mesh(f, pred_mesh_path, args.bound, args.mc_res, device)
    save_render_png(pred_mesh_path, args.out_dir / "pred_mesh_render.png")
    print("saved render -> pred_mesh_render.png")
    save_sdf_comparison(mesh, f, args.out_dir / "sdf_comparison.png", args.bound, device)
    print("saved SDF comparison -> sdf_comparison.png")


if __name__ == "__main__":
    main()
