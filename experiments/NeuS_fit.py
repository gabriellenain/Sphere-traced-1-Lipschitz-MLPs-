#!/usr/bin/env python3
"""Fit the 1-Lipschitz FTheta SDF to a geometry exported from NeuS.

Minimal workflow:
1. Run NeuS for a few thousand steps outside this repo.
2. Export a mesh from NeuS (PLY / OBJ / GLB / etc).
3. Run this script to distill that geometry into FTheta.

This avoids trying to load arbitrary NeuS checkpoints directly while still
giving a geometry-aware warm start for the 1-Lipschitz model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lip_tracer.config import BLENDER_SCENE, EvalConfig, ModelConfig
from lip_tracer.model import FTheta
from lip_tracer.visual_hull import render_views
from lip_tracer.visualize import visualize


def _load_mesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(mesh_path, force="scene")
    if isinstance(mesh, trimesh.Scene):
        geom = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geom:
            raise ValueError(f"No mesh geometry found in {mesh_path}")
        mesh = trimesh.util.concatenate(geom)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh {mesh_path} is empty")
    mesh.remove_unreferenced_vertices()
    return mesh


def _save_mesh_preview(mesh, out_path: Path, title: str = "NeuS Mesh") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    fig = plt.figure(figsize=(10, 5))
    views = [(20, 35), (20, 125)]
    for i, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            color="lightsteelblue",
            alpha=0.8,
            linewidth=0.1,
            edgecolor="none",
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} view {i}", fontsize=10)
        ax.set_box_aspect((1, 1, 1))
        ctr = verts.mean(0)
        half = max(0.55 * (verts.max(0) - verts.min(0)).max(), 0.1)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved NeuS mesh preview -> {out_path}")


def _chunked_signed_distance(mesh, points: np.ndarray, chunk: int) -> np.ndarray:
    import trimesh

    # trimesh uses the opposite sign convention from this repo:
    #   trimesh:  positive inside, negative outside
    #   here:     negative inside, positive outside
    out = np.empty(points.shape[0], dtype=np.float32)
    for i in range(0, points.shape[0], chunk):
        sd = trimesh.proximity.signed_distance(mesh, points[i:i + chunk])
        out[i:i + chunk] = -sd.astype(np.float32)
    return out


def _sample_training_data(
    mesh,
    bound: float,
    n_surface: int,
    n_near: int,
    n_uniform: int,
    near_std: float,
    chunk: int,
    seed: int,
    target_mode: str,
) -> dict[str, np.ndarray]:
    """Sample regression targets from a mesh.

    target_mode="signed-distance" uses trimesh's signed distance query. This is
    useful for small/watertight meshes, but can be extremely slow and noisy for
    large non-watertight NeuS exports.

    target_mode="normal-offset" trains only from surface samples and normal
    offsets: x = p + t n, target=t. This is much faster and usually the better
    Chamfer-oriented distillation target for non-watertight NeuS meshes.
    """
    rng = np.random.default_rng(seed)

    print(f"  sampling surface points: {n_surface:,}", flush=True)
    surf_pts, face_idx = mesh.sample(n_surface, return_index=True)
    surf_pts = surf_pts.astype(np.float32)
    surf_nrm = mesh.face_normals[face_idx].astype(np.float32)
    surf_nrm /= np.linalg.norm(surf_nrm, axis=-1, keepdims=True).clip(min=1e-8)

    print(f"  sampling near-surface points: {n_near:,}", flush=True)
    near_base, near_face_idx = mesh.sample(n_near, return_index=True)
    near_base = near_base.astype(np.float32)
    near_nrm = mesh.face_normals[near_face_idx].astype(np.float32)
    near_nrm /= np.linalg.norm(near_nrm, axis=-1, keepdims=True).clip(min=1e-8)
    near_offset = rng.normal(scale=near_std, size=(n_near, 1)).astype(np.float32)
    near_pts = near_base + near_offset * near_nrm
    if target_mode == "normal-offset":
        print("  using normal offsets as near-surface SDF targets", flush=True)
        near_sdf = near_offset[:, 0].astype(np.float32)
        uni_pts = np.empty((0, 3), dtype=np.float32)
        uni_sdf = np.empty((0,), dtype=np.float32)
    else:
        print("  querying signed distances for near-surface points", flush=True)
        near_sdf = _chunked_signed_distance(mesh, near_pts, chunk)
        print(f"  sampling uniform points: {n_uniform:,}", flush=True)
        uni_pts = rng.uniform(-bound, bound, size=(n_uniform, 3)).astype(np.float32)
        print("  querying signed distances for uniform points", flush=True)
        uni_sdf = _chunked_signed_distance(mesh, uni_pts, chunk)

    return {
        "surface_points": surf_pts,
        "near_points": near_pts,
        "near_sdf": near_sdf,
        "uniform_points": uni_pts,
        "uniform_sdf": uni_sdf,
    }


def fit_from_neus_mesh(
    mesh_path: Path,
    out_dir: Path,
    scene: Path,
    steps: int,
    batch: int,
    lr: float,
    bound: float,
    near_std: float,
    n_surface: int,
    n_near: int,
    n_uniform: int,
    sdf_chunk: int,
    w_surface: float,
    w_eikonal: float,
    model_cfg: ModelConfig,
    seed: int,
    target_mode: str,
    preview_max_faces: int,
    log_every: int,
) -> Path:
    import trimesh

    mesh = _load_mesh(mesh_path)
    if not mesh.is_watertight:
        print("warning: mesh is not watertight; signed distance may be noisy")

    print(f"mesh: {mesh_path}")
    print(f"  verts={len(mesh.vertices):,} faces={len(mesh.faces):,}")
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    print(f"  mesh bounds: min={bounds[0].tolist()} max={bounds[1].tolist()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if preview_max_faces > 0 and len(mesh.faces) <= preview_max_faces:
        _save_mesh_preview(mesh, out_dir / "neus_mesh_preview.png", title="NeuS")
    else:
        print(
            f"skipping mesh preview: faces={len(mesh.faces):,} "
            f"> preview_max_faces={preview_max_faces:,}",
            flush=True,
        )

    print(f"sampling pseudo-SDF targets from mesh (target_mode={target_mode}) ...", flush=True)
    data = _sample_training_data(
        mesh=mesh,
        bound=bound,
        n_surface=n_surface,
        n_near=n_near,
        n_uniform=n_uniform,
        near_std=near_std,
        chunk=sdf_chunk,
        seed=seed,
        target_mode=target_mode,
    )
    print(
        f"  sampled: surface={len(data['surface_points']):,} "
        f"near={len(data['near_points']):,} uniform={len(data['uniform_points']):,}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = FTheta(
        hidden=model_cfg.hidden,
        depth=model_cfg.depth,
        group_size=model_cfg.group_size,
        activation=model_cfg.activation,
        input_encoding=model_cfg.input_encoding,
        multires=model_cfg.multires,
    ).to(device)
    opt = torch.optim.Adam(f.parameters(), lr=lr)

    surf_pts_t = torch.from_numpy(data["surface_points"]).to(device)
    near_pts_t = torch.from_numpy(data["near_points"]).to(device)
    near_sdf_t = torch.from_numpy(data["near_sdf"]).to(device)
    uni_pts_t = torch.from_numpy(data["uniform_points"]).to(device)
    uni_sdf_t = torch.from_numpy(data["uniform_sdf"]).to(device)

    n_surf = max(1, batch // 4)
    n_near_batch = max(1, batch // 2)
    n_uni = max(0, batch - n_surf - n_near_batch)
    if len(uni_pts_t) == 0:
        n_near_batch = batch - n_surf
        n_uni = 0

    for step in range(steps):
        s_idx = torch.randint(len(surf_pts_t), (n_surf,), device=device)
        n_idx = torch.randint(len(near_pts_t), (n_near_batch,), device=device)

        x_surf = surf_pts_t[s_idx]
        x_near = near_pts_t[n_idx]
        y_near = near_sdf_t[n_idx]
        x_sdf_parts = [x_near]
        y_sdf_parts = [y_near]
        x_eik_parts = [x_surf, x_near]
        if n_uni > 0:
            u_idx = torch.randint(len(uni_pts_t), (n_uni,), device=device)
            x_uni = uni_pts_t[u_idx]
            y_uni = uni_sdf_t[u_idx]
            x_sdf_parts.append(x_uni)
            y_sdf_parts.append(y_uni)
            x_eik_parts.append(x_uni)

        x_sdf = torch.cat(x_sdf_parts, dim=0)
        y_sdf = torch.cat(y_sdf_parts, dim=0)
        pred_sdf = f.sdf(x_sdf)
        sdf_loss = F.mse_loss(pred_sdf, y_sdf)

        surf_loss = f(x_surf).square().mean()

        x_eik = torch.cat(x_eik_parts, dim=0).detach().requires_grad_(True)
        with torch.enable_grad():
            grad = torch.autograd.grad(f.sdf(x_eik).sum(), x_eik, create_graph=True)[0]
        eik_loss = (grad.norm(dim=-1) - 1.0).square().mean()

        loss = sdf_loss + w_surface * surf_loss + w_eikonal * eik_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                surf_abs = f(x_surf).abs().mean().item()
                near_abs = (pred_sdf - y_near.new_tensor(y_sdf)).abs().mean().item()
            print(
                f"[{step:5d}/{steps}] loss={loss.item():.6f} "
                f"sdf={sdf_loss.item():.6f} surf={surf_loss.item():.6f} "
                f"eik={eik_loss.item():.6f} |f(surf)|={surf_abs:.6f} |Δsdf|={near_abs:.6f}",
                flush=True,
            )

    ckpt_path = out_dir / "checkpoint_neus_fit.pt"
    torch.save(
        {
            "f": f.state_dict(),
            "group_size": f.group_size,
            "depth": f.depth,
            "activation": f.activation,
            "input_encoding": f.input_encoding,
            "multires": f.multires,
            "mesh_path": str(mesh_path),
            "bound": bound,
        },
        ckpt_path,
    )
    print(f"saved checkpoint -> {ckpt_path}")

    use_blender = (scene / "transforms_train.json").exists()
    eval_cfg = EvalConfig(mc_res=128, render_down=2)
    visualize(f, eval_cfg=eval_cfg, use_blender=use_blender, out=out_dir / "surface.png")
    if use_blender:
        img = render_views(f, scene=scene, n_views=8, down=4)
        from PIL import Image

        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(out_dir / "render_strip.png")
        print(f"saved render strip -> {out_dir / 'render_strip.png'}")

    return ckpt_path


def main() -> None:
    mc = ModelConfig()
    ap = argparse.ArgumentParser(description="Fit FTheta to a NeuS-exported mesh")
    ap.add_argument("--mesh", type=Path, required=True, help="path to a mesh exported from NeuS")
    ap.add_argument("--out-dir", type=Path, required=True, help="where to save the fitted checkpoint")
    ap.add_argument("--scene", type=Path, default=BLENDER_SCENE, help="scene path for optional render previews")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bound", type=float, default=1.5, help="sampling cube half-extent")
    ap.add_argument("--near-std", type=float, default=0.01, help="surface jitter std for near-surface samples")
    ap.add_argument("--n-surface", type=int, default=50000)
    ap.add_argument("--n-near", type=int, default=100000)
    ap.add_argument("--n-uniform", type=int, default=100000)
    ap.add_argument("--sdf-chunk", type=int, default=20000)
    ap.add_argument(
        "--target-mode",
        type=str,
        default="signed-distance",
        choices=["signed-distance", "normal-offset"],
        help="normal-offset avoids slow signed-distance queries for large non-watertight meshes",
    )
    ap.add_argument(
        "--preview-max-faces",
        type=int,
        default=500000,
        help="skip matplotlib mesh preview when face count exceeds this; 0 always skips",
    )
    ap.add_argument("--w-surface", type=float, default=1.0)
    ap.add_argument("--w-eikonal", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=200,
                    help="print training progress every N steps")
    ap.add_argument("--hidden", type=int, default=mc.hidden)
    ap.add_argument("--depth", type=int, default=mc.depth)
    ap.add_argument("--group-size", type=int, default=mc.group_size)
    ap.add_argument("--activation", type=str, default=mc.activation, choices=["groupsort", "nact"])
    ap.add_argument(
        "--input-encoding",
        type=str,
        default=mc.input_encoding,
        choices=["identity", "pe", "neus"],
        help="'pe' enables positional encoding; 'neus' is accepted as a legacy alias for 'pe'",
    )
    ap.add_argument("--multires", type=int, default=mc.multires)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_encoding = "pe" if args.input_encoding == "neus" else args.input_encoding

    cfg = {
        "mesh": str(args.mesh),
        "scene": str(args.scene),
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "bound": args.bound,
        "near_std": args.near_std,
        "n_surface": args.n_surface,
        "n_near": args.n_near,
        "n_uniform": args.n_uniform,
        "sdf_chunk": args.sdf_chunk,
        "target_mode": args.target_mode,
        "preview_max_faces": args.preview_max_faces,
        "w_surface": args.w_surface,
        "w_eikonal": args.w_eikonal,
        "log_every": args.log_every,
        "model": {
            "hidden": args.hidden,
            "depth": args.depth,
            "group_size": args.group_size,
            "activation": args.activation,
            "input_encoding": input_encoding,
            "multires": args.multires,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    fit_from_neus_mesh(
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        scene=args.scene,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        bound=args.bound,
        near_std=args.near_std,
        n_surface=args.n_surface,
        n_near=args.n_near,
        n_uniform=args.n_uniform,
        sdf_chunk=args.sdf_chunk,
        w_surface=args.w_surface,
        w_eikonal=args.w_eikonal,
        model_cfg=ModelConfig(
            hidden=args.hidden,
            depth=args.depth,
            group_size=args.group_size,
            activation=args.activation,
            input_encoding=input_encoding,
            multires=args.multires,
        ),
        seed=args.seed,
        target_mode=args.target_mode,
        preview_max_faces=args.preview_max_faces,
        log_every=max(1, args.log_every),
    )


if __name__ == "__main__":
    main()
