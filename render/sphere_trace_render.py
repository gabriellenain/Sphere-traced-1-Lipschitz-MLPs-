"""Minimal paper-figure renderer for an SDF checkpoint.

Produces a clean, publication-quality image: hemisphere lighting (sky/ground),
soft key light, fresnel rim, on a white background, with 2x supersampling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from lip_tracer.config import TraceConfig
from lip_tracer.data import load_blender_views, load_views
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd


def _resolve_trained_scene(ckpt_path: Path) -> Path | None:
    """Return the scene stored in the checkpoint's config.json, or None."""
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return None
    s = cfg.get("scene")
    return Path(s) if s else None


def _infer_dataset(scene: Path) -> str:
    """Guess 'lego' vs 'dtu' from the scene path."""
    parts = str(scene).lower()
    if "nerf_synthetic" in parts or "/lego" in parts:
        return "lego"
    return "dtu"


def load_model(ckpt_path: Path, device: str):
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
    ).to(device)
    f.load_state_dict(state, strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f


def sdf_ao(f, x_hit, n, device, taps=(0.01, 0.025, 0.05, 0.1, 0.2), strength=2.0,
           chunk=65536):
    """Inigo-Quilez-style AO: probe the SDF along the normal at a few radii."""
    occ = torch.zeros(x_hit.shape[0], 1, device=device)
    with torch.no_grad():
        for s in range(0, x_hit.shape[0], chunk):
            e = min(s + chunk, x_hit.shape[0])
            xs = x_hit[s:e]; ns = n[s:e]
            occ_chunk = torch.zeros(xs.shape[0], 1, device=device)
            for i, r in enumerate(taps):
                d = f(xs + ns * r).reshape(-1, 1)
                occ_chunk = occ_chunk + (r - d).clamp_min(0.0) * (0.5 ** i) / r
            occ[s:e] = occ_chunk
    return (1.0 - strength * occ).clamp(0.0, 1.0)


def shade(n, view_dir, ao, device):
    """Paper-grade neutral clay: soft hemisphere + gentle key + subtle rim,
    AO clamped to avoid crushed blacks, mild gamma for print-friendly midtones."""
    sky     = torch.tensor([0.78, 0.79, 0.82], device=device)
    ground  = torch.tensor([0.40, 0.41, 0.43], device=device)
    base    = torch.tensor([0.58, 0.59, 0.61], device=device)
    ambient = torch.tensor([0.32, 0.33, 0.35], device=device)

    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    t = (0.5 * (n @ up) + 0.5).unsqueeze(-1).clamp(0, 1)
    hemi = sky * t + ground * (1.0 - t)

    key_dir = torch.tensor([-0.4, 0.3, 0.85], device=device)
    key_dir = key_dir / key_dir.norm()
    diffuse = (n * key_dir).sum(-1, keepdim=True).clamp(0, 1)

    fill_dir = torch.tensor([0.5, -0.2, 0.4], device=device)
    fill_dir = fill_dir / fill_dir.norm()
    fill = (n * fill_dir).sum(-1, keepdim=True).clamp(0, 1) * 0.15

    ndotv = (n * (-view_dir)).sum(-1, keepdim=True).clamp(0, 1)
    rim = (1.0 - ndotv).pow(4) * 0.10

    ao_soft = ao.clamp(min=0.15).pow(1.5)

    color = base * (0.50 * hemi * ao_soft + 0.65 * diffuse + fill + 0.15 * ambient) + rim
    color = color.clamp(0, 1).pow(1.0 / 2.2)
    return color


def reproject_color(x_hit, hit, n, src_views, src_K, src_c2w, src_imgs, device,
                    k_neighbors=3, ref_center=None):
    """Color each hit point by bilinear sampling into the k nearest training views."""
    import torch.nn.functional as F
    V = src_imgs.shape[0]
    H, W = src_imgs.shape[1:3]
    centers = src_c2w[:, :3, 3].to(device)                       # (V, 3)
    if ref_center is None:
        ref_center = centers.mean(0)
    dists = (centers - ref_center).norm(dim=-1)
    k = min(k_neighbors, V)
    nn = torch.topk(dists, k, largest=False).indices.tolist()

    color_acc = torch.zeros(x_hit.shape[0], 3, device=device)
    weight_acc = torch.zeros(x_hit.shape[0], 1, device=device)
    imgs_chw = src_imgs.permute(0, 3, 1, 2).to(device)           # (V, 3, H, W)
    for vi in nn:
        w2c = torch.linalg.inv(src_c2w[vi]).to(device)
        Kv = src_K[vi].to(device)
        xc = (w2c[:3, :3] @ x_hit.T).T + w2c[:3, 3]              # (N, 3)
        z = xc[:, 2:3]
        front = (z > 1e-4).float()
        uv_h = (Kv @ xc.T).T
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
        in_b = ((uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) &
                (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)).float().unsqueeze(-1)
        grid = torch.stack([uv[:, 0] / (W - 1) * 2 - 1,
                            uv[:, 1] / (H - 1) * 2 - 1], dim=-1)
        N = x_hit.shape[0]
        rgb = F.grid_sample(imgs_chw[vi:vi+1].expand(N, -1, -1, -1),
                            grid.view(N, 1, 1, 2),
                            mode="bilinear", align_corners=True).view(N, 3)
        cam_dir = (src_c2w[vi, :3, 3].to(device) - x_hit)
        cam_dir = cam_dir / cam_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        face = (n * cam_dir).sum(-1, keepdim=True).clamp(0, 1)
        w = in_b * front * (face + 1e-3)
        color_acc = color_acc + rgb * w
        weight_acc = weight_acc + w
        if hit.any():
            print(f"    cam{vi}: in_b={in_b[hit].mean():.3f}  front={front[hit].mean():.3f}  "
                  f"face={face[hit].mean():.3f}  rgb={rgb[hit].mean():.3f}", flush=True)

    valid = weight_acc > 1e-6
    color = torch.where(valid, color_acc / weight_acc.clamp_min(1e-6),
                        torch.ones_like(color_acc))
    h = hit.unsqueeze(-1)
    print(f"  [reproject dbg] hit={hit.float().mean():.3f}  "
          f"valid(hit)={valid[hit].float().mean():.3f}  "
          f"color(hit)={color[hit].mean():.3f}  "
          f"w_acc(hit)={weight_acc[hit].mean():.4f}  "
          f"nn={nn}", flush=True)
    color = torch.where(h, color, torch.ones_like(color))
    return color.clamp(0, 1)


def render(f, c2w, K, H, W, device, cfg, ss=2, chunk=65536,
           src_views=None, src_K=None, src_c2w=None, src_imgs=None,
           color_neighbors=3):
    Hs, Ws = H * ss, W * ss
    Ks = K.copy()
    Ks[0, 0] *= ss; Ks[1, 1] *= ss
    Ks[0, 2] *= ss; Ks[1, 2] *= ss

    ys, xs = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing="ij")
    d_cam = np.stack([(xs + 0.5 - Ks[0, 2]) / Ks[0, 0],
                      (ys + 0.5 - Ks[1, 2]) / Ks[1, 1],
                      np.ones_like(xs)], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    o = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).copy()
    d = dirs.reshape(-1, 3)

    n_rays = o.shape[0]
    n_chunks = (n_rays + chunk - 1) // chunk
    print(f"[render] {Hs}x{Ws} ({n_rays} rays) in {n_chunks} chunks of {chunk}", flush=True)
    t0 = time.time()
    x_hits, hits, ts = [], [], []
    for ci, s in enumerate(range(0, n_rays, chunk)):
        e = min(s + chunk, n_rays)
        x_h, t_h, h = trace_nograd(
            f,
            torch.from_numpy(o[s:e]).float().to(device),
            torch.from_numpy(d[s:e]).float().to(device),
            cfg,
        )
        x_hits.append(x_h); hits.append(h); ts.append(t_h)
        elapsed = time.time() - t0
        rate = (ci + 1) / max(elapsed, 1e-6)
        eta = (n_chunks - ci - 1) / max(rate, 1e-6)
        print(f"[trace] chunk {ci+1}/{n_chunks}  hit_frac={h.float().mean().item():.3f}  "
              f"elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)
    x_hit = torch.cat(x_hits, 0)
    hit   = torch.cat(hits, 0)
    t_all = torch.cat(ts, 0)
    t_hit = t_all[hit]
    depth_img = torch.zeros(n_rays, device=device)
    if t_hit.numel() > 0:
        lo, hi = t_hit.min(), t_hit.max()
        depth_img[hit] = 1.0 - (t_all[hit] - lo) / (hi - lo + 1e-8)
    depth_img = depth_img.unsqueeze(-1).expand(-1, 3)
    view_dir = torch.from_numpy(d).float().to(device)

    print(f"[render] sphere-trace done in {time.time()-t0:.1f}s; computing normals", flush=True)
    t1 = time.time()
    # Exact analytic normals via autograd — no smoothing. Sphere-tracing's whole
    # point is per-pixel exact surface intersection + exact gradient at the hit.
    n = torch.empty_like(x_hit)
    for s in range(0, x_hit.shape[0], chunk):
        e = min(s + chunk, x_hit.shape[0])
        xc = x_hit[s:e].detach().requires_grad_(True)
        with torch.enable_grad():
            y = f(xc)
            g = torch.autograd.grad(y.sum(), xc, create_graph=False)[0]
        n[s:e] = g.detach()
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # Optional gaussian smoothing — OFF by default for maximum detail.
    sigma = getattr(render, "_normal_blur_sigma", 0.0)
    if sigma > 0 and hit.any():
        from scipy.ndimage import gaussian_filter
        nmap = n.reshape(Hs, Ws, 3).cpu().numpy()
        hmask = hit.reshape(Hs, Ws).cpu().numpy()
        smoothed = np.stack([gaussian_filter(nmap[..., c], sigma=sigma) for c in range(3)], axis=-1)
        nmap_out = np.where(hmask[..., None], smoothed, nmap)
        norm_len = np.linalg.norm(nmap_out, axis=-1, keepdims=True).clip(1e-6)
        n = torch.from_numpy((nmap_out / norm_len).reshape(-1, 3)).float().to(device)

    print(f"[render] normals done in {time.time()-t1:.1f}s", flush=True)

    print(f"[render] computing AO", flush=True)
    t2 = time.time()
    ao = sdf_ao(f, x_hit, n, device, chunk=chunk)
    ao = torch.where(hit.unsqueeze(-1), ao, torch.ones_like(ao))
    print(f"[render] AO done in {time.time()-t2:.1f}s", flush=True)

    shaded = shade(n, view_dir, ao, device)
    n_color = (0.5 * (n + 1.0)).clamp(0, 1)

    if src_imgs is not None:
        ref_center = torch.from_numpy(c2w[:3, 3]).float().to(device)
        rep_color = reproject_color(
            x_hit, hit, n, src_views, src_K, src_c2w, src_imgs, device,
            k_neighbors=color_neighbors, ref_center=ref_center,
        )
    else:
        rep_color = torch.ones_like(shaded)

    alpha = hit.float().unsqueeze(-1)
    rgba_s = torch.cat([shaded * alpha, alpha], dim=-1).reshape(Hs, Ws, 4).cpu().numpy()
    rgba_n = torch.cat([n_color * alpha, alpha], dim=-1).reshape(Hs, Ws, 4).cpu().numpy()
    rgba_c = torch.cat([rep_color * alpha, alpha], dim=-1).reshape(Hs, Ws, 4).cpu().numpy()
    depth_raw = depth_img.reshape(Hs, Ws, 3).cpu().numpy()

    def _composite(rgba):
        rgba = rgba.reshape(H, ss, W, ss, 4).mean(axis=(1, 3))
        a = rgba[..., 3:4]
        rgb = rgba[..., :3] / np.clip(a, 1e-6, 1.0)
        return rgb * a + (1.0 - a)

    depth_out = depth_raw.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))  # black background
    return _composite(rgba_s), _composite(rgba_n), depth_out, _composite(rgba_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scene", type=Path, default=None,
                    help="DTU/lego scene directory. Auto-resolved from checkpoint config.json; "
                         "only needed as fallback if config.json is missing.")
    ap.add_argument("--dataset", choices=["dtu", "lego"], default=None,
                    help="Dataset type. Auto-inferred from the scene path if not given.")
    ap.add_argument("--views", type=str, default=None,
                    help="comma-separated view indices; default: 6 evenly spaced")
    ap.add_argument("--n_views", type=int, default=6)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="focal-length scale (<1 zooms out, >1 zooms in)")
    ap.add_argument("--out_dir", type=Path, default=Path("paper_render"))
    ap.add_argument("--color_neighbors", type=int, default=3,
                    help="reproject color from K nearest training views")
    ap.add_argument("--down", type=int, default=1,
                    help="downsample source images by this factor (DTU only)")
    ap.add_argument("--normal-blur-sigma", type=float, default=0.0,
                    help="gaussian sigma for normal smoothing (0 = off — recommended for max detail)")
    args = ap.parse_args()

    # Always use the scene the checkpoint was trained on.
    trained_scene = _resolve_trained_scene(args.ckpt)
    if trained_scene is not None:
        if args.scene is None:
            print(f"[info] scene auto-resolved from checkpoint config: {trained_scene}", flush=True)
            args.scene = trained_scene
        elif trained_scene.resolve() != args.scene.resolve():
            print(f"[info] using trained scene (overrides --scene):", flush=True)
            print(f"       trained: {trained_scene}", flush=True)
            print(f"       ignored: {args.scene}", flush=True)
            args.scene = trained_scene
    elif args.scene is None:
        ap.error("--scene is required: checkpoint has no config.json with a scene path.")

    if args.dataset is None:
        args.dataset = _infer_dataset(args.scene)
        print(f"[info] dataset inferred from scene path: {args.dataset}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[render] device={device}  threads={torch.get_num_threads()}", flush=True)
    print(f"[render] loading ckpt {args.ckpt}", flush=True)
    f = load_model(args.ckpt, device)
    print(f"[render] loading scene {args.scene} (dataset={args.dataset})", flush=True)
    views = (load_blender_views(scene=args.scene, split="train", down=1)
             if args.dataset == "lego" else load_views(scene=args.scene, down=args.down))
    n_total = views["c2w"].shape[0]
    if args.views:
        view_ids = [int(v) for v in args.views.split(",")]
    else:
        k = min(args.n_views, n_total)
        view_ids = np.linspace(0, n_total - 1, k).round().astype(int).tolist()
    print(f"[render] views={view_ids} H={views['H']} W={views['W']} ss={args.ss} zoom={args.zoom}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    render._normal_blur_sigma = args.normal_blur_sigma
    cfg = TraceConfig(iters=128, eps=1e-5, t_far=10.0, newton_steps=8)
    rows = []
    gt_imgs = views.get("images")
    src_K = views["K"].float()       # original K for reprojection — before zoom
    src_c2w = views["c2w"].float()
    if args.zoom != 1.0:
        views["K"][:, 0, 0] *= args.zoom
        views["K"][:, 1, 1] *= args.zoom
    for vi in view_ids:
        print(f"[render] === view {vi} ===", flush=True)
        # exclude the rendered view itself from color sources
        mask = torch.ones(src_c2w.shape[0], dtype=torch.bool); mask[vi] = False
        shaded, ncolor, depth, color = render(
            f, views["c2w"][vi].numpy(), views["K"][vi].numpy(),
            views["H"], views["W"], device, cfg, ss=args.ss,
            src_K=src_K[mask], src_c2w=src_c2w[mask],
            src_imgs=gt_imgs[mask] if gt_imgs is not None else None,
            color_neighbors=args.color_neighbors,
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
        imageio.imwrite(args.out_dir / f"view{vi:03d}_depth.png",   _u8(depth))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_color.png",   _u8(color))
        imageio.imwrite(args.out_dir / f"view{vi:03d}_gt.png",      _u8(gt))
        rows.append(np.concatenate([gt, shaded, ncolor, depth, color], axis=1))

    grid = np.concatenate(rows, axis=0)
    imageio.imwrite(args.out_dir / "grid.png",
                    np.clip(grid * 255 + 0.5, 0, 255).astype(np.uint8))
    print(f"[render] total {time.time()-t_total:.1f}s", flush=True)
    print(f"wrote {args.out_dir}/grid.png  ({len(view_ids)} views: GT | shaded | normals | depth | color)")


if __name__ == "__main__":
    main()
