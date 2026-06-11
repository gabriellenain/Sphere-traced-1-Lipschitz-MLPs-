#!/usr/bin/env python3
"""Diagnose why sphere-traced SDF has holes that the visual hull does not.

Produces a multi-panel PNG for one training view showing:
  A) Hit map (green=hit, red=miss)
  B) sdf_min map — closest the tracer got along each ray (log scale)
  C) |∇f| map at the ray origin — near-zero means flat SDF region
  D) SDF profiles f(o + t*d) along ~10 sampled hole rays — do zero crossings exist?
  E) SDF profiles along ~5 hit rays for reference
  F) Histogram of sdf_min values, split by hit/miss

Usage:
    python diagnose_holes.py <run_dir> [--view 0] [--down 4] [--t-steps 400] [--out diagnose.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model(run_dir: Path, device: str):
    from lip_tracer.train import load_config_json
    cfg = load_config_json(run_dir / "config.json")
    ckpt_path = run_dir / "checkpoint_best_loss.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoint_latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from lip_tracer.model import make_model
    mc = cfg.model
    f = make_model(hidden=mc.hidden, depth=mc.depth, group_size=mc.group_size,
                   activation=mc.activation, input_encoding=mc.input_encoding,
                   multires=mc.multires, architecture=getattr(mc, "architecture", "cpl"))
    f.load_state_dict(ckpt["f"], strict=False)
    f = f.to(device).eval()
    return f, cfg


def _make_rays(c2w: np.ndarray, K: np.ndarray, H: int, W: int, down: int, device: str):
    """Return ray origins (H*W, 3) and directions (H*W, 3) for a camera."""
    Hd, Wd = H // down, W // down
    ys, xs = np.meshgrid(np.arange(Hd) * down + down / 2 - 0.5,
                         np.arange(Wd) * down + down / 2 - 0.5, indexing="ij")
    dirs_cam = np.stack([(xs - K[0, 2]) / K[0, 0],
                         (ys - K[1, 2]) / K[1, 1],
                         np.ones_like(xs)], axis=-1)  # (Hd, Wd, 3)
    R = c2w[:3, :3]
    dirs_world = (dirs_cam.reshape(-1, 3) @ R.T).astype(np.float32)
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8
    origin = np.broadcast_to(c2w[:3, 3].astype(np.float32), (Hd * Wd, 3)).copy()
    o = torch.from_numpy(origin).to(device)
    d = torch.from_numpy(dirs_world).to(device)
    return o, d, Hd, Wd


def _trace_with_diagnostics(f, o, d, cfg_trace, t_steps: int, device: str, chunk: int = 4096):
    """Run sphere tracing + collect sdf_min and grad-norm at origin."""
    from lip_tracer.config import TraceConfig
    N = o.shape[0]
    hit     = torch.zeros(N, dtype=torch.bool, device=device)
    t_out   = torch.zeros(N, device=device)
    sdf_min = torch.full((N,), float("inf"), device=device)

    with torch.no_grad():
        for i in range(0, N, chunk):
            o_b = o[i:i + chunk]; d_b = d[i:i + chunk]
            B = o_b.shape[0]
            t  = torch.zeros(B, device=device)
            sm = torch.full((B,), float("inf"), device=device)
            cv = torch.zeros(B, dtype=torch.bool, device=device)
            sdf_last = torch.zeros(B, device=device)
            for _ in range(cfg_trace.iters):
                esc = t >= cfg_trace.t_far
                if not (~(cv | esc)).any(): break
                sdf_last = f(o_b + t.unsqueeze(-1) * d_b)
                sm = torch.minimum(sm, sdf_last.abs())
                cv  = sdf_last.abs() < cfg_trace.eps
                t   = t + torch.where(cv | esc, torch.zeros_like(t), sdf_last)
            sdf_last = f(o_b + t.unsqueeze(-1) * d_b)
            sm = torch.minimum(sm, sdf_last.abs())
            h  = (sdf_last.abs() < cfg_trace.eps) & (t < cfg_trace.t_far) & (t >= 0)
            hit[i:i + chunk]     = h
            t_out[i:i + chunk]   = t
            sdf_min[i:i + chunk] = sm

    # gradient norm at ray origin — chunked to avoid keeping all activations in RAM
    grad_chunks = []
    for i in range(0, N, chunk):
        o_b = o[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(o_b).sum(), o_b)[0].detach()
        grad_chunks.append(g.norm(dim=-1).cpu())
    grad_norm = torch.cat(grad_chunks)

    return hit.cpu(), t_out.cpu(), sdf_min.detach().cpu(), grad_norm


def _ray_sdf_profiles(f, o, d, t_far: float, t_steps: int, indices, device: str):
    """Sample f along selected rays at t_steps uniform steps. Returns (len(indices), t_steps)."""
    ts = torch.linspace(0.0, t_far, t_steps, device=device)  # (T,)
    profiles = []
    with torch.no_grad():
        for idx in indices:
            o_r = o[idx:idx + 1].to(device)
            d_r = d[idx:idx + 1].to(device)
            pts = o_r + ts.unsqueeze(-1) * d_r  # (T, 3)
            profiles.append(f(pts).cpu().numpy())
    return np.stack(profiles), ts.cpu().numpy()  # (R, T), (T,)


# ── main ─────────────────────────────────────────────────────────────────────

def _color_variance_at_holes(
    o: torch.Tensor, d: torch.Tensor,
    sdf_min: torch.Tensor, hit: torch.Tensor,
    c2ws_all, Ks_all, img_paths,
    H: int, W: int, ref_vi: int, n_views: int,
) -> np.ndarray:
    """Per-hole-pixel colour std across n_views other cameras.

    Back-projects each miss pixel using sdf_min as a proxy depth, then
    projects that 3D point into n_views randomly chosen other cameras and
    samples the RGB colour.  Returns an (N,) float32 array with NaN for
    hit pixels.

    Low std  (<0.05) → colours agree across views → topology/init problem.
                       fg mask loss should be enough.
    High std (>0.15) → colours disagree → view-dependent / specular.
                       fg loss will fight photo loss; depth prior needed.
    """
    from PIL import Image as _PIL

    N = o.shape[0]
    miss_mask = ~hit
    if not miss_mask.any() or not img_paths:
        return np.full(N, np.nan, dtype=np.float32)

    # proxy 3D points for hole rays
    x3d_miss = (o[miss_mask] + sdf_min[miss_mask].unsqueeze(-1) * d[miss_mask]).cpu()  # (M, 3)
    M = x3d_miss.shape[0]

    # pick n_views cameras excluding the reference view
    cands = [i for i in range(len(img_paths)) if i != ref_vi]
    chosen = np.random.default_rng(0).choice(cands, size=min(n_views, len(cands)), replace=False)

    colors_per_view = []
    for vi in chosen:
        c2w = c2ws_all[vi] if isinstance(c2ws_all[vi], torch.Tensor) else torch.tensor(c2ws_all[vi], dtype=torch.float32)
        K   = torch.tensor(Ks_all[vi], dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        R, t_cam = w2c[:3, :3], w2c[:3, 3]

        xc  = (R @ x3d_miss.T).T + t_cam          # (M, 3)
        uv_h = (K @ xc.T).T                        # (M, 3)
        uv   = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)

        valid = ((xc[:, 2] > 0.01)
                 & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < H)).numpy().astype(bool)

        img = np.array(_PIL.open(img_paths[vi]).convert("RGB"), dtype=np.float32) / 255.0
        colors = np.full((M, 3), np.nan, dtype=np.float32)
        if valid.any():
            ix = uv[:, 0].numpy()[valid].clip(0, W - 1).astype(np.int32)
            iy = uv[:, 1].numpy()[valid].clip(0, H - 1).astype(np.int32)
            colors[valid] = img[iy, ix]
        colors_per_view.append(colors)

    stack = np.stack(colors_per_view, axis=0)           # (V, M, 3)
    color_std = np.nanstd(stack, axis=0).mean(axis=-1)  # (M,) — mean over RGB

    result = np.full(N, np.nan, dtype=np.float32)
    result[miss_mask.numpy()] = color_std
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--view",    type=int,   default=0)
    ap.add_argument("--down",    type=int,   default=4)
    ap.add_argument("--t-steps", type=int,   default=400)
    ap.add_argument("--n-hole-rays", type=int, default=12)
    ap.add_argument("--n-hit-rays",  type=int, default=5)
    ap.add_argument("--color-views", type=int, default=6,
                    help="how many other cameras to use for colour-consistency check at hole pixels")
    ap.add_argument("--out",     type=Path,  default=None)
    ap.add_argument("--device",  type=str,   default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = args.out or (args.run_dir / "diagnose_holes.png")

    print(f"[diagnose] loading model from {args.run_dir} …")
    f, cfg = _load_model(args.run_dir, device)

    # load cameras only (avoid loading all images into RAM)
    scene = cfg.scene
    is_blender = (scene / "transforms_train.json").exists()
    if is_blender:
        import json
        from PIL import Image as _PIL
        meta = json.loads((scene / "transforms_train.json").read_text())
        frames = meta["frames"]
        fl = meta.get("fl_x") or meta.get("camera_angle_x")
        # collect c2ws
        c2ws_all, Ks_all, img_paths = [], [], []
        for fr in frames:
            c2w_raw = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
            # blender convention: flip Y and Z
            c2w_bl = c2w_raw.clone()
            c2w_bl[:3, 1] *= -1; c2w_bl[:3, 2] *= -1
            c2ws_all.append(c2w_bl)
            p = scene / (fr["file_path"] + (".png" if not fr["file_path"].endswith(".png") else ""))
            img_paths.append(p)
        # load one image to get H, W and K
        _im0 = _PIL.open(img_paths[0])
        H, W = _im0.height, _im0.width
        if "fl_x" in meta:
            fx, fy = meta["fl_x"], meta.get("fl_y", meta["fl_x"])
            cx, cy = meta.get("cx", W / 2), meta.get("cy", H / 2)
        else:
            fov = meta["camera_angle_x"]
            fx = fy = W / (2 * np.tan(fov / 2))
            cx, cy = W / 2, H / 2
        K_single = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        Ks_all = [K_single] * len(c2ws_all)
        vi = min(args.view, len(c2ws_all) - 1)
        c2w = c2ws_all[vi].numpy()
        K   = Ks_all[vi]
        # load only the one GT image
        img_gt_pil = _PIL.open(img_paths[vi]).convert("RGBA")
        img_gt = np.array(img_gt_pil, dtype=np.float32)[..., :3] / 255.0
    else:
        from lip_tracer.data import load_views
        views = load_views(scene)
        vi = min(args.view, views["c2w"].shape[0] - 1)
        c2w = views["c2w"][vi].numpy()
        K   = views["K"][vi].numpy()
        H, W = views["H"], views["W"]
        img_gt = views["images"][vi].numpy() if "images" in views else None
        c2ws_all = [views["c2w"][i] for i in range(views["c2w"].shape[0])]
        Ks_all   = [views["K"][i].numpy() for i in range(views["K"].shape[0])]
        img_paths = []  # DTU images already in views["images"], handled below

    print(f"[diagnose] building rays for view {vi} at down={args.down} …")
    o, d, Hd, Wd = _make_rays(c2w, K, H, W, args.down, device)
    N = Hd * Wd

    print(f"[diagnose] tracing {N:,} rays ({Hd}×{Wd}) …")
    hit, t_out, sdf_min, grad_norm = _trace_with_diagnostics(
        f, o, d, cfg.trace, args.t_steps, device)

    hit_map      = hit.reshape(Hd, Wd).numpy()
    sdf_min_map  = sdf_min.reshape(Hd, Wd).numpy()
    grad_map     = grad_norm.reshape(Hd, Wd).numpy()

    # pick sample rays: deepest holes (largest sdf_min among misses)
    miss_idx = torch.where(~hit)[0]
    hit_idx  = torch.where(hit)[0]
    rng = np.random.default_rng(42)

    # spread hole rays spatially: pick from different grid quadrants
    if len(miss_idx) > 0:
        # sort by sdf_min descending (worst holes first) then subsample
        order = sdf_min[miss_idx].argsort(descending=True)
        top   = miss_idx[order[:min(len(order), 200)]].numpy()
        chosen_hole = rng.choice(top, size=min(args.n_hole_rays, len(top)), replace=False)
    else:
        chosen_hole = np.array([], dtype=np.int64)

    chosen_hit = (hit_idx[rng.choice(len(hit_idx), size=min(args.n_hit_rays, len(hit_idx)),
                                     replace=False)].numpy()
                  if len(hit_idx) > 0 else np.array([], dtype=np.int64))

    print(f"[diagnose] computing colour variance at hole pixels ({args.color_views} views) …")
    color_std = _color_variance_at_holes(
        o.cpu(), d.cpu(), sdf_min, hit,
        c2ws_all, Ks_all, img_paths,
        H, W, vi, args.color_views,
    )
    color_std_map = color_std.reshape(Hd, Wd)
    hole_color_std = color_std[~hit.numpy()]
    mean_hole_std  = float(np.nanmean(hole_color_std)) if len(hole_color_std) > 0 else float("nan")

    print(f"[diagnose] profiling {len(chosen_hole)} hole rays + {len(chosen_hit)} hit rays …")
    ts_arr = None
    hole_profiles = hit_profiles = None
    if len(chosen_hole) > 0:
        hole_profiles, ts_arr = _ray_sdf_profiles(f, o, d, cfg.trace.t_far, args.t_steps,
                                                   chosen_hole, device)
    if len(chosen_hit) > 0:
        hit_profiles, ts_arr2 = _ray_sdf_profiles(f, o, d, cfg.trace.t_far, args.t_steps,
                                                   chosen_hit, device)
        if ts_arr is None:
            ts_arr = ts_arr2

    # ── plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BG = "#0d0d0d"
    plt.style.use("dark_background")

    n_profile_rows = 2 if (hole_profiles is not None or hit_profiles is not None) else 0
    n_rows = 2 + n_profile_rows
    fig = plt.figure(figsize=(20, 5 * n_rows), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.42, wspace=0.35)

    def _ax(row, col, **kw):
        a = fig.add_subplot(gs[row, col], facecolor=BG, **kw)
        a.tick_params(labelsize=7)
        return a

    # ── Row 0: hit map, sdf_min (log), |∇f|, GT image ────────────────────────
    ax_hit = _ax(0, 0)
    rgb_hit = np.where(hit_map[..., None],
                       np.array([[[0.2, 0.8, 0.2]]]),
                       np.array([[[0.9, 0.2, 0.2]]]))
    ax_hit.imshow(rgb_hit)
    ax_hit.set_title(f"Hit map  ({hit_map.sum():,} hit / {(~hit_map).sum():,} miss)", fontsize=9)
    ax_hit.axis("off")

    ax_smin = _ax(0, 1)
    eps = cfg.trace.eps
    smin_vis = np.log10(np.clip(sdf_min_map, 1e-5, None))
    log_eps = np.log10(eps)
    im_smin = ax_smin.imshow(smin_vis, cmap="plasma_r",
                              vmin=np.log10(1e-5), vmax=np.log10(cfg.trace.t_far))
    cb = fig.colorbar(im_smin, ax=ax_smin, fraction=0.046, pad=0.02)
    cb.set_label("log10(sdf_min)", fontsize=7)
    cb.ax.axhline(log_eps, color="cyan", lw=1.5, label=f"eps={eps}")
    cb.ax.tick_params(labelsize=7)
    ax_smin.set_title("sdf_min along ray  (cyan = eps threshold)", fontsize=9)
    ax_smin.axis("off")
    # overlay hit contour
    ax_smin.contour(hit_map.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)

    ax_grad = _ax(0, 2)
    gp = np.percentile(grad_map, 99)
    im_grad = ax_grad.imshow(grad_map, cmap="inferno", vmin=0, vmax=max(gp, 1e-3))
    fig.colorbar(im_grad, ax=ax_grad, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    ax_grad.set_title("|∇f| at ray origin  (0=flat SDF, 1=proper eikonal)", fontsize=9)
    ax_grad.axis("off")
    ax_grad.contour(hit_map.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)

    ax_img = _ax(0, 3)
    if img_gt is not None:
        ax_img.imshow(img_gt[:Hd * args.down:args.down, :Wd * args.down:args.down])
        ax_img.set_title(f"GT image (view {vi})", fontsize=9)
    else:
        ax_img.text(0.5, 0.5, "no GT image", transform=ax_img.transAxes,
                    ha="center", va="center", color="white")
    ax_img.axis("off")
    ax_img.contour(hit_map.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)

    # ── Row 1: histograms + summary ───────────────────────────────────────────
    ax_h1 = _ax(1, 0)
    sdf_min_d = sdf_min.detach()
    vals_hit  = sdf_min_d[hit].numpy()  if hit.any()  else np.array([0.0])
    vals_miss = sdf_min_d[~hit].numpy() if (~hit).any() else np.array([1.0])
    bins = np.logspace(-5, np.log10(cfg.trace.t_far + 1e-3), 80)
    ax_h1.hist(vals_hit,  bins=bins, color="#00c8ff", alpha=0.7, label=f"hit  ({len(vals_hit):,})",  density=True)
    ax_h1.hist(vals_miss, bins=bins, color="#ff4444", alpha=0.7, label=f"miss ({len(vals_miss):,})", density=True)
    ax_h1.axvline(eps, color="cyan", lw=1.2, ls="--", label=f"eps={eps}")
    ax_h1.set_xscale("log")
    ax_h1.set_xlabel("sdf_min", fontsize=8)
    ax_h1.set_ylabel("density", fontsize=8)
    ax_h1.set_title("sdf_min distribution: hit vs miss", fontsize=9)
    ax_h1.legend(fontsize=7, framealpha=0.2)

    ax_h2 = _ax(1, 1)
    ax_h2.hist(grad_map[hit_map],  bins=50, color="#00c8ff", alpha=0.7, label="hit",  density=True)
    ax_h2.hist(grad_map[~hit_map], bins=50, color="#ff4444", alpha=0.7, label="miss", density=True)
    ax_h2.set_xlabel("|∇f| at origin", fontsize=8)
    ax_h2.set_ylabel("density", fontsize=8)
    ax_h2.set_title("|∇f| distribution: hit vs miss", fontsize=9)
    ax_h2.legend(fontsize=7, framealpha=0.2)

    # summary text
    miss_smin_median = float(np.median(vals_miss))
    miss_above_eps   = float((vals_miss > eps).mean())
    hit_frac         = float(hit.float().mean())
    grad_hit_med     = float(np.median(grad_map[hit_map]))  if hit_map.any()  else 0.0
    grad_miss_med    = float(np.median(grad_map[~hit_map])) if (~hit_map).any() else 0.0

    if mean_hole_std < 0.05:
        color_verdict = "LOW  → topology/init problem  (fg loss should fix)"
        color_verdict_c = "#00ff88"
    elif mean_hole_std > 0.15:
        color_verdict = "HIGH → specular/view-dep.  (fg loss will fight photo)"
        color_verdict_c = "#ff4444"
    else:
        color_verdict = "MED  → mixed causes"
        color_verdict_c = "#ffaa00"

    ax_sum = _ax(1, 2)
    ax_sum.axis("off")
    summary = (
        f"view:            {vi}\n"
        f"resolution:      {Hd}×{Wd}  (down={args.down})\n"
        f"hit rate:        {hit_frac:.1%}\n\n"
        f"miss sdf_min median:  {miss_smin_median:.4f}\n"
        f"miss sdf_min > eps:   {miss_above_eps:.1%}\n"
        f"  → {miss_above_eps:.0%} of misses never got close\n\n"
        f"median |∇f| hit:   {grad_hit_med:.3f}\n"
        f"median |∇f| miss:  {grad_miss_med:.3f}\n"
        f"  (1.0 = perfect eikonal)\n\n"
        f"colour std at holes: {mean_hole_std:.3f}\n"
        f"  ({args.color_views} views, proxy depth=sdf_min)\n\n"
        f"trace iters:     {cfg.trace.iters}\n"
        f"trace eps:       {cfg.trace.eps}\n"
        f"w_eikonal:       {cfg.train.w_eikonal}\n"
        f"w_sil:           {cfg.train.w_sil}\n"
        f"w_free:          {cfg.train.w_free}"
    )
    ax_sum.text(0.05, 0.97, summary, transform=ax_sum.transAxes,
                fontsize=9, va="top", ha="left", family="monospace",
                color="white", linespacing=1.6)

    # colour variance map
    ax_cvar = _ax(1, 3)
    cvar_vis = np.where(~hit.numpy().reshape(Hd, Wd), color_std_map, np.nan)
    im_cvar = ax_cvar.imshow(cvar_vis, cmap="RdYlGn_r", vmin=0.0, vmax=0.3)
    cb_cvar = fig.colorbar(im_cvar, ax=ax_cvar, fraction=0.046, pad=0.02)
    cb_cvar.set_label("colour std (miss pixels only)", fontsize=7)
    cb_cvar.ax.tick_params(labelsize=7)
    ax_cvar.set_title(
        f"Colour consistency at holes  (mean={mean_hole_std:.3f})\n"
        f"green=consistent(topology)  red=specular",
        fontsize=8)
    ax_cvar.axis("off")
    ax_cvar.text(0.05, 0.03, color_verdict, transform=ax_cvar.transAxes,
                 fontsize=8, color=color_verdict_c, fontweight="bold", va="bottom")

    # verdict (text rendered into ax_sum below)
    if miss_above_eps > 0.7:
        v1 = "✗ SDF DELETED / COLLAPSED"
        v1c = "#ff4444"
        v1d = "Most miss rays have sdf_min >> eps.\nThe zero level-set is simply absent\nin those regions — the SDF was pushed\npositive by the photometric loss."
    elif miss_smin_median < eps * 3:
        v1 = "? FLAT SDF NEAR ZERO"
        v1c = "#ffaa00"
        v1d = "sdf_min is near eps but rays still\nmiss. SDF has a very flat gradient\n(|∇f|≈0), making steps tiny and the\ntracer can't converge in budget."
    else:
        v1 = "? ITER BUDGET TOO SMALL"
        v1c = "#ffaa00"
        v1d = "sdf_min is moderate. There may be a\nzero crossing but the tracer can't\nreach it in the allotted iters.\nSee SDF profiles below."

    if grad_miss_med < 0.2:
        v2 = "✗ FLAT GRADIENT (no eikonal)"
        v2c = "#ff4444"
        v2d = "Median |∇f| on miss rays is very low.\nWithout eikonal loss the 1-Lip network\nlearns a nearly-constant SDF in unseen\nregions → tracer stalls."
    elif grad_miss_med > 0.8:
        v2 = "✓ gradient norm OK"
        v2c = "#00ff88"
        v2d = "|∇f| ≈ 1 on miss rays — SDF is well-\nconditioned there."
    else:
        v2 = "~ gradient norm weak"
        v2c = "#ffaa00"
        v2d = f"|∇f| ≈ {grad_miss_med:.2f} on miss rays."

    ax_sum.text(0.05, 0.0, "DIAGNOSIS", transform=ax_sum.transAxes,
                fontsize=10, fontweight="bold", va="bottom", color="white")
    ax_sum.text(0.05, -0.08, v1, transform=ax_sum.transAxes,
                fontsize=9, fontweight="bold", va="top", color=v1c)
    ax_sum.text(0.05, -0.22, v1d, transform=ax_sum.transAxes,
                fontsize=7, va="top", color="white", linespacing=1.4)
    ax_sum.text(0.05, -0.46, v2, transform=ax_sum.transAxes,
                fontsize=9, fontweight="bold", va="top", color=v2c)
    ax_sum.text(0.05, -0.56, v2d, transform=ax_sum.transAxes,
                fontsize=7, va="top", color="white", linespacing=1.4)

    # ── Rows 2–3: SDF profiles ────────────────────────────────────────────────
    if ts_arr is not None:
        ax_hp = fig.add_subplot(gs[2, :2], facecolor=BG)
        ax_hp.tick_params(labelsize=7)
        ax_hp.axhline(0, color="white", lw=0.7, alpha=0.5)
        ax_hp.axhline(cfg.trace.eps, color="cyan", lw=0.8, ls="--", alpha=0.8, label=f"+eps={eps}")
        ax_hp.axhline(-cfg.trace.eps, color="cyan", lw=0.8, ls="--", alpha=0.8)
        if hole_profiles is not None:
            cmap_h = plt.cm.Reds
            for k, prof in enumerate(hole_profiles):
                col = cmap_h(0.4 + 0.5 * k / max(len(hole_profiles) - 1, 1))
                ax_hp.plot(ts_arr, prof, color=col, lw=0.9, alpha=0.85,
                           label=f"hole#{k}" if k < 4 else None)
                # mark any zero crossings
                sign_changes = np.where(np.diff(np.sign(prof)))[0]
                for sc in sign_changes:
                    ax_hp.axvline(ts_arr[sc], color=col, lw=0.5, ls=":", alpha=0.6)
        n_crossings = 0
        if hole_profiles is not None:
            n_crossings = sum(1 for p in hole_profiles if np.any(np.diff(np.sign(p)) != 0))
        ax_hp.set_xlabel("t (ray distance)", fontsize=8)
        ax_hp.set_ylabel("f(o + t·d)", fontsize=8)
        ax_hp.set_title(
            f"SDF profiles: HOLE rays  "
            f"({n_crossings}/{len(hole_profiles) if hole_profiles is not None else 0} have zero crossings — "
            f"vertical dots = crossing location)", fontsize=9)
        ax_hp.legend(fontsize=6, framealpha=0.2, ncol=3)
        ax_hp.set_ylim(-cfg.trace.t_far * 0.3, cfg.trace.t_far * 0.3)

        ax_hitp = fig.add_subplot(gs[2, 2:], facecolor=BG)
        ax_hitp.tick_params(labelsize=7)
        ax_hitp.axhline(0, color="white", lw=0.7, alpha=0.5)
        ax_hitp.axhline(cfg.trace.eps,  color="cyan", lw=0.8, ls="--", alpha=0.8, label=f"+eps")
        ax_hitp.axhline(-cfg.trace.eps, color="cyan", lw=0.8, ls="--", alpha=0.8)
        if hit_profiles is not None:
            cmap_g = plt.cm.Greens
            for k, prof in enumerate(hit_profiles):
                col = cmap_g(0.4 + 0.5 * k / max(len(hit_profiles) - 1, 1))
                ax_hitp.plot(ts_arr2, prof, color=col, lw=0.9, alpha=0.85,
                             label=f"hit#{k}")
        ax_hitp.set_xlabel("t (ray distance)", fontsize=8)
        ax_hitp.set_ylabel("f(o + t·d)", fontsize=8)
        ax_hitp.set_title("SDF profiles: HIT rays (reference)", fontsize=9)
        ax_hitp.legend(fontsize=6, framealpha=0.2)
        ax_hitp.set_ylim(-cfg.trace.t_far * 0.3, cfg.trace.t_far * 0.3)

        # row 3: zoomed SDF profiles for hole rays near t=0 to t_far/2
        if hole_profiles is not None and n_rows >= 4:
            ax_zoom = fig.add_subplot(gs[3, :], facecolor=BG)
            ax_zoom.tick_params(labelsize=7)
            ax_zoom.axhline(0,   color="white", lw=0.8, alpha=0.5)
            ax_zoom.axhline(cfg.trace.eps,  color="cyan", lw=1.0, ls="--", alpha=0.9, label=f"±eps={eps}")
            ax_zoom.axhline(-cfg.trace.eps, color="cyan", lw=1.0, ls="--", alpha=0.9)
            # show min-sdf per hole ray and mark
            for k, (prof, idx) in enumerate(zip(hole_profiles, chosen_hole)):
                col = plt.cm.Reds(0.4 + 0.5 * k / max(len(hole_profiles) - 1, 1))
                ax_zoom.plot(ts_arr, prof, color=col, lw=1.2, alpha=0.9)
                t_min_idx = np.argmin(np.abs(prof))
                ax_zoom.plot(ts_arr[t_min_idx], prof[t_min_idx], "o", color=col,
                             ms=4, label=f"ray{k} sdf_min={abs(prof[t_min_idx]):.4f}")
            ax_zoom.set_xlabel("t", fontsize=8)
            ax_zoom.set_ylabel("f(o + t·d)", fontsize=8)
            ax_zoom.set_title("SDF profiles HOLE rays — marker = minimum |f| along ray", fontsize=9)
            ylim = max(0.5, float(np.abs(hole_profiles).max() * 0.5))
            ax_zoom.set_ylim(-ylim, ylim)
            ax_zoom.legend(fontsize=6, framealpha=0.2, ncol=4)

    plt.style.use("default")
    fig.suptitle(
        f"Hole diagnostic — {args.run_dir.name}  view={vi}  "
        f"iters={cfg.trace.iters}  eps={cfg.trace.eps}  w_eikonal={cfg.train.w_eikonal}",
        fontsize=12, y=1.002, color="white")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[diagnose] saved → {out}")

    # print textual summary
    print("\n=== DIAGNOSIS SUMMARY ===")
    print(f"  hit rate:              {hit_frac:.1%}")
    print(f"  miss sdf_min median:   {miss_smin_median:.5f}  (eps={eps})")
    print(f"  miss sdf_min > eps:    {miss_above_eps:.1%}  of all miss rays")
    print(f"  median |∇f| on hits:   {grad_hit_med:.3f}")
    print(f"  median |∇f| on misses: {grad_miss_med:.3f}")
    if hole_profiles is not None:
        nc = sum(1 for p in hole_profiles if np.any(np.diff(np.sign(p)) != 0))
        print(f"  zero crossings in sampled hole rays: {nc}/{len(hole_profiles)}")
        print(f"  → {'surface EXISTS but tracer misses it' if nc > len(hole_profiles) // 2 else 'surface ABSENT along hole rays'}")
    print(f"  colour std at holes:   {mean_hole_std:.3f}  ({args.color_views} views, proxy=sdf_min depth)")
    print(f"  → {color_verdict}")


if __name__ == "__main__":
    main()
