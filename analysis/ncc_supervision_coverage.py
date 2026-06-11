#!/usr/bin/env python3
"""Render per-pixel NCC supervision coverage for one checkpoint and reference view.

This is an evaluation-only diagnostic. It mirrors the position-branch NCC gates
in lip_tracer.loss.photo_loss / pmvs_ncc_loss without changing training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lip_tracer.config import TraceConfig
from lip_tracer.data import load_views, precompute_alt_cameras
from lip_tracer.loss import pmvs_ncc_loss
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd


MAP_NAMES = (
    "n_center_current",
    "n_patch_current",
    "n_kept_current",
    "n_center_all",
    "n_patch_all",
    "n_kept_all",
    "rho_max_all",
    "gain_patch",
)


def infer_config_path(ckpt: Path) -> Path:
    """Find the run config above a checkpoint path."""
    for parent in (ckpt.parent, *ckpt.parents):
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not infer config.json above {ckpt}; pass --config explicitly")


def scheduled_value(start: float, end: float, step: int, steps: int) -> float:
    if start > 0 and end > 0 and end != start:
        t = step / max(steps - 1, 1)
        return start * (end / start) ** t
    return start


def trace_chunked(f, o, d, trace_cfg, chunk):
    xs, ts, hits = [], [], []
    for begin in range(0, o.shape[0], chunk):
        end = begin + chunk
        x, t, hit = trace_nograd(f, o[begin:end], d[begin:end], trace_cfg)
        xs.append(x)
        ts.append(t)
        hits.append(hit)
    return torch.cat(xs), torch.cat(ts), torch.cat(hits)


def grad_normals(f, x, chunk):
    """Compute unit input gradients locally; model parameters stay frozen."""
    normals = []
    for begin in range(0, x.shape[0], chunk):
        xc = x[begin:begin + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            grad = torch.autograd.grad(f(xc).sum(), xc)[0].detach()
        normals.append(grad)
    grad = torch.cat(normals)
    return F.normalize(grad, dim=-1)


def patch_validity_and_rho(
    x3d, normals, vi_a, vi_b, K_all, w2c_all, H, W, patch, half_pix,
):
    """Mirror pmvs_ncc_loss tangent-grid construction and patch bounds test."""
    P = patch
    n = F.normalize(normals, dim=-1)
    up = n.new_zeros(n.shape[0], 3)
    up[:, 1] = 1.0
    swap = n[:, 1].abs() > 0.9
    up[swap, 1] = 0.0
    up[swap, 0] = 1.0
    t1 = F.normalize(torch.cross(n, up, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)

    R_a = w2c_all[vi_a, :3, :3]
    t_a = w2c_all[vi_a, :3, 3]
    xc_a = torch.einsum("bij,bj->bi", R_a, x3d) + t_a
    z_ref = xc_a[:, 2].clamp(min=1e-3)
    f_x = K_all[vi_a, 0, 0]
    step_3d = ((2.0 * half_pix / max(P - 1, 1)) * z_ref / f_x).detach()

    offs = torch.linspace(-(P - 1) / 2, (P - 1) / 2, P, device=x3d.device)
    oi, oj = torch.meshgrid(offs, offs, indexing="ij")
    oi, oj = oi.reshape(-1), oj.reshape(-1)
    pts3d = (
        x3d.unsqueeze(1)
        + step_3d[:, None, None]
        * (oi[None, :, None] * t1.unsqueeze(1)
           + oj[None, :, None] * t2.unsqueeze(1))
    )

    def project(vi):
        R = w2c_all[vi, :3, :3]
        t_v = w2c_all[vi, :3, 3]
        xc = (R.unsqueeze(1) @ pts3d.unsqueeze(-1)).squeeze(-1) + t_v.unsqueeze(1)
        ph = (K_all[vi].unsqueeze(1) @ xc.unsqueeze(-1)).squeeze(-1)
        uv = ph[:, :, :2] / ph[:, :, 2:3].clamp(min=1e-6)
        inside = (
            (xc[:, :, 2] > 0)
            & (uv[:, :, 0] >= 0) & (uv[:, :, 0] < W)
            & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < H)
        )
        return inside

    inside_a = project(vi_a)
    inside_b = project(vi_b)
    joint_inside = inside_a & inside_b
    return joint_inside.all(1), joint_inside.float().mean(1)


class CoverageEvaluator:
    def __init__(self, f, images, masks, K_all, c2w_all, trace_cfg, train_cfg,
                 ref_view, H, W, pair_chunk):
        self.f = f
        self.images = images
        self.masks = masks
        self.K_all = K_all
        self.c2w_all = c2w_all
        self.w2c_all = torch.linalg.inv(c2w_all)
        self.origins = c2w_all[:, :3, 3]
        self.trace_cfg = trace_cfg
        self.tcfg = train_cfg
        self.ref_view = ref_view
        self.H, self.W = H, W
        self.pair_chunk = pair_chunk

    def _center_gate(self, x, n, vi_b):
        op = self.origins[vi_b]
        direction = op - x
        dist = direction.norm(dim=-1).clamp(min=1e-6)
        dp = direction / dist.unsqueeze(-1)
        if self.tcfg.get("occ_mode", "pinhole") == "from_hit":
            _, tp, hitp = trace_chunked(
                self.f, x + 1e-2 * dp, dp, self.trace_cfg, self.pair_chunk)
            not_occl = ~hitp | (tp > dist - 0.1)
        else:
            _, tp, hitp = trace_chunked(
                self.f, op, -dp, self.trace_cfg, self.pair_chunk)
            not_occl = hitp & (dist <= tp + 1e-1)

        w2c = self.w2c_all[vi_b]
        xc = torch.einsum("bij,bj->bi", w2c[:, :3, :3], x) + w2c[:, :3, 3]
        uv_h = torch.einsum("bij,bj->bi", self.K_all[vi_b], xc)
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
        in_frame = (
            (xc[:, 2] > 0)
            & (uv[:, 0] >= 0) & (uv[:, 0] < self.W)
            & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
        )
        cos_ok = (n * dp).sum(-1).abs() > self.tcfg.get("cos_thresh", 0.1)
        if self.masks is None:
            fg_alt = torch.ones_like(in_frame)
        else:
            uv_c = uv.long().clamp(0).clone()
            uv_c[:, 0].clamp_(max=self.W - 1)
            uv_c[:, 1].clamp_(max=self.H - 1)
            fg_alt = self.masks[vi_b, uv_c[:, 1], uv_c[:, 0]]
        return in_frame & not_occl & cos_ok & fg_alt

    def _ncc_kept(self, x, n, vi_b):
        kept = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for begin in range(0, x.shape[0], self.pair_chunk):
            end = begin + self.pair_chunk
            xb, nb, vb = x[begin:end], n[begin:end], vi_b[begin:end]
            va = torch.full_like(vb, self.ref_view)
            _, _, _, zncc_full = pmvs_ncc_loss(
                self.images, xb, nb, va, vb, self.K_all, self.w2c_all,
                self.H, self.W,
                self.tcfg["ncc_patch"], self.tcfg["ncc_half_pix"],
                self.tcfg.get("sample_mode", "bilinear"),
                self.tcfg["_effective_sigma"], self.tcfg["_effective_radius"],
                self.tcfg.get("ncc_min", 0.4),
                return_full=True,
                ncc_color=self.tcfg.get("ncc_color", "gray"),
                ncc_grad_alpha=self.tcfg.get("ncc_grad_alpha", 0.0),
                patch_wsigma=self.tcfg["_effective_wsigma"],
                patch_bilateral_gamma=self.tcfg["_effective_bgamma"],
            )
            kept[begin:end] = (
                torch.isfinite(zncc_full)
                & (zncc_full > self.tcfg.get("ncc_min", 0.4))
            )
        return kept

    @torch.no_grad()
    def evaluate(self, x, n, cameras, camera_chunk, need_rho):
        B = x.shape[0]
        n_center = torch.zeros(B, dtype=torch.int32, device=x.device)
        n_patch = torch.zeros_like(n_center)
        n_kept = torch.zeros_like(n_center)
        rho_max = torch.zeros(B, dtype=torch.float32, device=x.device)

        for cam_begin in range(0, len(cameras), camera_chunk):
            cams = cameras[cam_begin:cam_begin + camera_chunk]
            C = cams.shape[0]
            xp = x[:, None, :].expand(B, C, 3).reshape(-1, 3)
            np_ = n[:, None, :].expand(B, C, 3).reshape(-1, 3)
            vb = cams[None, :].expand(B, C).reshape(-1)
            center = torch.zeros(xp.shape[0], dtype=torch.bool, device=x.device)
            patch = torch.zeros_like(center)
            rho = torch.zeros(xp.shape[0], dtype=torch.float32, device=x.device)

            for begin in range(0, xp.shape[0], self.pair_chunk):
                end = begin + self.pair_chunk
                center[begin:end] = self._center_gate(
                    xp[begin:end], np_[begin:end], vb[begin:end])
                va = torch.full_like(vb[begin:end], self.ref_view)
                patch[begin:end], rho[begin:end] = patch_validity_and_rho(
                    xp[begin:end], np_[begin:end], va, vb[begin:end],
                    self.K_all, self.w2c_all, self.H, self.W,
                    self.tcfg["ncc_patch"], self.tcfg["ncc_half_pix"])

            kept = torch.zeros_like(center)
            gated = center.nonzero(as_tuple=True)[0]
            if gated.numel():
                kept[gated] = self._ncc_kept(xp[gated], np_[gated], vb[gated])

            n_center += center.reshape(B, C).sum(1)
            n_patch += (center & patch).reshape(B, C).sum(1)
            n_kept += (center & kept).reshape(B, C).sum(1)
            if need_rho:
                rho_max = torch.maximum(rho_max, rho.reshape(B, C).max(1).values)
        return n_center, n_patch, n_kept, rho_max


def save_heatmap(path, values, foreground, title, vmax=None, cmap="viridis"):
    shown = values.astype(np.float32).copy()
    shown[~foreground] = np.nan
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path, help="checkpoint path")
    ap.add_argument("--scene", required=True, type=Path, help="scan/data directory")
    ap.add_argument("--view", required=True, type=int, help="reference view index")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--config", type=Path, default=None,
                    help="run config JSON (default: infer above checkpoint)")
    ap.add_argument("--down", type=int, default=1,
                    help="optional diagnostic image downsampling factor (default: 1)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ray-chunk", type=int, default=8192)
    ap.add_argument("--normal-chunk", type=int, default=4096)
    ap.add_argument("--camera-chunk", type=int, default=4)
    ap.add_argument("--pair-chunk", type=int, default=2048)
    args = ap.parse_args()

    config_path = args.config or infer_config_path(args.ckpt)
    cfg = json.loads(config_path.read_text())
    tcfg = dict(cfg["train"])
    step_count = int(tcfg.get("steps", 300000))

    sd = torch.load(args.ckpt, map_location="cpu")
    step = int(sd.get("step", -1)) if isinstance(sd, dict) else -1
    tcfg["_effective_sigma"] = scheduled_value(
        tcfg.get("gaussian_sigma", 2.0),
        tcfg.get("gaussian_sigma_end", tcfg.get("gaussian_sigma", 2.0)),
        step, step_count,
    ) if tcfg.get("sample_mode", "bilinear") == "gaussian" else tcfg.get("gaussian_sigma", 2.0)
    tcfg["_effective_radius"] = max(1, int(math.ceil(2.0 * tcfg["_effective_sigma"])))
    tcfg["_effective_wsigma"] = scheduled_value(
        tcfg.get("ncc_patch_wsigma", 0.0),
        tcfg.get("ncc_patch_wsigma_end", 0.0), step, step_count)
    tcfg["_effective_bgamma"] = scheduled_value(
        tcfg.get("ncc_bilateral_gamma", 0.0),
        tcfg.get("ncc_bilateral_gamma_end", 0.0), step, step_count)

    dev = torch.device(args.device)
    f = make_model(**cfg["model"]).to(dev).eval()
    f.load_state_dict(sd["f"] if isinstance(sd, dict) and "f" in sd else sd)
    for parameter in f.parameters():
        parameter.requires_grad_(False)

    down = args.down
    views = load_views(args.scene, down=down)
    H, W = int(views["H"]), int(views["W"])
    images = views["images"].float().to(dev)
    masks = views.get("masks")
    masks = masks.to(dev) if masks is not None else None
    K_all = views["K"].float().to(dev)
    c2w_all = views["c2w"].float().to(dev)
    V = int(images.shape[0])
    if not 0 <= args.view < V:
        raise ValueError(f"--view must be in [0, {V - 1}], got {args.view}")

    args.out.mkdir(parents=True, exist_ok=True)
    rgb = (images[args.view].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(args.out / "reference_rgb.png")

    yy, xx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    K = K_all[args.view]
    d_cam = torch.stack(((xx - K[0, 2]) / K[0, 0],
                         (yy - K[1, 2]) / K[1, 1],
                         torch.ones_like(xx)), dim=-1)
    c2w = c2w_all[args.view]
    d = d_cam.reshape(-1, 3) @ c2w[:3, :3].T
    d = F.normalize(d, dim=-1)
    o = c2w[:3, 3].expand_as(d).contiguous()

    trace_cfg = TraceConfig(**cfg["trace"])
    print(f"[trace] view={args.view} shape={H}x{W} down={down} rays={H * W}")
    with torch.no_grad():
        x_all, _, hit = trace_chunked(f, o, d, trace_cfg, args.ray_chunk)
    fg_self = masks[args.view].reshape(-1) if masks is not None else torch.ones_like(hit)
    active = hit & fg_self
    active_idx = active.nonzero(as_tuple=True)[0]
    print(f"[trace] foreground hits={active_idx.numel()} / {H * W}")
    if active_idx.numel() == 0:
        raise RuntimeError("reference view has no sphere-traced foreground hits")
    x = x_all[active_idx]
    n = grad_normals(f, x, args.normal_chunk)

    evaluator = CoverageEvaluator(
        f, images, masks, K_all, c2w_all, trace_cfg, tcfg,
        args.view, H, W, args.pair_chunk)
    alt_nn = precompute_alt_cameras(views, int(tcfg["n_alt"])).to(dev)
    current = alt_nn[args.view]
    all_cameras = torch.tensor(
        [vi for vi in range(V) if vi != args.view], device=dev, dtype=torch.long)
    print(f"[coverage] current={current.tolist()} all={len(all_cameras)} cameras")
    current_maps = evaluator.evaluate(x, n, current, args.camera_chunk, need_rho=False)
    all_maps = evaluator.evaluate(x, n, all_cameras, args.camera_chunk, need_rho=True)

    arrays = {}
    flat = {
        "n_center_current": current_maps[0],
        "n_patch_current": current_maps[1],
        "n_kept_current": current_maps[2],
        "n_center_all": all_maps[0],
        "n_patch_all": all_maps[1],
        "n_kept_all": all_maps[2],
        "rho_max_all": all_maps[3],
    }
    for name, values in flat.items():
        dtype = np.float32 if name == "rho_max_all" else np.int16
        image = np.zeros(H * W, dtype=dtype)
        image[active_idx.cpu().numpy()] = values.cpu().numpy()
        arrays[name] = image.reshape(H, W)
    arrays["gain_patch"] = arrays["n_patch_all"] - arrays["n_patch_current"]
    foreground = active.reshape(H, W).cpu().numpy()
    arrays["foreground_hit"] = foreground

    for name, values in arrays.items():
        np.save(args.out / f"{name}.npy", values)
    for name in MAP_NAMES:
        vmax = 1.0 if name == "rho_max_all" else None
        save_heatmap(args.out / f"{name}.png", arrays[name], foreground, name, vmax=vmax)

    metadata = {
        "checkpoint": str(args.ckpt),
        "config": str(config_path),
        "scene": str(args.scene),
        "checkpoint_step": step,
        "reference_view": args.view,
        "down": down,
        "shape": [H, W],
        "foreground_hits": int(active_idx.numel()),
        "current_cameras": current.tolist(),
        "all_cameras": all_cameras.tolist(),
        "ncc_patch": tcfg["ncc_patch"],
        "ncc_half_pix": tcfg["ncc_half_pix"],
        "ncc_min": tcfg.get("ncc_min", 0.4),
        "effective_sigma": tcfg["_effective_sigma"],
        "effective_radius": tcfg["_effective_radius"],
        "effective_wsigma": tcfg["_effective_wsigma"],
        "effective_bgamma": tcfg["_effective_bgamma"],
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[done] wrote heatmaps and arrays to {args.out}")


if __name__ == "__main__":
    main()
