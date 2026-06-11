#!/usr/bin/env python3
"""Per-pixel sphere tracing debugger.

Fires a ray through user-specified pixels and records SDF(t) at every iteration,
then plots the full SDF profile along the ray plus the traced path.

Usage examples:
    # pixels given as "u,v" (column, row) in the *full-resolution* image
    python debug_pixel_trace.py \\
        outputs/run_xxx/checkpoint_final.pt \\
        --view 0 \\
        --pixels "320,240" "400,210" "380,280" \\
        --max-iters 64 \\
        --n-profile 512

    # or read pixel coords from a file (one "u,v" per line)
    python debug_pixel_trace.py outputs/run_xxx/checkpoint_final.pt \\
        --view 0 --pixels-file my_pixels.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.model import make_model
from lip_tracer.config import TraceConfig
import lip_tracer.data as data_mod


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
})


# ── Model / checkpoint loading ────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # try loading config.json next to the checkpoint first
    cfg_path = ckpt_path.parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        m = cfg["model"]
        f = make_model(
            hidden=m["hidden"], depth=m["depth"],
            group_size=m.get("group_size", 2),
            activation=m.get("activation", "groupsort"),
            input_encoding=m.get("input_encoding", "pe"),
            multires=m.get("multires", 6),
            architecture=m.get("architecture", "cpl"),
        )
        trace_cfg = TraceConfig(**cfg["trace"])
        scene = Path(cfg["scene"])
    else:
        # fall back to metadata stored in the checkpoint itself
        architecture    = ckpt.get("architecture", "cpl")
        group_size      = ckpt.get("group_size", 2)
        activation      = ckpt.get("activation", "groupsort")
        input_encoding  = ckpt.get("input_encoding", "pe")
        multires        = ckpt.get("multires", 6)
        state = ckpt.get("f", ckpt)
        hidden = next(
            v.shape[1] for k, v in state.items()
            if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
            and not k.startswith("encoder")
        )
        depth = ckpt.get("depth", sum(
            1 for k in state if k.startswith("net.") and k.endswith(".weight")
        ))
        f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                       activation=activation, input_encoding=input_encoding,
                       multires=multires, architecture=architecture)
        trace_cfg = TraceConfig()
        scene = Path(ckpt.get("scene", data_mod.SCENE if hasattr(data_mod, "SCENE") else "."))

    model_state = ckpt.get("f", ckpt.get("model", ckpt))
    f.load_state_dict(model_state, strict=False)
    f.eval().to(device)
    print(f"  loaded: {ckpt_path.name}  scene={scene.name}  "
          f"hidden={f.hidden}  enc={f.input_encoding}")
    return f, trace_cfg, scene


# ── Ray from pixel ────────────────────────────────────────────────────────────

def pixel_ray(c2w: torch.Tensor, K: torch.Tensor, u: float, v: float):
    """Return (origin, direction) for pixel (u=col, v=row) in full-res image."""
    # normalised camera-space direction
    dir_cam = torch.tensor([
        (u - K[0, 2].item()) / K[0, 0].item(),
        (v - K[1, 2].item()) / K[1, 1].item(),
        1.0,
    ], dtype=torch.float32)
    dir_world = c2w[:3, :3] @ dir_cam
    dir_world = dir_world / dir_world.norm()
    origin = c2w[:3, 3]
    return origin, dir_world


# ── Verbose single-ray sphere trace ──────────────────────────────────────────

@torch.no_grad()
def trace_verbose(f, origin: torch.Tensor, direction: torch.Tensor,
                  cfg: TraceConfig, max_iters: int | None = None):
    """Sphere-trace one ray and return per-iteration state.

    Returns
    -------
    records : list[dict]  — one dict per iteration: t, sdf, x (3,), active
    hit     : bool
    t_final : float
    """
    n_iters = max_iters or cfg.iters
    o = origin.unsqueeze(0)   # (1,3)
    d = direction.unsqueeze(0)

    t         = torch.zeros(1)
    converged = torch.zeros(1, dtype=torch.bool)
    records   = []

    for i in range(n_iters):
        escaped = (t >= cfg.t_far)
        active  = ~(converged | escaped)
        x = o + t.unsqueeze(-1) * d
        sdf = f(x).squeeze()
        records.append({
            "iter":   i,
            "t":      t.item(),
            "sdf":    sdf.item(),
            "x":      x.squeeze().tolist(),
            "active": active.item(),
        })
        just_conv = active & (sdf.abs() < cfg.eps)
        converged = converged | just_conv
        step = torch.where(converged | escaped, torch.zeros_like(sdf.unsqueeze(0)), sdf.unsqueeze(0))
        t = t + step

        if converged.all() or escaped.all():
            break

    # Newton refinement (same as tracer)
    hit = converged.item() and t.item() < cfg.t_far and t.item() >= 0
    if hit:
        eps_fd = 1e-3
        for _ in range(cfg.newton_steps):
            xp = o + t.unsqueeze(-1) * d
            fv = f(xp).squeeze()
            dd = (f(xp + eps_fd * d).squeeze() - fv) / eps_fd
            dd = dd.abs().clamp(min=1e-6)
            t = t - fv / dd

    return records, hit, t.item()


# ── SDF profile along the ray ─────────────────────────────────────────────────

@torch.no_grad()
def sdf_profile(f, origin: torch.Tensor, direction: torch.Tensor,
                t_near: float, t_far: float, n: int = 512):
    ts = torch.linspace(t_near, t_far, n)
    xs = origin.unsqueeze(0) + ts.unsqueeze(-1) * direction.unsqueeze(0)  # (n,3)
    chunks = []
    for i in range(0, n, 1024):
        chunks.append(f(xs[i:i+1024]).cpu())
    return ts.numpy(), torch.cat(chunks).numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pixel(ax_profile, ax_trace, ax_img,
               ts_dense, sdfs_dense,
               records, hit, t_final,
               img_crop, pixel_label: str, cfg: TraceConfig):
    """Fill three axes for one pixel."""

    # ── SDF profile ───────────────────────────────────────────────────────────
    ax_profile.axhline(0, color="k", lw=0.6, ls="--")
    ax_profile.axhline( cfg.eps, color="gray", lw=0.5, ls=":")
    ax_profile.axhline(-cfg.eps, color="gray", lw=0.5, ls=":")
    ax_profile.plot(ts_dense, sdfs_dense, lw=1.0, color="#2166ac", label="SDF(t)")

    # mark trace steps
    t_vals  = [r["t"]   for r in records]
    sdf_vals = [r["sdf"] for r in records]
    colors  = ["green" if hit and i == len(records) - 1 else
               ("#e66101" if not r["active"] else "#4dac26")
               for i, r in enumerate(records)]
    ax_profile.scatter(t_vals, sdf_vals, c=colors, s=20, zorder=5)

    # mark final hit
    if hit:
        ax_profile.axvline(t_final, color="green", lw=0.8, ls="--", alpha=0.7)
        ax_profile.scatter([t_final], [0], marker="*", s=80, color="green", zorder=6)

    ax_profile.set_xlabel("t  (ray distance)")
    ax_profile.set_ylabel("SDF value")
    ax_profile.set_title(f"SDF profile  pix={pixel_label}  hit={'YES' if hit else 'NO'}")
    ax_profile.set_xlim(ts_dense[0], ts_dense[-1])
    ax_profile.legend(fontsize=7)

    # ── Trace path: step-size per iteration ──────────────────────────────────
    steps = [records[i+1]["t"] - records[i]["t"] for i in range(len(records)-1)]
    ax_trace.bar(range(len(steps)), steps, color="#4dac26", width=0.8)
    ax_trace.axhline(cfg.eps, color="gray", lw=0.5, ls=":")
    ax_trace.set_xlabel("Iteration")
    ax_trace.set_ylabel("Step size (Δt)")
    ax_trace.set_title(f"Step sizes  ({len(records)} iters)")

    # ── Image crop ────────────────────────────────────────────────────────────
    ax_img.imshow(img_crop.clip(0, 1))
    ax_img.axis("off")
    ax_img.set_title("Crop (10 px)")


def print_table(records, hit, t_final, pixel_label, cfg):
    print(f"\n── pixel {pixel_label}  hit={'YES' if hit else 'NO'}  t_final={t_final:.4f} ──")
    print(f"  {'iter':>4}  {'t':>8}  {'sdf':>10}  {'|sdf|<eps':>10}  {'active':>6}")
    for r in records:
        flag = "<-- HIT" if abs(r["sdf"]) < cfg.eps and r["active"] else ""
        print(f"  {r['iter']:>4}  {r['t']:>8.4f}  {r['sdf']:>10.5f}  "
              f"  {str(abs(r['sdf'])<cfg.eps):>9}  {str(r['active']):>6}  {flag}")
    if hit:
        print(f"  Newton-refined t = {t_final:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="Path to checkpoint .pt")
    ap.add_argument("--view", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--pixels", nargs="+", metavar="U,V",
                    help='Pixel coords as "col,row" e.g. "320,240" (full-res)')
    ap.add_argument("--pixels-file", type=Path,
                    help="File with one 'u,v' per line")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="Override trace iterations (default: from config)")
    ap.add_argument("--n-profile", type=int, default=512,
                    help="Samples for dense SDF profile along ray (default 512)")
    ap.add_argument("--t-near", type=float, default=0.0,
                    help="Profile start (default 0.0)")
    ap.add_argument("--t-far",  type=float, default=None,
                    help="Profile end (default: from trace config)")
    ap.add_argument("--crop-half", type=int, default=40,
                    help="Half-size of image crop around pixel (px, default 40)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: next to ckpt)")
    ap.add_argument("--blender", action="store_true")
    ap.add_argument("--scene", type=Path, default=None,
                    help="Override scene path from config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ── load ──────────────────────────────────────────────────────────────────
    f, trace_cfg, scene = load_checkpoint(args.ckpt, args.device)
    if args.max_iters is not None:
        import dataclasses
        trace_cfg = dataclasses.replace(trace_cfg, iters=args.max_iters)
    t_far_plot = args.t_far or trace_cfg.t_far

    scene_path = args.scene or scene
    if args.blender:
        views = data_mod.load_blender_views(scene=scene_path)
    else:
        views = data_mod.load_views(scene_path)

    H, W = views["H"], views["W"]
    c2w  = views["c2w"][args.view]   # (4,4)
    K    = views["K"][args.view]     # (3,3)
    img  = views["images"][args.view].numpy()   # (H, W, 3)
    print(f"  view {args.view}  image {H}×{W}")

    # ── parse pixels ──────────────────────────────────────────────────────────
    pixel_strs = list(args.pixels or [])
    if args.pixels_file:
        pixel_strs += [l.strip() for l in args.pixels_file.read_text().splitlines()
                       if l.strip() and not l.startswith("#")]
    if not pixel_strs:
        ap.error("Provide at least one pixel with --pixels or --pixels-file")

    pixels = []
    for s in pixel_strs:
        u, v = map(float, s.split(","))
        pixels.append((u, v))
    print(f"  tracing {len(pixels)} pixel(s): {pixels}")

    # ── trace + plot ──────────────────────────────────────────────────────────
    n = len(pixels)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.5 * n),
                              gridspec_kw={"wspace": 0.3, "hspace": 0.45})
    if n == 1:
        axes = axes[np.newaxis, :]   # keep 2-D indexing

    for row, (u, v) in enumerate(pixels):
        origin, direction = pixel_ray(c2w, K, u, v)
        origin    = origin.to(args.device)
        direction = direction.to(args.device)

        # SDF profile (dense)
        ts_dense, sdfs_dense = sdf_profile(
            f, origin, direction,
            t_near=args.t_near, t_far=t_far_plot, n=args.n_profile,
        )

        # verbose trace
        records, hit, t_final = trace_verbose(f, origin, direction, trace_cfg,
                                              max_iters=args.max_iters)
        print_table(records, hit, t_final, f"{u:.0f},{v:.0f}", trace_cfg)

        # image crop around the pixel
        c_half = args.crop_half
        v_i, u_i = int(round(v)), int(round(u))
        r0, r1 = max(0, v_i - c_half), min(H, v_i + c_half)
        c0, c1 = max(0, u_i - c_half), min(W, u_i + c_half)
        crop = img[r0:r1, c0:c1].copy()
        # mark pixel with a red dot
        py = v_i - r0; px = u_i - c0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ry, rx = py + dy, px + dx
                if 0 <= ry < crop.shape[0] and 0 <= rx < crop.shape[1]:
                    crop[ry, rx] = [1.0, 0.0, 0.0]

        plot_pixel(
            axes[row, 0], axes[row, 1], axes[row, 2],
            ts_dense, sdfs_dense,
            records, hit, t_final,
            crop, f"{u:.0f},{v:.0f}", trace_cfg,
        )

    out = args.out or (args.ckpt.parent / f"pixel_trace_view{args.view}.png")
    fig.suptitle(
        f"{args.ckpt.parent.name}  view={args.view}  iters={trace_cfg.iters}  eps={trace_cfg.eps}",
        fontsize=10,
    )
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
