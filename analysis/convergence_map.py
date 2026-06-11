#!/usr/bin/env python3
"""Convergence map: visualize where sphere tracing fails and at which iteration it converges.

Usage:
    python convergence_map.py outputs/run_20260510_094943_scan65/checkpoint_final.pt
    python convergence_map.py outputs/run_20260510_094943_scan65/checkpoint_final.pt --view 10 --down 4
    python convergence_map.py outputs/run_20260510_094943_scan65/checkpoint_final.pt --max-iters 64 --down 4
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.model import make_model
from lip_tracer.config import TraceConfig
import lip_tracer.data as data_mod


# ── Figure style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})


# ── Model loading ─────────────────────────────────────────────────────────────

def load_run(ckpt_path: Path):
    cfg = json.loads((ckpt_path.parent / "config.json").read_text())
    m = cfg["model"]
    f = make_model(
        hidden=m["hidden"], depth=m["depth"],
        group_size=m.get("group_size", 2),
        activation=m.get("activation", "groupsort"),
        input_encoding=m.get("input_encoding", "pe"),
        multires=m.get("multires", 6),
        architecture=m.get("architecture", "cpl"),
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state.get("f", state.get("model", state))
    f.load_state_dict(model_state)
    f.eval()
    trace_cfg = TraceConfig(**cfg["trace"])
    return f, trace_cfg, Path(cfg["scene"])


# ── Ray casting ───────────────────────────────────────────────────────────────

def cast_rays(c2w: torch.Tensor, K: torch.Tensor, H: int, W: int, down: int = 1):
    H2, W2 = H // down, W // down
    ys, xs = torch.meshgrid(
        torch.arange(H2, dtype=torch.float32) * down + down / 2,
        torch.arange(W2, dtype=torch.float32) * down + down / 2,
        indexing="ij",
    )
    uv1 = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(H2 * W2)], dim=-1)
    dirs_cam = uv1 @ torch.linalg.inv(K).T
    dirs_world = dirs_cam @ c2w[:3, :3].T
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True)
    origins = c2w[:3, 3].unsqueeze(0).expand(H2 * W2, -1)
    return origins, dirs_world, H2, W2


# ── Sphere tracing with iteration tracking ────────────────────────────────────

@torch.no_grad()
def trace_with_iter_count(
    f, origins: torch.Tensor, dirs: torch.Tensor,
    cfg: TraceConfig, chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        hit       : (N,) bool
        iter_conv : (N,) int   — iteration at which ray converged; cfg.iters if never
        sdf_min   : (N,) float — minimum |SDF| seen along the ray
    """
    N = origins.shape[0]
    hit_all      = torch.zeros(N, dtype=torch.bool)
    iter_all     = torch.full((N,), cfg.iters, dtype=torch.long)
    sdf_min_all  = torch.full((N,), float("inf"))

    for start in range(0, N, chunk):
        o, d = origins[start:start + chunk], dirs[start:start + chunk]
        B = o.shape[0]

        t         = torch.zeros(B)
        converged = torch.zeros(B, dtype=torch.bool)
        iter_conv = torch.full((B,), cfg.iters, dtype=torch.long)
        sdf_min   = torch.full((B,), float("inf"))

        for i in range(cfg.iters):
            escaped = t >= cfg.t_far
            active  = ~(converged | escaped)
            if not active.any():
                break
            x   = o + t.unsqueeze(-1) * d
            sdf = f(x)
            abs_sdf = sdf.abs()
            sdf_min = torch.where(active, torch.minimum(sdf_min, abs_sdf), sdf_min)

            just_conv = active & (abs_sdf < cfg.eps)
            iter_conv = torch.where(just_conv & (iter_conv == cfg.iters),
                                    torch.full_like(iter_conv, i), iter_conv)
            converged = converged | just_conv
            step      = torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
            t         = t + step

        hit_all[start:start + chunk]     = converged & (t < cfg.t_far)
        iter_all[start:start + chunk]    = iter_conv
        sdf_min_all[start:start + chunk] = sdf_min

        done = min(start + chunk, N)
        print(f"  {done:>{len(str(N))}}/{N} rays  "
              f"({100*done/N:.0f}%)  hit so far: "
              f"{100*hit_all[:done].float().mean():.1f}%", end="\r")

    print()
    return hit_all, iter_all, sdf_min_all


# ── Visualisation ─────────────────────────────────────────────────────────────

def make_figure(
    img: np.ndarray,          # (H, W, 3) uint8
    mask: np.ndarray,         # (H, W) bool — foreground
    hit_map: np.ndarray,      # (H, W) bool
    iter_map: np.ndarray,     # (H, W) int
    max_iters: int,
    view_idx: int,
    out_path: Path,
):
    H, W = hit_map.shape
    fg_miss = mask & ~hit_map   # foreground rays that failed

    # ── Per-pixel category ────────────────────────────────────────────────────
    # 0 = background (expected miss)
    # 1..max_iters = hit at that iteration
    # max_iters+1  = foreground miss
    cat = np.zeros((H, W), dtype=np.float32)
    cat[hit_map] = iter_map[hit_map].astype(np.float32)
    cat[fg_miss] = max_iters + 1

    # ── Colormap: viridis for convergence iters, crimson for miss ─────────────
    base = plt.cm.viridis(np.linspace(0.05, 0.95, max_iters + 1))
    miss_color = np.array([[0.85, 0.1, 0.1, 1.0]])   # crimson for fg miss
    bg_color   = np.array([[0.92, 0.92, 0.92, 1.0]])  # light grey for background
    all_colors = np.vstack([bg_color, base, miss_color])  # 0, 1..max+1, max+2
    # shift: cat=0→bg, cat=1..max→viridis, cat=max+1→crimson
    cmap = ListedColormap(all_colors)
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, max_iters + 2.5, 1.0), ncolors=len(all_colors)
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_fg   = mask.sum()
    n_hit  = hit_map[mask].sum()
    n_miss = fg_miss.sum()
    hit_iters = iter_map[hit_map].astype(float)

    stats = (
        f"FG hit rate: {100*n_hit/max(n_fg,1):.1f}%  |  "
        f"FG miss: {100*n_miss/max(n_fg,1):.1f}%  |  "
        f"Iter (p50/p90/max): "
        f"{np.percentile(hit_iters,50):.0f} / "
        f"{np.percentile(hit_iters,90):.0f} / "
        f"{hit_iters.max():.0f}"
    ) if len(hit_iters) > 0 else ""

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"wspace": 0.05})

    # Left: reference photo
    axes[0].imshow(img)
    axes[0].set_title(f"Input image (view {view_idx})", pad=6)
    axes[0].axis("off")

    # Right: convergence map
    im = axes[1].imshow(cat, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Sphere tracing convergence", pad=6)
    axes[1].axis("off")

    # Colorbar for convergence iterations (exclude bg and miss)
    sm = plt.cm.ScalarMappable(
        cmap=ListedColormap(base),
        norm=mcolors.Normalize(vmin=0, vmax=max_iters),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.035, pad=0.02,
                      ticks=[0, max_iters // 4, max_iters // 2,
                             3 * max_iters // 4, max_iters])
    cb.set_label("Convergence iteration", labelpad=6)

    # Legend
    legend_elements = [
        Patch(facecolor=bg_color[0], edgecolor="0.7", label="Background"),
        Patch(facecolor=miss_color[0], label=f"Foreground miss ({100*n_miss/max(n_fg,1):.1f}%)"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right",
                   fontsize=8, framealpha=0.85)

    fig.suptitle(stats, fontsize=9, y=0.02, va="bottom", color="0.3")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--view",      type=int, default=0,
                        help="View index to render (default: 0)")
    parser.add_argument("--down",      type=int, default=4,
                        help="Downscale factor for resolution (default: 4)")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Override trace iters (useful to test if more iters helps)")
    parser.add_argument("--out",       type=Path, default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} …")
    f, trace_cfg, scene = load_run(args.checkpoint)

    if args.max_iters is not None:
        print(f"Overriding trace iters: {trace_cfg.iters} → {args.max_iters}")
        trace_cfg = TraceConfig(
            iters=args.max_iters, eps=trace_cfg.eps, t_far=trace_cfg.t_far,
            eik_stride=trace_cfg.eik_stride, newton_steps=trace_cfg.newton_steps,
            grad_mode=trace_cfg.grad_mode,
        )

    print(f"Loading scene from {scene} …")
    data_mod.SCENE = scene
    views = data_mod.load_views(scene, down=1)

    V   = views["c2w"].shape[0]
    vi  = args.view % V
    print(f"Using view {vi}/{V}")

    img  = views["images"][vi].numpy()   # (H, W, 3) float32 [0,1]
    mask = views["masks"][vi].numpy()    # (H, W) bool
    H, W = img.shape[:2]
    c2w  = views["c2w"][vi]
    K    = views["K"][vi]

    print(f"Casting rays at 1/{args.down} resolution ({H//args.down}×{W//args.down}) …")
    origins, dirs, H2, W2 = cast_rays(c2w, K, H, W, down=args.down)

    print(f"Sphere tracing {H2*W2:,} rays (max_iters={trace_cfg.iters}) …")
    hit, iter_conv, sdf_min = trace_with_iter_count(f, origins, dirs, trace_cfg)

    hit_map  = hit.numpy().reshape(H2, W2)
    iter_map = iter_conv.numpy().reshape(H2, W2)

    from torch.nn.functional import interpolate
    mask_down = interpolate(
        torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
        size=(H2, W2), mode="nearest",
    ).squeeze().numpy().astype(bool)

    img_down = interpolate(
        torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
        size=(H2, W2), mode="bilinear", align_corners=False,
    ).squeeze().permute(1, 2, 0).numpy().clip(0, 1)

    out = args.out or (
        args.checkpoint.parent /
        f"convergence_map_view{vi:02d}"
        f"{'_iters' + str(args.max_iters) if args.max_iters else ''}.png"
    )

    make_figure(img_down, mask_down, hit_map, iter_map,
                trace_cfg.iters, vi, out)


if __name__ == "__main__":
    main()
