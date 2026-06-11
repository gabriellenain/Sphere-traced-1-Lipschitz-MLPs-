#!/usr/bin/env python3
"""ICLR-style pre/post Newton convergence maps for DTU scene runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.config import TraceConfig
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd
import lip_tracer.data as data_mod


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def default_view_for_run(run_dir: Path) -> int:
    name = run_dir.name.lower()
    if "scan122" in name:
        return 50
    if "scan65" in name:
        return 16
    return 0


def resolve_ckpt(run_dir: Path, ckpt_name: str) -> Path:
    ckpt = Path(ckpt_name)
    if ckpt.is_absolute() or ckpt.exists():
        return ckpt
    candidates = [
        run_dir / ckpt_name,
        run_dir / "ckpt" / ckpt_name,
        run_dir / "ckpt" / f"{ckpt_name}.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    for fallback in ("checkpoint_final.pt", "checkpoint_best_photo.pt", "checkpoint_latest.pt"):
        cand = run_dir / "ckpt" / fallback
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no checkpoint found for {run_dir}")


def load_run(run_dir: Path, ckpt_name: str, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt_path = resolve_ckpt(run_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = cfg["model"]
    f = make_model(
        hidden=model_cfg["hidden"],
        depth=model_cfg["depth"],
        group_size=model_cfg.get("group_size", 2),
        activation=model_cfg.get("activation", "groupsort"),
        input_encoding=model_cfg.get("input_encoding", "pe"),
        multires=model_cfg.get("multires", 6),
        architecture=model_cfg.get("architecture", "cpl"),
    ).to(device)
    f.load_state_dict(ckpt.get("f", ckpt.get("model", ckpt)), strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    trace_cfg = TraceConfig(**cfg["trace"])
    return f, trace_cfg, Path(cfg["scene"]), ckpt_path


def rays_for_view(views: dict, view_idx: int, device: str):
    img = views["images"][view_idx].numpy()
    mask = views["masks"][view_idx].numpy().astype(bool)
    H, W = img.shape[:2]
    K = views["K"][view_idx].numpy()
    c2w = views["c2w"][view_idx].numpy()

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d_cam = np.stack([
        (xs + 0.5 - K[0, 2]) / K[0, 0],
        (ys + 0.5 - K[1, 2]) / K[1, 1],
        np.ones_like(xs),
    ], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).copy()
    return (
        torch.from_numpy(origins.astype(np.float32)).to(device),
        torch.from_numpy(dirs.reshape(-1, 3).astype(np.float32)).to(device),
        img,
        mask,
        H,
        W,
    )


def trace_maps(f, origins, dirs, trace_cfg: TraceConfig, chunk: int):
    n = origins.shape[0]
    hit_parts, pre_parts, post_parts = [], [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        with torch.no_grad():
            x_hit, _, hit, f_pre = trace_nograd(
                f, origins[s:e], dirs[s:e], trace_cfg, return_diag=True)
            f_post = f(x_hit).abs().detach()
        hit_parts.append(hit.detach().cpu())
        pre_parts.append(f_pre.detach().cpu())
        post_parts.append(f_post.cpu())
        print(f"  rays {e:>{len(str(n))}}/{n} ({100*e/n:5.1f}%)", flush=True)
    return (
        torch.cat(hit_parts).numpy(),
        torch.cat(pre_parts).numpy(),
        torch.cat(post_parts).numpy(),
    )


def residual_panel(residual_ratio: np.ndarray, valid: np.ndarray) -> np.ndarray:
    score = np.clip(np.log10(np.maximum(residual_ratio, 1e-4)), -4.0, 1.0)
    score[~valid] = np.nan
    return score


def make_figure(
    img: np.ndarray,
    mask: np.ndarray,
    hit: np.ndarray,
    pre_abs: np.ndarray,
    post_abs: np.ndarray,
    run_dir: Path,
    ckpt_path: Path,
    view_idx: int,
    trace_cfg: TraceConfig,
    out_path: Path,
) -> None:
    H, W = mask.shape
    hit = hit.reshape(H, W).astype(bool)
    pre = pre_abs.reshape(H, W) / max(trace_cfg.eps, 1e-12)
    post = post_abs.reshape(H, W) / max(trace_cfg.eps, 1e-12)
    fg_hit = mask & hit
    fg_miss = mask & ~hit
    pre_ok = fg_hit & (pre < 1.0)
    post_ok = fg_hit & (post < 1.0)

    cmap = plt.cm.magma_r.copy()
    cmap.set_bad("#f0f0f0")
    norm = mcolors.Normalize(vmin=-4.0, vmax=1.0)
    panels = [
        ("Input", None, None),
        ("Before Newton", residual_panel(pre, fg_hit), pre_ok),
        ("After 2 Newton Steps", residual_panel(post, fg_hit), post_ok),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), constrained_layout=False)
    axes[0].imshow(img)
    axes[0].set_title("Input", pad=2, fontsize=8)
    axes[0].set_axis_off()

    for ax, (_, arr, ok) in zip(axes[1:], panels[1:]):
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[fg_miss] = np.array([0.78, 0.08, 0.12, 0.88], dtype=np.float32)
        fail = fg_hit & ~ok
        overlay[fail] = np.array([0.98, 0.62, 0.05, 0.72], dtype=np.float32)
        ax.imshow(overlay, interpolation="nearest")
        ax.set_axis_off()

    axes[1].set_title("Before Newton", pad=2, fontsize=8)
    axes[2].set_title("After Newton", pad=2, fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(left=0.006, right=0.91, top=0.90, bottom=0.13, wspace=0.035)
    cax = fig.add_axes([0.925, 0.205, 0.012, 0.66])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.set_title(r"$\log_{10}$", fontsize=6, pad=2)
    cb.ax.axhline(0.0, color="white", lw=0.8)

    n_fg = max(int(mask.sum()), 1)
    n_hit = max(int(fg_hit.sum()), 1)
    hit_rate = 100.0 * float(fg_hit.sum()) / n_fg
    pre_rate = 100.0 * float(pre_ok.sum()) / n_hit
    post_rate = 100.0 * float(post_ok.sum()) / n_hit
    pre_med = float(np.nanmedian(pre[fg_hit])) if fg_hit.any() else float("nan")
    post_med = float(np.nanmedian(post[fg_hit])) if fg_hit.any() else float("nan")

    scene = "scan122" if "scan122" in run_dir.name else "scan65" if "scan65" in run_dir.name else run_dir.name
    print(
        f"{scene} view {view_idx}: FG hit {hit_rate:.1f}%, "
        f"conv {pre_rate:.1f}% -> {post_rate:.1f}%, "
        f"median |f|/eps {pre_med:.2f} -> {post_med:.2f}",
        flush=True,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#c7141f", label="Foreground miss"),
            Patch(facecolor="#fa9e0d", label=r"Hit, $|f| \geq \epsilon$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.56, -0.005),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"saved {out_path}")


def parse_view_overrides(items: list[str]) -> dict[str, int]:
    out = {}
    for item in items:
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        out[key.lower()] = int(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--ckpt", default="checkpoint_final.pt")
    parser.add_argument("--out-dir", type=Path, default=Path("figs/iclr_convergence_newton"))
    parser.add_argument("--down", type=int, default=4)
    parser.add_argument("--view", action="append", default=[],
                        help="Override per scan, e.g. --view scan122:50 --view scan65:16")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--newton-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    overrides = parse_view_overrides(args.view)
    for run_dir in args.runs:
        print(f"=== {run_dir} ===", flush=True)
        f, trace_cfg, scene, ckpt_path = load_run(run_dir, args.ckpt, args.device)
        trace_cfg.eps = args.eps
        trace_cfg.newton_steps = args.newton_steps
        data_mod.SCENE = scene
        print(f"loading scene {scene} at down={args.down}", flush=True)
        views = data_mod.load_views(scene, down=args.down)
        run_key = "scan122" if "scan122" in run_dir.name.lower() else "scan65" if "scan65" in run_dir.name.lower() else run_dir.name.lower()
        view_idx = overrides.get(run_key, default_view_for_run(run_dir)) % int(views["images"].shape[0])
        print(f"checkpoint {ckpt_path.name}; view {view_idx}; trace eps={trace_cfg.eps} newton={trace_cfg.newton_steps}", flush=True)
        origins, dirs, img, mask, H, W = rays_for_view(views, view_idx, args.device)
        hit, pre_abs, post_abs = trace_maps(f, origins, dirs, trace_cfg, args.chunk)
        out_name = f"{run_key}_view{view_idx:02d}_convergence_eps1e-3_newton2.png"
        make_figure(img, mask, hit, pre_abs, post_abs, run_dir, ckpt_path, view_idx, trace_cfg, args.out_dir / out_name)


if __name__ == "__main__":
    main()
