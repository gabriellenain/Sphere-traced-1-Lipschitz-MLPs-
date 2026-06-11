#!/usr/bin/env python3
"""Auto-debug sphere-tracing steps on foreground miss regions.

For a run directory, this script:
  1. traces one view with the configured iteration count,
  2. finds foreground rays that did not converge,
  3. retraces them with a longer budget,
  4. records the exact per-iteration t and f_theta(o+t d).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.config import TraceConfig
from lip_tracer.model import make_model
import lip_tracer.data as data_mod


def load_run(run_dir: Path, ckpt_name: str, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt_path = run_dir / "ckpt" / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = run_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = cfg["model"]
    f = make_model(
        hidden=m["hidden"],
        depth=m["depth"],
        group_size=m.get("group_size", 2),
        activation=m.get("activation", "groupsort"),
        input_encoding=m.get("input_encoding", "pe"),
        multires=m.get("multires", 6),
        architecture=m.get("architecture", "cpl"),
    ).to(device)
    f.load_state_dict(ckpt.get("f", ckpt), strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f, TraceConfig(**cfg["trace"]), Path(cfg["scene"]), ckpt_path


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


@torch.no_grad()
def trace_batch(f, origins, dirs, cfg: TraceConfig, iters: int, chunk: int):
    n = origins.shape[0]
    hit_all = torch.zeros(n, dtype=torch.bool, device=origins.device)
    iter_all = torch.full((n,), iters, dtype=torch.long, device=origins.device)
    min_abs_all = torch.full((n,), float("inf"), device=origins.device)
    final_abs_all = torch.full((n,), float("inf"), device=origins.device)
    final_t_all = torch.zeros(n, device=origins.device)
    ever_neg_all = torch.zeros(n, dtype=torch.bool, device=origins.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        o, d = origins[s:e], dirs[s:e]
        t = torch.zeros(e - s, device=origins.device)
        converged = torch.zeros(e - s, dtype=torch.bool, device=origins.device)
        iter_conv = torch.full((e - s,), iters, dtype=torch.long, device=origins.device)
        min_abs = torch.full((e - s,), float("inf"), device=origins.device)
        neg = torch.zeros(e - s, dtype=torch.bool, device=origins.device)
        for i in range(iters):
            escaped = t >= cfg.t_far
            active = ~(converged | escaped)
            if not active.any():
                break
            sdf = f(o + t.unsqueeze(-1) * d)
            abs_sdf = sdf.abs()
            min_abs = torch.where(active, torch.minimum(min_abs, abs_sdf), min_abs)
            neg = neg | (active & (sdf < 0))
            just = active & (abs_sdf < cfg.eps)
            iter_conv = torch.where(just & (iter_conv == iters), torch.full_like(iter_conv, i), iter_conv)
            converged = converged | just
            t = t + torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
        final_abs = f(o + t.unsqueeze(-1) * d).abs()
        hit = (converged | (final_abs < cfg.eps)) & (t < cfg.t_far) & (t >= 0)
        hit_all[s:e] = hit
        iter_all[s:e] = iter_conv
        min_abs_all[s:e] = min_abs
        final_abs_all[s:e] = final_abs
        final_t_all[s:e] = t
        ever_neg_all[s:e] = neg
        print(f"  trace {iters:3d}: {e:>{len(str(n))}}/{n}", flush=True)
    return {
        "hit": hit_all.cpu().numpy(),
        "iter": iter_all.cpu().numpy(),
        "min_abs": min_abs_all.cpu().numpy(),
        "final_abs": final_abs_all.cpu().numpy(),
        "final_t": final_t_all.cpu().numpy(),
        "ever_neg": ever_neg_all.cpu().numpy(),
    }


@torch.no_grad()
def trace_one_records(f, origin, direction, cfg: TraceConfig, iters: int):
    t = torch.zeros((), device=origin.device)
    converged = torch.zeros((), dtype=torch.bool, device=origin.device)
    rows = []
    for i in range(iters):
        escaped = t >= cfg.t_far
        active = ~(converged | escaped)
        x = origin + t * direction
        sdf = f(x.unsqueeze(0)).squeeze()
        rows.append({
            "iter": i,
            "t": float(t.cpu()),
            "sdf": float(sdf.cpu()),
            "abs_over_eps": float((sdf.abs() / cfg.eps).cpu()),
            "active": bool(active.cpu()),
            "converged_before_step": bool(converged.cpu()),
        })
        just = active & (sdf.abs() < cfg.eps)
        converged = converged | just
        t = t + torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
        if bool((converged | escaped).cpu()):
            # Keep exact stopping behavior visible, but no need to append flat tail.
            break
    return rows


def farthest_subset(coords: np.ndarray, n: int) -> np.ndarray:
    if len(coords) <= n:
        return np.arange(len(coords))
    chosen = [0]
    dist = np.full(len(coords), np.inf, dtype=np.float64)
    for _ in range(1, n):
        last = coords[chosen[-1]]
        dist = np.minimum(dist, ((coords - last) ** 2).sum(axis=1))
        chosen.append(int(np.argmax(dist)))
    return np.asarray(chosen, dtype=np.int64)


def choose_pixels(mask: np.ndarray, miss24: np.ndarray, hit_long: np.ndarray,
                  ever_neg_long: np.ndarray, n_each: int,
                  focus_box: tuple[int, int, int, int] | None = None):
    """Split foreground misses into three diagnostic classes.

    fixed     : converges with the longer budget (slow but fine).
    miss_root : never converges, but f goes ≤ 0 along the ray → a root exists,
                the tracer skipped it (bracketing would recover these).
    miss_nort : never converges and f stays > 0 → genuine tangent miss.
    """
    H, W = mask.shape
    miss = miss24.reshape(H, W) & mask
    if focus_box is not None:
        x0, y0, x1, y1 = focus_box
        roi = np.zeros_like(miss, dtype=bool)
        roi[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        miss = miss & roi
    hit_m = hit_long.reshape(H, W)
    neg_m = ever_neg_long.reshape(H, W)
    fixed = miss & hit_m
    stuck = miss & ~hit_m
    miss_root = stuck & neg_m
    miss_nort = stuck & ~neg_m
    chosen = []
    labels = []
    for name, arr in [("fixed", fixed), ("miss_root", miss_root), ("miss_nort", miss_nort)]:
        coords = np.column_stack(np.where(arr))
        if len(coords) == 0:
            continue
        # Sort from top-left for deterministic farthest-point seeding.
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        idx = farthest_subset(coords.astype(np.float64), n_each)
        for k, (y, x) in enumerate(coords[idx], start=1):
            chosen.append((int(y), int(x)))
            labels.append(f"{name}_{k}")
    return chosen, labels, fixed, miss_root, miss_nort


def save_records(out_dir: Path, labels, pixels, records):
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, (y, x), rows in zip(labels, pixels, records):
        path = out_dir / f"{label}_y{y}_x{x}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def make_figure(img, mask, fixed, miss_root, miss_nort, labels, pixels, records, cfg,
                short_iters, out_path: Path,
                focus_box: tuple[int, int, int, int] | None = None):
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(records), 1)))
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.7),
                             gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})

    if focus_box is None:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
    else:
        x0, y0, x1, y1 = focus_box
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    axes[0].imshow(img[y0:y1, x0:x1])
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[fixed]     = np.array([0.10, 0.35, 0.95, 0.65])  # blue  — slow but fixable
    overlay[miss_root] = np.array([0.99, 0.68, 0.20, 0.80])  # amber — root exists (tracer)
    overlay[miss_nort] = np.array([0.85, 0.05, 0.08, 0.75])  # red   — tangent miss (geometry)
    axes[0].imshow(overlay[y0:y1, x0:x1])
    for c, label, (y, x) in zip(colors, labels, pixels):
        axes[0].scatter([x - x0], [y - y0], s=34, facecolors="none", edgecolors=[c], linewidths=1.8)
        axes[0].text(x - x0 + 2, y - y0 + 2, label.split("_")[-1], color=c, fontsize=8, weight="bold")
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=[0.10, 0.35, 0.95], label="fixed by budget"),
        plt.Line2D([], [], marker="s", ls="", color=[0.99, 0.68, 0.20], label="miss: root exists"),
        plt.Line2D([], [], marker="s", ls="", color=[0.85, 0.05, 0.08], label="miss: no root"),
    ]
    axes[0].legend(handles=handles, fontsize=6, loc="lower right", frameon=False)
    axes[0].set_title(f"Foreground misses at {short_iters} steps")
    axes[0].axis("off")

    for c, label, rows in zip(colors, labels, records):
        it = np.array([r["iter"] for r in rows])
        abs_eps = np.array([r["abs_over_eps"] for r in rows])
        t = np.array([r["t"] for r in rows])
        absf = np.array([abs(r["sdf"]) for r in rows])
        short = label.split("_")[0] if "_" in label else label
        axes[1].plot(it, abs_eps, marker="o", ms=2.3, lw=1.0, color=c, label=label)
        # per-step geometric ratio |f_i| / |f_{i-1}| — plateau ≈ effective |∇f·d|
        ratio = absf[1:] / np.maximum(absf[:-1], 1e-12)
        axes[2].plot(it[1:], ratio, marker="o", ms=2.3, lw=1.0, color=c)
        axes[3].plot(it, t, marker="o", ms=2.3, lw=1.0, color=c, label=label)
    axes[1].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[1].axvline(short_iters, color="0.55", lw=0.8, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Sphere-tracing iteration")
    axes[1].set_ylabel(r"$|f_\theta(o+td)|/\epsilon$")
    axes[1].set_title("Residual per step")
    axes[1].grid(True, alpha=0.2)

    axes[2].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[2].axvline(short_iters, color="0.55", lw=0.8, ls=":")
    axes[2].set_ylim(0, 1.15)
    axes[2].set_xlabel("Sphere-tracing iteration")
    axes[2].set_ylabel(r"step ratio $|f_i|/|f_{i-1}|$")
    axes[2].set_title(r"Stall ratio (plateau $\approx |\nabla f\cdot d|$)")
    axes[2].grid(True, alpha=0.2)

    axes[3].axvline(short_iters, color="0.55", lw=0.8, ls=":")
    axes[3].set_xlabel("Sphere-tracing iteration")
    axes[3].set_ylabel("Ray depth t")
    axes[3].set_title("Accumulated ray depth")
    axes[3].grid(True, alpha=0.2)
    axes[3].legend(fontsize=6, loc="best", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--view", type=int, default=16)
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--short-iters", type=int, default=24)
    ap.add_argument("--long-iters", type=int, default=128)
    ap.add_argument("--n-each", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--out-dir", type=Path, default=Path("figs/sphere_trace_steps"))
    ap.add_argument("--focus-box", type=str, default=None,
                    help="Optional downsampled ROI x0,y0,x1,y1 used to select and display misses.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    f, cfg, scene, ckpt_path = load_run(args.run_dir, args.ckpt, args.device)
    data_mod.SCENE = scene
    views = data_mod.load_views(scene, down=args.down)
    vi = args.view % int(views["images"].shape[0])
    origins, dirs, img, mask, H, W = rays_for_view(views, vi, args.device)

    print(f"run={args.run_dir.name} ckpt={ckpt_path.name} view={vi} down={args.down}")
    print(f"eps={cfg.eps:g} t_far={cfg.t_far:g} short={args.short_iters} long={args.long_iters}")
    short = trace_batch(f, origins, dirs, cfg, args.short_iters, args.chunk)
    long = trace_batch(f, origins, dirs, cfg, args.long_iters, args.chunk)
    fg = mask.reshape(-1)
    miss24 = fg & ~short["hit"]
    fixed_long = miss24 & long["hit"]
    stuck_long = miss24 & ~long["hit"]
    print(
        f"foreground rays: {int(fg.sum())}\n"
        f"24-step misses: {int(miss24.sum())} ({100*miss24.sum()/max(fg.sum(),1):.2f}%)\n"
        f"fixed by {args.long_iters} steps: {int(fixed_long.sum())} ({100*fixed_long.sum()/max(miss24.sum(),1):.2f}% of misses)\n"
        f"still missed at {args.long_iters}: {int(stuck_long.sum())}",
        flush=True,
    )

    focus_box = None
    if args.focus_box:
        focus_box = tuple(int(v) for v in args.focus_box.split(","))
        if len(focus_box) != 4:
            raise ValueError("--focus-box must be x0,y0,x1,y1")
    pixels, labels, fixed_map, miss_root_map, miss_nort_map = choose_pixels(
        mask, miss24, long["hit"], long["ever_neg"], args.n_each, focus_box)
    print(
        f"still missed: root-exists(tracer-fixable)={int((stuck_long & long['ever_neg']).sum())} "
        f"no-root(tangent)={int((stuck_long & ~long['ever_neg']).sum())}",
        flush=True,
    )
    if not pixels:
        raise SystemExit("No foreground misses found.")
    records = []
    for label, (y, x) in zip(labels, pixels):
        idx = y * W + x
        rows = trace_one_records(f, origins[idx], dirs[idx], cfg, args.long_iters)
        records.append(rows)
        last = rows[-1]
        print(
            f"{label}: pixel down=({x},{y}) full≈({x*args.down},{y*args.down}) "
            f"iters={len(rows)} last |f|/eps={last['abs_over_eps']:.3g} t={last['t']:.4f}",
            flush=True,
        )

    suffix = "_focus" if focus_box else ""
    stem = f"{args.run_dir.name}_view{vi:02d}_down{args.down}_trace_steps{suffix}"
    save_records(args.out_dir / f"{stem}_csv", labels, pixels, records)
    make_figure(img, mask, fixed_map, miss_root_map, miss_nort_map, labels, pixels, records, cfg,
                args.short_iters, args.out_dir / f"{stem}.png", focus_box)
    summary = {
        "run_dir": str(args.run_dir),
        "ckpt": str(ckpt_path),
        "view": vi,
        "down": args.down,
        "eps": cfg.eps,
        "short_iters": args.short_iters,
        "long_iters": args.long_iters,
        "foreground_rays": int(fg.sum()),
        "misses_short": int(miss24.sum()),
        "fixed_by_long": int(fixed_long.sum()),
        "still_miss_long": int(stuck_long.sum()),
        "still_miss_root_exists": int((stuck_long & long["ever_neg"]).sum()),
        "still_miss_no_root": int((stuck_long & ~long["ever_neg"]).sum()),
        "selected_pixels_down": [{"label": l, "x": x, "y": y} for l, (y, x) in zip(labels, pixels)],
        "focus_box_down": focus_box,
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {args.out_dir / f'{stem}.png'}")


if __name__ == "__main__":
    main()
