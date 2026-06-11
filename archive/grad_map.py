#!/usr/bin/env python3
"""Visualize the input-gradient norm of an SDF checkpoint in space.

Example:
  python grad_map.py \
    --ckpt outputs/run_20260430_171910_dtu/checkpoint_best_photo.pt \
    --out outputs/grad_map_scan122_z0.png \
    --bound 0.8 --res 512 --axis z --slice 0.0

  python grad_map.py \
    --ckpt outputs/run_20260430_171910_dtu/checkpoint_best_photo.pt \
    --out outputs/grad_scan122 \
    --bound 0.8 --res 512 --axis all --slice -0.4 0.0 0.4 \
    --volume-res 96
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    architecture = ckpt.get("architecture", "cpl")
    hidden = ckpt.get("hidden", 256)
    depth = ckpt.get("depth", 8)
    group_size = ckpt.get("group_size", 2)
    activation = ckpt.get("activation", "groupsort")
    input_encoding = ckpt.get("input_encoding", "identity")
    multires = ckpt.get("multires", 6)

    for value in state.values():
        if value.ndim >= 2:
            hidden = value.shape[-1]
            break

    from lip_tracer.model import make_model
    f = make_model(
        hidden=hidden,
        depth=depth,
        group_size=group_size,
        activation=activation,
        input_encoding=input_encoding,
        multires=multires,
        architecture=architecture,
    )
    f.load_state_dict(state, strict=False)
    return f.to(device).eval(), ckpt


def make_slice(axis: str, slice_value: float, bound: float, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(-bound, bound, res, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords[::-1])
    pts = np.zeros((res * res, 3), dtype=np.float32)
    a = {"x": 0, "y": 1, "z": 2}[axis]
    free = [i for i in range(3) if i != a]
    pts[:, free[0]] = xx.ravel()
    pts[:, free[1]] = yy.ravel()
    pts[:, a] = slice_value
    return pts, xx, yy


def make_volume(bound: float, res: int) -> np.ndarray:
    coords = np.linspace(-bound, bound, res, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def _forward_from_encoded(f, h: torch.Tensor) -> torch.Tensor:
    """Evaluate the post-encoding part of the model on encoded coordinates."""
    if f.__class__.__name__ == "RegularMLP":
        return f.net(h).squeeze(-1)

    h = torch.nn.functional.pad(h, (0, f.hidden - h.shape[-1]))
    h = f.net(h)
    if torch.is_grad_enabled():
        w = f.head_weight / torch.linalg.vector_norm(f.head_weight).clamp(min=1e-6)
        f._head_w_buf.copy_(w.detach())
    else:
        w = f._head_w_buf
    return (h * w).sum(-1) + f.head_bias.squeeze(-1)


def eval_sdf_and_grad_norm(f, pts: np.ndarray, device: str, chunk: int,
                            label: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdf_parts: list[np.ndarray] = []
    grad_parts: list[np.ndarray] = []
    grad_encoded_parts: list[np.ndarray] = []
    n = len(pts)
    n_chunks = (n + chunk - 1) // chunk
    report_every = max(1, n_chunks // 20)
    t0 = time.time()
    last = t0
    tag = f"[{label}] " if label else ""
    print(f"{tag}eval start: {n} pts in {n_chunks} chunks of {chunk}", flush=True)
    for ci, i in enumerate(range(0, n, chunk)):
        x = torch.from_numpy(pts[i:i + chunk]).to(device).requires_grad_(True)
        with torch.enable_grad():
            y = f(x)
            grad = torch.autograd.grad(y.sum(), x, create_graph=False)[0]

            encoder = getattr(f, "encoder", None)
            encoded = encoder(x.detach()) if encoder is not None else x.detach()
            encoded = encoded.requires_grad_(True)
            y_encoded = _forward_from_encoded(f, encoded)
            grad_encoded = torch.autograd.grad(
                y_encoded.sum(), encoded, create_graph=False
            )[0]
        sdf_parts.append(y.detach().float().cpu().numpy())
        grad_parts.append(grad.norm(dim=-1).detach().float().cpu().numpy())
        grad_encoded_parts.append(grad_encoded.norm(dim=-1).detach().float().cpu().numpy())
        done = ci + 1
        if done == n_chunks or done % report_every == 0:
            now = time.time()
            elapsed = now - t0
            rate = done / max(elapsed, 1e-9)
            eta = (n_chunks - done) / max(rate, 1e-9)
            print(f"{tag}  chunk {done}/{n_chunks} ({100.0 * done / n_chunks:5.1f}%) "
                  f"elapsed={elapsed:6.1f}s eta={eta:6.1f}s", flush=True)
            last = now
    print(f"{tag}eval done in {time.time() - t0:.1f}s", flush=True)
    return (
        np.concatenate(sdf_parts),
        np.concatenate(grad_parts),
        np.concatenate(grad_encoded_parts),
    )


def _safe_name(value: float) -> str:
    if math.isclose(value, 0.0):
        return "0"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _coords_for_axis(axis: str) -> tuple[str, str]:
    if axis == "x":
        return "y", "z"
    if axis == "y":
        return "x", "z"
    return "x", "y"


def save_png(out: Path, grad_norm: np.ndarray, sdf: np.ndarray, bound: float, res: int,
             axis: str, slice_value: float, vmax: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = grad_norm.reshape(res, res)
    s = sdf.reshape(res, res)
    if vmax is None:
        vmax = float(np.percentile(g, 99.0))
        vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(
        g,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        extent=[-bound, bound, -bound, bound],
        interpolation="nearest",
    )
    ax.contour(
        np.linspace(-bound, bound, res),
        np.linspace(-bound, bound, res)[::-1],
        s,
        levels=[0.0],
        colors="cyan",
        linewidths=0.8,
    )
    ax.set_title(f"|grad f| on {axis}={slice_value:g}")
    xlabel, ylabel = _coords_for_axis(axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="|grad f|")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_pe_compare_png(out: Path, grad_x: np.ndarray, grad_encoded: np.ndarray,
                        bound: float, res: int, axis: str, slice_value: float,
                        vmax: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gx = grad_x.reshape(res, res)
    ge = grad_encoded.reshape(res, res)
    ratio = gx / np.maximum(ge, 1e-12)
    if vmax is None:
        vmax = float(np.percentile(np.concatenate([grad_x, grad_encoded]), 99.0))
        vmax = max(vmax, 1e-6)
    ratio_vmax = float(np.percentile(ratio, 99.0))
    ratio_vmax = max(ratio_vmax, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    panels = [
        (gx, r"$\|\partial f / \partial x\|$", "magma", 0.0, vmax),
        (ge, r"$\|\partial f / \partial \gamma\|$", "magma", 0.0, vmax),
        (ratio, r"$\|\partial f / \partial x\| / \|\partial f / \partial \gamma\|$",
         "viridis", 0.0, ratio_vmax),
    ]
    for ax, (img, title, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_i,
            extent=[-bound, bound, -bound, bound],
            interpolation="nearest",
        )
        ax.set_title(title)
        xlabel, ylabel = _coords_for_axis(axis)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.suptitle(f"Gradient norms before/after positional encoding on {axis}={slice_value:g}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_hist_png(out: Path, grad_norm: np.ndarray, xmax: float, ymax: float | None,
                  density: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(grad_norm, bins=120, range=(0.0, xmax), density=density,
            color="steelblue", alpha=0.85)
    ax.axvline(1.0, color="crimson", linewidth=1.2, label="1-Lipschitz target")
    ax.set_xlim(0.0, xmax)
    if ymax is not None:
        ax.set_ylim(0.0, ymax)
    ax.set_xlabel("|grad f|")
    ax.set_ylabel("density" if density else "count")
    ax.set_title("|grad f| distribution")
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_pe_hist_png(out: Path, grad_x: np.ndarray, grad_encoded: np.ndarray,
                     xmax: float, ymax: float | None, density: bool,
                     title: str = "Gradient norm before/after positional encoding") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(grad_x, bins=120, range=(0.0, xmax), density=density,
            color="steelblue", alpha=0.55, label=r"$\|\partial f / \partial x\|$")
    ax.hist(grad_encoded, bins=120, range=(0.0, xmax), density=density,
            color="darkorange", alpha=0.55, label=r"$\|\partial f / \partial \gamma\|$")
    ax.axvline(1.0, color="crimson", linewidth=1.2, label="1-Lipschitz target")
    ax.set_xlim(0.0, xmax)
    if ymax is not None:
        ax.set_ylim(0.0, ymax)
    ax.set_xlabel("gradient norm")
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_hist_overlay_png(out: Path, items: list[tuple[str, np.ndarray]],
                          xmax: float, ymax: float | None, density: bool,
                          title: str = "|grad f| distribution") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["steelblue", "darkorange", "seagreen", "purple"]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for (label, grad_norm), color in zip(items, colors):
        ax.hist(grad_norm, bins=120, range=(0.0, xmax), density=density,
                color=color, alpha=0.55, label=label)
    ax.axvline(1.0, color="crimson", linewidth=1.2, label="1-Lipschitz target")
    ax.set_xlim(0.0, xmax)
    if ymax is not None:
        ax.set_ylim(0.0, ymax)
    ax.set_xlabel("|grad f|")
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_volume_projection_png(out: Path, grad_norm: np.ndarray, bound: float, res: int,
                               vmax: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = grad_norm.reshape(res, res, res)
    if vmax is None:
        vmax = float(np.percentile(g, 99.0))
        vmax = max(vmax, 1e-6)

    projections = [
        ("xy max over z", g.max(axis=2).T[::-1], "x", "y"),
        ("xz max over y", g.max(axis=1).T[::-1], "x", "z"),
        ("yz max over x", g.max(axis=0).T[::-1], "y", "z"),
        ("xy mean over z", g.mean(axis=2).T[::-1], "x", "y"),
        ("xz mean over y", g.mean(axis=1).T[::-1], "x", "z"),
        ("yz mean over x", g.mean(axis=0).T[::-1], "y", "z"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (title, img, xlabel, ylabel) in zip(axes.ravel(), projections):
        im = ax.imshow(img, cmap="magma", vmin=0.0, vmax=vmax,
                       extent=[-bound, bound, -bound, bound],
                       interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="|grad f|")
    fig.suptitle("Spatial projections of |grad f|")
    fig.savefig(out, dpi=160)
    plt.close(fig)


PAPER_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _paper_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })
    return plt


def _save_fig(fig, out: Path) -> None:
    fig.savefig(out)
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)


def compute_layer_spectral_data(f) -> list[tuple[str, np.ndarray]]:
    """Per-layer full singular value spectrum of the underlying weight matrix.

    Works for both ConvexPotentialLayer (the inner W in x - (2/||W||^2) W^T relu(Wx+b))
    and nn.Linear. Returns a list of (short_name, singular_values) ordered by depth.
    """
    out: list[tuple[str, np.ndarray]] = []
    idx = 0
    for name, mod in f.named_modules():
        cls = mod.__class__.__name__
        if cls in ("ConvexPotentialLayer", "Linear") and hasattr(mod, "weight"):
            w = mod.weight.detach().cpu().float().numpy()
            if w.ndim != 2:
                continue
            try:
                sv = np.linalg.svd(w, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            out.append((f"L{idx}", sv))
            idx += 1
    return out


def compute_layer_spectral_norms(f) -> list[tuple[str, float]]:
    return [(name, float(sv[0])) for name, sv in compute_layer_spectral_data(f)]


def compute_cpl_sigma_sq(f) -> list[tuple[str, float, float]]:
    """For each ConvexPotentialLayer, return (name, estimated_sigma_sq, true_sigma_sq).

    estimated_sigma_sq is the running buffer used at inference time
    (`_sigma_sq_buf`). true_sigma_sq is sigma_max(W)^2 from full SVD.
    The 1-Lipschitz guarantee requires estimated >= true; an under-estimate
    means the effective Lipschitz constant exceeds 1.
    """
    out: list[tuple[str, float, float]] = []
    idx = 0
    for name, mod in f.named_modules():
        if mod.__class__.__name__ != "ConvexPotentialLayer":
            continue
        if not hasattr(mod, "weight") or not hasattr(mod, "_sigma_sq_buf"):
            continue
        w = mod.weight.detach().cpu().float().numpy()
        if w.ndim != 2:
            continue
        try:
            sv = np.linalg.svd(w, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        true_sq = float(sv[0]) ** 2
        est_sq = float(mod._sigma_sq_buf.detach().cpu().float().item())
        out.append((f"L{idx}", est_sq, true_sq))
        idx += 1
    return out


def save_sigma_estimate_figure(out: Path,
                                per_label: list[tuple[str, list[tuple[str, float, float]]]]) -> None:
    """Two-panel: estimated vs true sigma_sq per layer, and relative error.
    Negative relative error means power-iteration UNDER-estimates -> Lipschitz violated."""
    plt = _paper_style()

    if not per_label or not per_label[0][1]:
        return

    n_layers = len(per_label[0][1])
    layer_idx = np.arange(n_layers)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.4, 1.0]})
    ax_top, ax_bot = axes
    rel_ppm_values: list[np.ndarray] = []

    for i, (label, items) in enumerate(per_label):
        color = PAPER_COLORS[i % len(PAPER_COLORS)]
        est = np.array([e for _, e, _ in items])
        true_ = np.array([t for _, _, t in items])
        rel = (est - true_) / np.maximum(true_, 1e-12)
        rel_ppm = 1e6 * rel
        rel_ppm_values.append(rel_ppm)

        ax_top.plot(layer_idx, true_, color=color, marker="o", markersize=4,
                    linestyle="-", label=f"{label}  true $\\sigma^2$")
        ax_top.plot(layer_idx, est, color=color, marker="x", markersize=5,
                    linestyle="none", alpha=0.9,
                    label=f"{label}  est. $\\sigma^2$")
        ax_bot.plot(layer_idx, rel_ppm, color=color, marker="o", markersize=4,
                    label=label)

    ax_top.set_ylabel(r"$\sigma^2(W_\ell)$")
    ax_top.set_title(r"Power-iteration estimate vs. true $\sigma^2$")
    ax_top.legend(loc="best", fontsize=8, ncol=max(1, len(per_label)))

    rel_all = np.concatenate(rel_ppm_values)
    finite_rel = rel_all[np.isfinite(rel_all)]
    if finite_rel.size:
        lo = float(finite_rel.min())
        hi = float(finite_rel.max())
        span = max(hi - lo, max(abs(lo), abs(hi)) * 0.2, 1.0)
        lo = min(lo - 0.2 * span, -0.1 * span)
        hi = max(hi + 0.2 * span, 0.1 * span)
        ax_bot.set_ylim(lo, hi)

    ax_bot.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    lo, hi = ax_bot.get_ylim()
    if lo < 0.0:
        ax_bot.axhspan(lo, min(0.0, hi), color="crimson", alpha=0.08,
                       label="under-estimate (unsafe)")
    ax_bot.set_xlabel("layer index")
    ax_bot.set_xticks(layer_idx)
    ax_bot.set_ylabel(r"rel. error $(\hat\sigma^2 - \sigma^2)/\sigma^2$ [ppm]")
    ax_bot.legend(loc="best", fontsize=8)

    _save_fig(fig, out)
    plt.close(fig)


def save_spectral_figure(out: Path, per_label: list[tuple[str, list[tuple[str, np.ndarray]]]],
                         hidden_hint: int | None = None) -> None:
    """Two-panel paper figure: top singular value vs depth (line plot) and full
    SV spectrum heatmap stacked across layers, one heatmap per run."""
    plt = _paper_style()

    if not per_label:
        return
    n_runs = len(per_label)
    n_layers = len(per_label[0][1])
    layer_idx = np.arange(n_layers)
    sv_dim = max(len(sv) for _, items in per_label for _, sv in items)

    fig = plt.figure(figsize=(7.0, 2.6 + 1.6 * n_runs), constrained_layout=True)
    gs = fig.add_gridspec(n_runs + 1, 1, height_ratios=[2.4] + [1.0] * n_runs)

    ax_top = fig.add_subplot(gs[0, 0])
    for i, (label, items) in enumerate(per_label):
        top_sv = np.array([sv[0] for _, sv in items])
        ax_top.plot(layer_idx, top_sv, marker="o", markersize=4,
                    color=PAPER_COLORS[i % len(PAPER_COLORS)], label=label)
    ax_top.axhline(1.0, color="black", linewidth=0.9, linestyle="--",
                   alpha=0.7, label=r"$\sigma=1$")
    ax_top.set_xlabel("layer index")
    ax_top.set_ylabel(r"$\sigma_{\max}(W_\ell)$")
    ax_top.set_title("Per-layer top singular value")
    ax_top.set_xticks(layer_idx)
    ax_top.legend(loc="best")

    vmax = max(sv.max() for _, items in per_label for _, sv in items)
    for ri, (label, items) in enumerate(per_label):
        ax = fig.add_subplot(gs[ri + 1, 0])
        spectrum = np.full((n_layers, sv_dim), np.nan)
        for li, (_, sv) in enumerate(items):
            spectrum[li, :len(sv)] = sv
        im = ax.imshow(spectrum, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0.0, vmax=vmax,
                       interpolation="nearest")
        ax.set_ylabel(f"{label}\nlayer")
        ax.set_yticks(layer_idx)
        if ri == n_runs - 1:
            ax.set_xlabel("singular value index")
        else:
            ax.set_xticklabels([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, label=r"$\sigma_i$", pad=0.01)

    fig.suptitle("Spectral analysis of weight matrices", fontsize=13)
    _save_fig(fig, out)
    plt.close(fig)


def save_radial_figure(out: Path, items: list[tuple[str, np.ndarray, np.ndarray]],
                       bound: float, n_bins: int = 60) -> None:
    """Two-panel paper figure:
       (top) median |grad f| with IQR band vs ||x||
       (bottom) fraction of points with |grad f| > 1 vs ||x||
    """
    plt = _paper_style()

    edges = np.linspace(0.0, bound * np.sqrt(3.0), n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.6, 1.0]})
    ax_top, ax_bot = axes

    for (label, pts, g), color in zip(items, PAPER_COLORS):
        r = np.linalg.norm(pts, axis=-1)
        idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)

        med = np.full(n_bins, np.nan)
        q25 = np.full(n_bins, np.nan)
        q75 = np.full(n_bins, np.nan)
        frac = np.full(n_bins, np.nan)
        for b in range(n_bins):
            sel = g[idx == b]
            if sel.size == 0:
                continue
            med[b] = np.median(sel)
            q25[b] = np.percentile(sel, 25)
            q75[b] = np.percentile(sel, 75)
            frac[b] = float((sel > 1.0).mean())

        ax_top.fill_between(centers, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax_top.plot(centers, med, color=color, label=label)
        ax_bot.plot(centers, frac, color=color, label=label)

    ax_top.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7,
                   label=r"$|\nabla f|=1$")
    ax_top.set_ylabel(r"$|\nabla f|$  (median, IQR)")
    ax_top.set_title("Radial profile of gradient magnitude")
    ax_top.legend(loc="best")

    ax_bot.set_xlim(0.0, bound * np.sqrt(3.0))
    ax_bot.set_ylim(0.0, 1.0)
    ax_bot.set_xlabel(r"$\|x\|$")
    ax_bot.set_ylabel(r"frac. $|\nabla f| > 1$")

    _save_fig(fig, out)
    plt.close(fig)


# Backwards-compatible aliases used elsewhere in the file.
def save_spectral_norm_bars(out: Path, per_label: list[tuple[str, list[tuple[str, float]]]]) -> None:
    converted = [(label, [(n, np.array([s])) for n, s in items]) for label, items in per_label]
    save_spectral_figure(out, converted)


def save_radial_profile(out: Path, items: list[tuple[str, np.ndarray, np.ndarray]],
                        bound: float, n_bins: int = 60) -> None:
    save_radial_figure(out, items, bound, n_bins)


def save_model_diagnostics(f, out_dir: Path, label: str) -> dict:
    extras: dict = {}

    sv_data = compute_layer_spectral_data(f)
    if sv_data:
        sigma_out = out_dir / "spectral_norms.png"
        save_spectral_figure(sigma_out, [(label, sv_data)])
        print(f"saved spectral figure -> {sigma_out}")
        for n, sv in sv_data:
            rank1 = float(sv[0] / max(sv[1], 1e-12)) if len(sv) > 1 else math.inf
            print(f"  sigma_max({n}) = {float(sv[0]):.4f}  rank-1 ratio = {rank1:.2f}")
    extras["spectral_data"] = sv_data

    sigma_sq_data = compute_cpl_sigma_sq(f)
    if sigma_sq_data:
        est_out = out_dir / "sigma_estimate.png"
        save_sigma_estimate_figure(est_out, [(label, sigma_sq_data)])
        print(f"saved sigma-estimate figure -> {est_out}")
        worst = min(((e - t) / max(t, 1e-12)) for _, e, t in sigma_sq_data)
        print(f"  worst relative error (est-true)/true = {worst:+.4f}  "
              f"({'UNDER-estimate' if worst < 0 else 'safe'})")
    extras["sigma_sq_data"] = sigma_sq_data

    return extras


def summarize_grad(grad_norm: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(grad_norm.mean()),
        "p50": float(np.percentile(grad_norm, 50)),
        "p90": float(np.percentile(grad_norm, 90)),
        "p99": float(np.percentile(grad_norm, 99)),
        "max": float(grad_norm.max()),
        "frac_gt_1": float((grad_norm > 1.0).mean()),
        "frac_gt_1p05": float((grad_norm > 1.05).mean()),
    }


def print_summary(label: str, grad_norm: np.ndarray) -> None:
    stats = summarize_grad(grad_norm)
    print(
        f"{label} |grad f| "
        f"mean={stats['mean']:.4f} "
        f"p50={stats['p50']:.4f} "
        f"p90={stats['p90']:.4f} "
        f"p99={stats['p99']:.4f} "
        f"max={stats['max']:.4f} "
        f">1={100.0 * stats['frac_gt_1']:.2f}% "
        f">1.05={100.0 * stats['frac_gt_1p05']:.2f}%"
    )


def process_ckpt(ckpt_path: Path, out_dir: Path, args, device: str,
                 axes: list[str]) -> dict:
    f, ckpt = load_model(ckpt_path, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    grads: dict[str, np.ndarray] = {}
    extras = save_model_diagnostics(f, out_dir, out_dir.name)

    for axis in axes:
        for slice_value in args.slice_values:
            pts, _, _ = make_slice(axis, slice_value, args.bound, args.res)
            sdf, grad_norm, grad_encoded_norm = eval_sdf_and_grad_norm(
                f, pts, device, args.chunk,
                label=f"{out_dir.name} slice {axis}={slice_value:g}",
            )
            stem = f"grad_{axis}_{_safe_name(slice_value)}"
            png_out = out_dir / f"{stem}.png"
            pe_png_out = out_dir / f"{stem}_pe_compare.png"
            npz_out = out_dir / f"{stem}.npz"
            save_png(png_out, grad_norm, sdf, args.bound, args.res, axis, slice_value, args.vmax)
            save_pe_compare_png(
                pe_png_out, grad_norm, grad_encoded_norm,
                args.bound, args.res, axis, slice_value, args.vmax,
            )
            np.savez_compressed(
                npz_out,
                points=pts,
                sdf=sdf,
                grad_norm=grad_norm,
                grad_norm_world=grad_norm,
                grad_norm_encoded=grad_encoded_norm,
                axis=axis, slice=slice_value, bound=args.bound, res=args.res,
                ckpt=str(ckpt_path), step=ckpt.get("step", -1),
            )
            print(f"saved heatmap -> {png_out}")
            print(f"saved PE compare heatmap -> {pe_png_out}")
            print(f"saved arrays  -> {npz_out}")
            print_summary(f"{axis}={slice_value:g} world", grad_norm)
            print_summary(f"{axis}={slice_value:g} encoded", grad_encoded_norm)

    if args.volume_res > 0:
        pts = make_volume(args.bound, args.volume_res)
        sdf, grad_norm, grad_encoded_norm = eval_sdf_and_grad_norm(
            f, pts, device, args.chunk,
            label=f"{out_dir.name} volume res={args.volume_res}",
        )
        vol_stem = f"grad_volume_res{args.volume_res}"
        npz_out = out_dir / f"{vol_stem}.npz"
        proj_out = out_dir / f"{vol_stem}_projections.png"
        hist_out = out_dir / f"{vol_stem}_hist.png"
        np.savez_compressed(
            npz_out,
            sdf=sdf.reshape(args.volume_res, args.volume_res, args.volume_res),
            grad_norm=grad_norm.reshape(args.volume_res, args.volume_res, args.volume_res),
            grad_norm_world=grad_norm.reshape(args.volume_res, args.volume_res, args.volume_res),
            grad_norm_encoded=grad_encoded_norm.reshape(args.volume_res, args.volume_res, args.volume_res),
            bound=args.bound, res=args.volume_res,
            ckpt=str(ckpt_path), step=ckpt.get("step", -1),
        )
        save_volume_projection_png(proj_out, grad_norm, args.bound, args.volume_res, args.vmax)
        save_hist_png(hist_out, grad_norm, args.hist_xmax, args.hist_ymax,
                      density=not args.hist_counts)
        pe_hist_out = out_dir / f"{vol_stem}_pe_hist.png"
        save_pe_hist_png(pe_hist_out, grad_norm, grad_encoded_norm,
                         args.hist_xmax, args.hist_ymax,
                         density=not args.hist_counts)
        print(f"saved volume arrays      -> {npz_out}")
        print(f"saved volume projections -> {proj_out}")
        print(f"saved volume histogram   -> {hist_out}")
        print(f"saved PE volume histogram -> {pe_hist_out}")
        print_summary(f"volume res={args.volume_res} world", grad_norm)
        print_summary(f"volume res={args.volume_res} encoded", grad_encoded_norm)
        grads["volume"] = grad_norm
        grads["volume_encoded"] = grad_encoded_norm
        extras["volume_points"] = pts

        radial_out = out_dir / "radial_profile.png"
        save_radial_profile(radial_out, [(out_dir.name, pts, grad_norm)], args.bound)
        print(f"saved radial profile     -> {radial_out}")

    return {"grads": grads, **extras}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True,
                    help="one or more checkpoint paths; with >1, outputs go to per-ckpt subdirs and histograms are overlaid")
    ap.add_argument("--labels", type=str, nargs="+", default=None,
                    help="optional label per --ckpt (default: parent dir name)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--axis", choices=["x", "y", "z", "all"], default="z")
    ap.add_argument("--slice", type=float, nargs="+", default=[0.0], dest="slice_values")
    ap.add_argument("--bound", type=float, default=0.8)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--volume-res", type=int, default=0,
                    help="optional 3D grid resolution for projections/histogram; 0 disables")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--vmax", type=float, default=None,
                    help="heatmap max; default is the 99th percentile")
    ap.add_argument("--hist-xmax", type=float, default=1.5,
                    help="fixed histogram x-axis max for comparable plots")
    ap.add_argument("--hist-ymax", type=float, default=None,
                    help="fixed histogram y-axis max; default lets matplotlib choose")
    ap.add_argument("--hist-counts", action="store_true",
                    help="plot raw counts instead of normalized density")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    axes = ["x", "y", "z"] if args.axis == "all" else [args.axis]

    ckpts: list[Path] = args.ckpt
    if args.labels is not None:
        if len(args.labels) != len(ckpts):
            raise SystemExit(f"--labels count ({len(args.labels)}) must match --ckpt count ({len(ckpts)})")
        labels = args.labels
    else:
        labels = [c.parent.name or c.stem for c in ckpts]

    if len(ckpts) == 1:
        single_out = (
            len(axes) * len(args.slice_values) == 1
            and args.out.suffix.lower() == ".png"
            and args.volume_res == 0
        )
        if single_out:
            out_dir = args.out.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            f, ckpt = load_model(ckpts[0], device)
            save_model_diagnostics(f, out_dir, labels[0])
            axis = axes[0]
            slice_value = args.slice_values[0]
            pts, _, _ = make_slice(axis, slice_value, args.bound, args.res)
            sdf, grad_norm, grad_encoded_norm = eval_sdf_and_grad_norm(
                f, pts, device, args.chunk,
                label=f"slice {axis}={slice_value:g}",
            )
            save_png(args.out, grad_norm, sdf, args.bound, args.res, axis, slice_value, args.vmax)
            pe_png_out = args.out.with_name(f"{args.out.stem}_pe_compare.png")
            save_pe_compare_png(
                pe_png_out, grad_norm, grad_encoded_norm,
                args.bound, args.res, axis, slice_value, args.vmax,
            )
            npz_out = args.out.with_suffix(".npz")
            np.savez_compressed(
                npz_out,
                points=pts,
                sdf=sdf,
                grad_norm=grad_norm,
                grad_norm_world=grad_norm,
                grad_norm_encoded=grad_encoded_norm,
                axis=axis, slice=slice_value, bound=args.bound, res=args.res,
                ckpt=str(ckpts[0]), step=ckpt.get("step", -1),
            )
            print(f"saved heatmap -> {args.out}")
            print(f"saved PE compare heatmap -> {pe_png_out}")
            print(f"saved arrays  -> {npz_out}")
            print_summary(f"{axis}={slice_value:g} world", grad_norm)
            print_summary(f"{axis}={slice_value:g} encoded", grad_encoded_norm)
            return
        process_ckpt(ckpts[0], args.out, args, device, axes)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    per_ckpt: list[dict] = []
    for ckpt_path, label in zip(ckpts, labels):
        sub = args.out / label
        print(f"\n=== {label} ({ckpt_path}) -> {sub} ===")
        per_ckpt.append(process_ckpt(ckpt_path, sub, args, device, axes))

    if args.volume_res > 0 and all("volume" in p["grads"] for p in per_ckpt):
        items = [(label, p["grads"]["volume"]) for label, p in zip(labels, per_ckpt)]
        out_path = args.out / "hist_overlay.png"
        save_hist_overlay_png(
            out_path, items, args.hist_xmax, args.hist_ymax,
            density=not args.hist_counts,
            title="|grad f| distribution",
        )
        print(f"saved overlay histogram -> {out_path}")

        radial_items = [(label, p["volume_points"], p["grads"]["volume"])
                        for label, p in zip(labels, per_ckpt)]
        radial_out = args.out / "radial_overlay.png"
        save_radial_profile(radial_out, radial_items, args.bound)
        print(f"saved radial overlay    -> {radial_out}")

    if args.volume_res > 0 and all("volume_encoded" in p["grads"] for p in per_ckpt):
        encoded_items = [(label, p["grads"]["volume_encoded"]) for label, p in zip(labels, per_ckpt)]
        encoded_out = args.out / "hist_encoded_overlay.png"
        save_hist_overlay_png(
            encoded_out, encoded_items, args.hist_xmax, args.hist_ymax,
            density=not args.hist_counts,
            title=r"$\|\partial f / \partial \gamma\|$ distribution",
        )
        print(f"saved encoded-gradient overlay -> {encoded_out}")

    sigma_per_label = [(label, p.get("spectral_data", [])) for label, p in zip(labels, per_ckpt)]
    if all(items for _, items in sigma_per_label):
        layer_counts = {len(items) for _, items in sigma_per_label}
        if len(layer_counts) == 1:
            sigma_out = args.out / "spectral_overlay.png"
            save_spectral_figure(sigma_out, sigma_per_label)
            print(f"saved spectral overlay  -> {sigma_out}")
        else:
            print(f"skipped spectral overlay: layer count mismatch {layer_counts}")

    sq_per_label = [(label, p.get("sigma_sq_data", [])) for label, p in zip(labels, per_ckpt)]
    if all(items for _, items in sq_per_label):
        layer_counts = {len(items) for _, items in sq_per_label}
        if len(layer_counts) == 1:
            sq_out = args.out / "sigma_estimate_overlay.png"
            save_sigma_estimate_figure(sq_out, sq_per_label)
            print(f"saved sigma-estimate overlay -> {sq_out}")
        else:
            print(f"skipped sigma-estimate overlay: layer count mismatch {layer_counts}")


if __name__ == "__main__":
    main()
