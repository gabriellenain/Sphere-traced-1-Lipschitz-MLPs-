#!/usr/bin/env python3
"""Distribution of convergence iteration for rays missed at a short budget.

Beyond "how many converge with more budget", this answers *why* the rest do
not, by recording — for every missed ray — whether f ever changes sign along
the ray (a root provably exists → tracer failure, fixable by bracketing) and
the geometry at the closest-approach point (|∇f| and the grazing angle).

Still-missed rays are split into two diagnostic classes:
  has_root : f goes ≤ 0 somewhere → a root exists, the tracer skipped/stalled.
             Recoverable by sign-change bracketing — not an iteration-budget
             problem.
  no_root  : f stays > 0 the whole ray → genuine tangent miss. The surface is
             slightly too thin / inboard; no tracer change recovers these,
             only the silhouette/mask loss.
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
from dataclasses import replace

from lip_tracer.config import TraceConfig
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd
import lip_tracer.data as data_mod
from plot_sphere_trace_steps import rays_for_view


@torch.no_grad()
def bracket_hits(f, origins, dirs, cfg: TraceConfig, chunk: int) -> np.ndarray:
    """Hit mask from the REAL run tracer (trace_nograd → bracketing+Newton on).

    Used by --bracket mode so 'misses' mean what the run actually fails to hit,
    not what a bare un-bracketed march fails to hit. The m≈-1 sign-flip
    oscillators that the bare march reports as 'stalled-has-root' are caught by
    bracketing here, so they drop out of the miss set entirely.
    """
    n = origins.shape[0]
    hit = torch.zeros(n, dtype=torch.bool, device=origins.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, _, h = trace_nograd(f, origins[s:e], dirs[s:e], cfg)
        hit[s:e] = h
    return hit.cpu().numpy()


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


@torch.no_grad()
def trace_until(f, origins, dirs, cfg: TraceConfig, max_iters: int, chunk: int):
    """Vanilla sphere trace, recording per-ray convergence diagnostics.

    Additions over a plain trace:
      ever_neg : did f go strictly negative at any active iteration → a root
                 provably exists between the entry point and that sample.
      t_min    : ray depth at the smallest |f| seen (closest approach).
      ratio    : geometric stall ratio |f_last|/|f_prev| over the final step —
                 plateau value ≈ effective |∇f·d|.
    """
    n = origins.shape[0]
    conv_iter = torch.full((n,), -1, dtype=torch.long, device=origins.device)
    escaped_iter = torch.full((n,), -1, dtype=torch.long, device=origins.device)
    final_abs = torch.full((n,), float("inf"), device=origins.device)
    final_t = torch.zeros(n, device=origins.device)
    min_abs = torch.full((n,), float("inf"), device=origins.device)
    t_at_min = torch.zeros(n, device=origins.device)
    ever_neg = torch.zeros(n, dtype=torch.bool, device=origins.device)
    stall_ratio = torch.full((n,), float("nan"), device=origins.device)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        o, d = origins[s:e], dirs[s:e]
        B = e - s
        t = torch.zeros(B, device=origins.device)
        conv = torch.zeros(B, dtype=torch.bool, device=origins.device)
        escaped = torch.zeros(B, dtype=torch.bool, device=origins.device)
        ci = torch.full((B,), -1, dtype=torch.long, device=origins.device)
        ei = torch.full((B,), -1, dtype=torch.long, device=origins.device)
        ma = torch.full((B,), float("inf"), device=origins.device)
        tm = torch.zeros(B, device=origins.device)
        neg = torch.zeros(B, dtype=torch.bool, device=origins.device)
        prev_abs = torch.full((B,), float("nan"), device=origins.device)
        ratio = torch.full((B,), float("nan"), device=origins.device)

        for i in range(max_iters):
            active = ~(conv | escaped)
            if not active.any():
                break
            x = o + t.unsqueeze(-1) * d
            sdf = f(x)
            abs_sdf = sdf.abs()
            new_min = active & (abs_sdf < ma)
            tm = torch.where(new_min, t, tm)
            ma = torch.where(new_min, abs_sdf, ma)
            neg = neg | (active & (sdf < 0))
            # geometric step ratio (only while genuinely advancing)
            r = abs_sdf / prev_abs.clamp(min=1e-12)
            ratio = torch.where(active & torch.isfinite(prev_abs), r, ratio)
            prev_abs = torch.where(active, abs_sdf, prev_abs)
            just_conv = active & (abs_sdf < cfg.eps)
            just_esc = active & (t >= cfg.t_far)
            ci = torch.where(just_conv & (ci < 0), torch.full_like(ci, i), ci)
            ei = torch.where(just_esc & (ei < 0), torch.full_like(ei, i), ei)
            conv = conv | just_conv
            escaped = escaped | just_esc
            t = t + torch.where(conv | escaped, torch.zeros_like(sdf), sdf)

        fa = f(o + t.unsqueeze(-1) * d).abs()
        final_abs[s:e] = fa
        final_t[s:e] = t
        min_abs[s:e] = ma
        t_at_min[s:e] = tm
        ever_neg[s:e] = neg
        stall_ratio[s:e] = ratio
        conv_iter[s:e] = ci
        escaped_iter[s:e] = ei
        print(f"  traced {e:>{len(str(n))}}/{n}", flush=True)

    return {
        "conv_iter": conv_iter.cpu().numpy(),
        "escaped_iter": escaped_iter.cpu().numpy(),
        "final_abs": final_abs.cpu().numpy(),
        "final_t": final_t.cpu().numpy(),
        "min_abs": min_abs.cpu().numpy(),
        "t_at_min": t_at_min,            # kept on-device for the grad pass
        "ever_neg": ever_neg.cpu().numpy(),
        "stall_ratio": stall_ratio.cpu().numpy(),
    }


def grad_at_points(f, origins, dirs, t, chunk: int):
    """|∇f| and |cos∠(∇f, d)| at o + t·d, for every ray.

    |∇f| < 1 means the 1-Lipschitz field is slack — steps undershoot the true
    distance and even a head-on ray converges only geometrically.
    |cos∠(∇f, d)| → 0 marks a grazing ray (sphere tracing stalls there).
    """
    n = origins.shape[0]
    gnorm = torch.zeros(n, device=origins.device)
    cosang = torch.zeros(n, device=origins.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        o, d, ts = origins[s:e], dirs[s:e], t[s:e]
        x = (o + ts.unsqueeze(-1) * d).detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(x).sum(), x)[0]
        gn = g.norm(dim=-1)
        gnorm[s:e] = gn
        cosang[s:e] = ((g * d).sum(-1) / gn.clamp(min=1e-9)).abs()
    return gnorm.cpu().numpy(), cosang.cpu().numpy()


# class palette, shared across panels:
#   recovered  → green   (the recovered-rays histogram colour)
#   has root   → amber   (still-missed, root exists → tracer-fixable)
#   no root    → red     (still-missed, no root → escapes to t_far)
C_RECOV, C_ROOT, C_NOROOT = "#4daf4a", "#fdae61", "#d73027"


def _dilate(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """Square dilation by ±k pixels so sparse miss dots read at print size."""
    out = mask.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


@torch.no_grad()
def trace_signed(f, o, d, iters: int):
    """Sphere trace WITHOUT a convergence stop — record signed f per step.

    Lets the stalled rays show their oscillation: they step t ← t + f(t) with
    a signed f, so an overshoot flips the sign and the ray ping-pongs.
    """
    B = o.shape[0]
    sdf_hist = torch.zeros(iters, B, device=o.device)
    t_hist = torch.zeros(iters, B, device=o.device)
    t = torch.zeros(B, device=o.device)
    for i in range(iters):
        sdf = f(o + t.unsqueeze(-1) * d)
        sdf_hist[i] = sdf
        t_hist[i] = t
        t = t + sdf
    return sdf_hist.cpu().numpy(), t_hist.cpu().numpy()


def _grad_dot_d(f, o, d, t_star, chunk: int = 16384):
    """Signed directional derivative ∇f·d at o + t_star·d."""
    n = o.shape[0]
    out = np.zeros(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x = (o[s:e] + t_star[s:e, None] * d[s:e]).clone().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(x).sum(), x)[0]
        out[s:e] = (g * d[s:e]).sum(-1).cpu().numpy()
    return out


def stalled_diagnosis(f, o, d, iters: int = 300):
    """For the stalled-has-root rays: measured step multiplier and ∇f·d.

    Sphere tracing is the iteration t ← t + f(t); near a root the error obeys
    e_{k+1} = (1 + ∇f·d)·e_k, so it converges only when |1 + ∇f·d| < 1. These
    rays sit at ∇f·d ≈ −2 (PE inflates |∇f| past 1), so the multiplier reaches
    −1 and the residual oscillates forever instead of decaying.
    """
    if o.shape[0] == 0:
        return {"ndotd": np.zeros(0), "mult": np.zeros(0)}
    sdf_hist, t_hist = trace_signed(f, o, d, iters)
    warm = min(40, iters // 3)
    tail = sdf_hist[warm:]
    ratios = tail[1:] / np.where(np.abs(tail[:-1]) < 1e-9, np.nan, tail[:-1])
    mult = np.nanmedian(np.clip(ratios, -3.0, 3.0), axis=0)
    k_star = np.argmin(np.abs(sdf_hist), axis=0)
    t_star = torch.from_numpy(t_hist[k_star, np.arange(o.shape[0])]).to(o.device)
    ndotd = _grad_dot_d(f, o, d, t_star)
    return {"ndotd": ndotd, "mult": mult}


def make_figure(long, short_iters, max_iters, eps, gnorm, cosang,
                img, cls_map, stalled, out_path: Path):
    conv_iter = long["conv_iter"]
    min_abs = long["min_abs"]
    ever_neg = long["ever_neg"]
    conv = conv_iter >= 0
    still = ~conv
    has_root = still & ever_neg          # tracer-fixable (root exists)
    no_root = still & ~ever_neg          # genuine tangent miss

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.4),
                             gridspec_kw={"width_ratios": [1.3, 1, 1.15, 1.05]})

    # --- panel 0: where the misses are, coloured by fate --------------------
    axes[0].imshow(img * 0.45 + 0.04)          # dimmed so the overlay pops
    overlay = np.zeros((*cls_map.shape, 4), dtype=np.float32)
    for cid, hexc in [(0, C_RECOV), (1, C_ROOT), (2, C_NOROOT)]:
        rgb = matplotlib.colors.to_rgb(hexc)
        overlay[_dilate(cls_map == cid)] = (*rgb, 0.95)
    axes[0].imshow(overlay)
    axes[0].axis("off")
    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=8, color=C_RECOV,
                   label=f"recovered with more iters ({int(conv.sum())})"),
        plt.Line2D([], [], marker="s", ls="", ms=8, color=C_ROOT,
                   label=f"stalled — root exists ({int(has_root.sum())})"),
        plt.Line2D([], [], marker="s", ls="", ms=8, color=C_NOROOT,
                   label=f"no root — escapes to $t_{{far}}$ ({int(no_root.sum())})"),
    ]
    axes[0].legend(handles=handles, fontsize=7.2, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.21))

    # --- panel 1: convergence-iteration distribution of recovered rays ------
    # rays that miss the 24-step budget but converge with more iterations.
    if conv.any():
        last_recov = int(conv_iter[conv].max())
        bins = np.arange(short_iters, last_recov + 2) - 0.5
        axes[1].hist(conv_iter[conv], bins=bins, color=C_RECOV, alpha=0.9)
        axes[1].set_xlim(short_iters, last_recov + 1)
        span = last_recov - short_iters
        step = next(s for s in (10, 25, 50, 100) if span / s <= 4)
        first = ((short_iters + step) // step) * step
        ticks = [short_iters] + list(range(first, last_recov + 1, step))
        axes[1].set_xticks(ticks)
    else:
        # no ray recovers beyond the budget → all misses are genuine
        # (tangent / stalled), not iteration-limited. Say so explicitly.
        axes[1].text(0.5, 0.5,
                     f"0 rays recover\nbeyond {short_iters} iters\n"
                     f"(all misses are\ntangent / stalled)",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=8.5, color="0.35")
        axes[1].set_xlim(short_iters, short_iters + 1)
        axes[1].set_xticks([short_iters])
    axes[1].set_xlabel("Iteration of first convergence for recovered points")
    axes[1].set_ylabel("Recovered-ray count")
    axes[1].grid(True, alpha=0.2)

    # --- panel 2: characteristics of the no-root rays (escape to t_far) -----
    # x = |cos∠(∇f,d)| (small = grazing), y = |∇f| (<1 slack Lipschitz field),
    # colour = how close the ray got to the surface (best |f|/ε).
    if no_root.any():
        sc = axes[2].scatter(
            cosang[no_root], gnorm[no_root], s=12,
            c=np.log10(np.maximum(min_abs[no_root] / eps, 1e-3)),
            cmap="viridis", alpha=0.8, edgecolors="none")
        cb = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.02)
        cb.set_label(r"best $\log_{10}(|f|/\epsilon)$")
    axes[2].axhline(1.0, color="0.4", ls="--", lw=0.8)
    if no_root.any():
        med_cos = float(np.median(cosang[no_root]))
        axes[2].axvline(med_cos, color="#222222", ls="-", lw=1.3)
        axes[2].text(med_cos + 0.03, 0.96, f"median |cos| = {med_cos:.2f}",
                     transform=axes[2].get_xaxis_transform(), fontsize=7.2,
                     color="#222222", va="top")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, max(1.15, float(np.nanmax(gnorm[no_root])) * 1.05) if no_root.any() else 1.15)
    axes[2].set_xlabel(r"$|\cos\angle(\nabla f, d)|$  for escaped ($t_{far}$) points")
    axes[2].set_ylabel(r"$|\nabla f|$ at closest approach")
    axes[2].grid(True, alpha=0.2)

    # --- panel 3: why the stalled-has-root rays never converge --------------
    # sphere tracing t←t+f(t) ⇒ e_{k+1}=(1+∇f·d)·e_k; converges iff |1+∇f·d|<1.
    nd, mu = stalled["ndotd"], stalled["mult"]
    xs = np.linspace(-2.5, -0.5, 80)
    axes[3].axhspan(-1.0, 1.0, color=C_RECOV, alpha=0.13)
    axes[3].plot(xs, 1.0 + xs, color="0.35", lw=1.4, ls="--",
                 label=r"theory $m=1+\nabla f\!\cdot\!d$")
    axes[3].axhline(-1.0, color=C_NOROOT, lw=1.0)
    axes[3].axhline(1.0, color=C_NOROOT, lw=1.0)
    if len(nd):
        axes[3].scatter(nd, mu, s=15, c=C_ROOT, edgecolors="0.3", linewidths=0.4,
                        zorder=3, label=f"stalled rays ({len(nd)})")
    axes[3].text(0.5, 0.93, "convergent band  |m| < 1", transform=axes[3].transAxes,
                 ha="center", fontsize=7.2, color="#2f7d32")
    axes[3].set_xlim(-2.5, -0.5)
    axes[3].set_ylim(-2.0, 1.3)
    axes[3].set_xlabel(r"$\nabla f\cdot d$ at closest approach")
    axes[3].set_ylabel(r"measured step multiplier $f_{k+1}/f_k$")
    axes[3].legend(fontsize=7, frameon=False, loc="lower left")
    axes[3].grid(True, alpha=0.2)

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
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--out-dir", type=Path, default=Path("figs/sphere_trace_steps"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--crop-fg", action="store_true",
                    help="crop panel-0 to the foreground bbox (e.g. MVMannequin, "
                         "where the object fills ~9%% of the frame)")
    ap.add_argument("--bracket", action="store_true",
                    help="define misses with the real run tracer (trace_nograd, "
                         "bracketing+Newton on) instead of the bare march — the map "
                         "then shows only the run's TRUE residual (no-root/tangent), "
                         "since bracketing already converts the stalled-has-root rays.")
    args = ap.parse_args()

    f, cfg, scene, ckpt_path = load_run(args.run_dir, args.ckpt, args.device)
    data_mod.SCENE = scene
    views = data_mod.load_views(scene, down=args.down)
    vi = args.view % int(views["images"].shape[0])
    origins, dirs, img, mask, H, W = rays_for_view(views, vi, args.device)
    print(f"run={args.run_dir.name} view={vi} down={args.down} eps={cfg.eps:g}")

    fg = mask.reshape(-1)
    if args.bracket:
        # Run-faithful: a "miss" is what the actual run tracer fails to hit.
        cfg_short = replace(cfg, iters=args.short_iters)
        hit_short = bracket_hits(f, origins, dirs, cfg_short, args.chunk)
        miss_idx = np.where(fg & ~hit_short)[0]
        print(f"{args.short_iters}-step foreground misses (BRACKETED run tracer): "
              f"{len(miss_idx)} / {int(fg.sum())}")
    else:
        short = trace_until(f, origins, dirs, cfg, args.short_iters, args.chunk)
        miss_idx = np.where(fg & (short["conv_iter"] < 0))[0]
        print(f"{args.short_iters}-step foreground misses: {len(miss_idx)} / {int(fg.sum())}")
    miss_origins = origins[torch.from_numpy(miss_idx).to(origins.device)]
    miss_dirs = dirs[torch.from_numpy(miss_idx).to(dirs.device)]
    long = trace_until(f, miss_origins, miss_dirs, cfg, args.max_iters, args.chunk)

    # closest-approach geometry for the missed rays
    gnorm, cosang = grad_at_points(f, miss_origins, miss_dirs, long["t_at_min"], args.chunk)

    still = long["conv_iter"] < 0
    has_root = int((still & long["ever_neg"]).sum())
    no_root = int((still & ~long["ever_neg"]).sum())
    print(f"still-missed: has_root(tracer-fixable)={has_root}  no_root(tangent)={no_root}")

    # per-pixel class map for the image panel:
    #   -1 background · 0 recovered · 1 stalled-has-root · 2 no-root
    miss_cls = np.full(len(miss_idx), 2, dtype=np.int64)
    miss_cls[still & long["ever_neg"]] = 1
    miss_cls[~still] = 0
    cls_map = np.full(H * W, -1, dtype=np.int64)
    cls_map[miss_idx] = miss_cls
    cls_map = cls_map.reshape(H, W)

    # stalled-has-root rays: measured step multiplier and ∇f·d
    amber_t = torch.from_numpy(still & long["ever_neg"]).to(origins.device)
    stalled = stalled_diagnosis(f, miss_origins[amber_t], miss_dirs[amber_t])
    if len(stalled["mult"]):
        print(f"stalled rays: ∇f·d median {np.median(stalled['ndotd']):+.3f}  "
              f"multiplier median {np.median(stalled['mult']):+.3f}  "
              f"|m|≥1: {(np.abs(stalled['mult']) >= 1).mean()*100:.0f}%")

    # zoom panel 0 to the object (object fills only ~9% of an MVMannequin frame)
    if args.crop_fg:
        ys_fg, xs_fg = np.where(mask)
        if ys_fg.size:
            pad = 0.12
            r0, r1, c0, c1 = ys_fg.min(), ys_fg.max(), xs_fg.min(), xs_fg.max()
            dr, dc = int((r1 - r0) * pad) + 1, int((c1 - c0) * pad) + 1
            r0, r1 = max(r0 - dr, 0), min(r1 + dr, H - 1) + 1
            c0, c1 = max(c0 - dc, 0), min(c1 + dc, W - 1) + 1
            img = img[r0:r1, c0:c1]
            cls_map = cls_map[r0:r1, c0:c1]
            print(f"crop-fg panel0 → rows[{r0}:{r1}] cols[{c0}:{c1}]")

    stem = f"{args.run_dir.name}_view{vi:02d}_down{args.down}_miss_convergence_to{args.max_iters}"
    if args.bracket:
        stem += "_bracket"
    out_png = args.out_dir / f"{stem}.png"
    make_figure(long, args.short_iters, args.max_iters, cfg.eps, gnorm, cosang,
                img, cls_map, stalled, out_png)
    summary = {
        "run_dir": str(args.run_dir),
        "ckpt": str(ckpt_path),
        "view": vi,
        "down": args.down,
        "eps": cfg.eps,
        "short_iters": args.short_iters,
        "max_iters": args.max_iters,
        "bracket": bool(args.bracket),
        "foreground_rays": int(fg.sum()),
        "misses_short": int(len(miss_idx)),
        "H": int(H), "W": int(W),
        "miss_idx": miss_idx.astype(int).tolist(),
        "recovered_by_max": int((long["conv_iter"] >= 0).sum()),
        "still_missed": int(still.sum()),
        "still_has_root": has_root,
        "still_no_root": no_root,
        "escaped": int((long["escaped_iter"] >= 0).sum()),
        "stalled_ndotd": stalled["ndotd"].tolist(),
        "stalled_mult": stalled["mult"].tolist(),
        "conv_iter": long["conv_iter"].tolist(),
        "escaped_iter": long["escaped_iter"].tolist(),
        "min_abs_over_eps": (long["min_abs"] / cfg.eps).tolist(),
        "ever_neg": long["ever_neg"].astype(int).tolist(),
        "grad_norm_at_min": gnorm.tolist(),
        "cos_angle_at_min": cosang.tolist(),
        "stall_ratio": long["stall_ratio"].tolist(),
    }
    out_json = args.out_dir / f"{stem}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"saved {out_png}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
