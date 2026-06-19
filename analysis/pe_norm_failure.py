#!/usr/bin/env python3
"""Why per-band and uniform PE normalization fail for 1-Lipschitz SDFs.

Three rigorous panels, no pairwise-feasibility hand-waving:

  (A) Per-band gradient budget  [ANALYTIC, exact, width/depth-independent].
      A 1-Lipschitz MLP g composed with an encoding gamma obeys, in world space,
          |d/dx_j (g o gamma)|  <=  || d gamma / d x_j ||  =: b_k   per band k.
      So b_k is the *maximum world-space slope* each band can ever deliver. An
      SDF needs |grad d| = 1 everywhere (eikonal). The bars show how each scheme
      allocates that achievable-slope budget across {passthrough, band 0..L-1}:
        free PE   : b_pass=1,            b_k = 2^k                (sum^2 = K^2, NOT 1-Lip)
        uniform   : b_pass=1/K,          b_k = 2^k / K            (K=sqrt((4^L+2)/3))
        per_band  : b_pass=1/sqrt(L+1),  b_k = 1/sqrt(L+1)
      uniform starves the coarse channel (b_pass=1/K~0.03 << 1 -> cannot rise to
      the surface); per_band starves the fine channels (b_top ~ 2^-(L-1)/sqrt(L+1)
      -> detail attenuated). Both are forced by the 1-Lipschitz constraint.

  (B) Trained |grad f| near the surface  [EMPIRICAL, from the checkpoints].
      Autograd gradient norm of each trained field on near-surface samples.
      Confirms the bound bites: uniform's field is flat (|grad f| << 1, fails the
      eikonal condition), free PE overshoots (>1, the sphere-trace problem),
      per_band sits below 1, plain no-PE concentrates at 1.

  (C) SDF error spectrum  [EMPIRICAL, from the checkpoints + GT].
      Radially-averaged power spectrum of the error field (pred - GT) on a grid.
      Localizes *where in frequency* each scheme is wrong: uniform's error is
      low-frequency (global shape), per_band's is high-frequency (detail).

Run with the env that has torch + trimesh + rtree, e.g.
  /scratch/_projets_/willow/1-lip-tracer/.venv/bin/python analysis/pe_norm_failure.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from lip_tracer.model import make_model
from fit_gt_sdf import load_mesh, signed_distance_repo_convention

SWEEP = Path("outputs/buddha_sweep")
GT_MESH = Path("data/gt_meshes/happy_buddha_norm.ply")

# label -> (run subdir, colour)
RUNS = [
    ("no-PE",            "none_W256_D16_1M",          "#444444"),
    ("free PE L=6",      "pe_W256_D16",               "#1f77b4"),
    ("PE per-band",      "pe6_per_band_W256_D16_1M",  "#ff7f0e"),
    ("PE uniform",       "pe6_uniform_W256_D16_1M",   "#d62728"),
]


# --------------------------------------------------------------------------- #
#  model loading
# --------------------------------------------------------------------------- #
def load_field(sub: str, device: str) -> torch.nn.Module:
    ck = torch.load(SWEEP / sub / "checkpoint_gt_sdf.pt", map_location=device)
    hidden = ck["f"]["head_weight"].shape[0]
    f = make_model(
        hidden=hidden,
        depth=ck.get("depth", 16),
        group_size=ck.get("group_size", 2),
        activation=ck.get("activation", "groupsort"),
        input_encoding=ck.get("input_encoding", "identity"),
        multires=ck.get("multires", 0),
        architecture=ck.get("architecture", "cpl"),
        lipschitz_mode=ck.get("lipschitz_mode", "none"),
    ).to(device)
    f.load_state_dict(ck["f"])
    f.eval()
    return f


def eval_sdf(f: torch.nn.Module, pts: np.ndarray, device: str,
             chunk: int = 200_000) -> np.ndarray:
    out = np.empty(len(pts), np.float32)
    with torch.no_grad():
        for i in range(0, len(pts), chunk):
            x = torch.from_numpy(pts[i:i + chunk]).to(device)
            out[i:i + chunk] = f(x).detach().cpu().numpy()
    return out


def grad_norm(f: torch.nn.Module, pts: np.ndarray, device: str,
              chunk: int = 100_000) -> np.ndarray:
    out = np.empty(len(pts), np.float32)
    for i in range(0, len(pts), chunk):
        x = torch.from_numpy(pts[i:i + chunk]).to(device).requires_grad_(True)
        y = f(x).sum()
        g, = torch.autograd.grad(y, x, create_graph=False)
        out[i:i + chunk] = g.norm(dim=-1).detach().cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
#  panel A  (analytic)
# --------------------------------------------------------------------------- #
def panel_budget(ax, L: int) -> None:
    K = float(np.sqrt((4.0 ** L + 2.0) / 3.0))
    bands = ["pass"] + [f"$2^{k}$" for k in range(L)]
    free = np.array([1.0] + [2.0 ** k for k in range(L)])
    unif = free / K
    perb = np.full(L + 1, 1.0 / np.sqrt(L + 1))

    x = np.arange(L + 1)
    w = 0.27
    ax.bar(x - w, free, w, label=f"free PE  (Lip={K:.1f})", color="#1f77b4")
    ax.bar(x,     unif, w, label="uniform  (Lip=1)",        color="#d62728")
    ax.bar(x + w, perb, w, label="per-band (Lip=1)",        color="#ff7f0e")
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.text(L - 0.4, 1.15, "eikonal target |∇d|=1", fontsize=8, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_xlabel("encoding channel (passthrough / frequency band)")
    ax.set_ylabel("achievable world-space slope  $b_k=\\|\\partial\\gamma/\\partial x_j\\|$")
    ax.set_title("(A) Per-band gradient budget  [analytic]", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.annotate("uniform starves\nthe coarse channel\n($b_{pass}=1/K$)",
                xy=(0 - w * 0.0, unif[0]), xytext=(0.7, unif[0] * 0.12),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
    ax.annotate("per-band starves\nthe fine channels",
                xy=(L + w, perb[-1]), xytext=(L - 2.3, perb[-1] * 0.12),
                fontsize=8, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.0))


# --------------------------------------------------------------------------- #
#  panel B  (trained gradient norm)
# --------------------------------------------------------------------------- #
def panel_gradnorm(ax, mesh, device: str, n: int, near_std: float,
                   seed: int) -> None:
    rng = np.random.default_rng(seed)
    surf, _ = mesh.sample(n, return_index=True)
    pts = surf.astype(np.float32) + (near_std *
                                     rng.standard_normal((n, 3)).astype(np.float32))
    bins = np.linspace(0.0, 2.0, 121)
    for label, sub, col in RUNS:
        f = load_field(sub, device)
        gn = grad_norm(f, pts, device)
        med = float(np.median(gn))
        ax.hist(gn.clip(0, 2.0), bins=bins, histtype="step", density=True,
                color=col, lw=1.8, label=f"{label}  (median {med:.2f})")
        print(f"  |grad f| {label:14s} median={med:.3f} "
              f"frac<0.5={np.mean(gn < 0.5):.2f} frac>1={np.mean(gn > 1.0):.2f}")
    ax.axvline(1.0, color="k", ls="--", lw=1.0)
    ax.text(1.02, ax.get_ylim()[1] * 0.92, "eikonal |∇f|=1", fontsize=8)
    ax.set_xlabel("trained $\\|\\nabla f\\|$  (near-surface samples)")
    ax.set_ylabel("density")
    ax.set_title("(B) Trained gradient norm  [empirical]", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
#  panel C  (error spectrum)
# --------------------------------------------------------------------------- #
def radial_psd(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = np.fft.fftshift(np.fft.fftn(field))
    power = (F.real ** 2 + F.imag ** 2)
    G = field.shape[0]
    c = G // 2
    ax = np.arange(G) - c
    kx, ky, kz = np.meshgrid(ax, ax, ax, indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).ravel()
    pw = power.ravel()
    kbin = kr.astype(np.int32)
    nb = kbin.max() + 1
    psum = np.bincount(kbin, weights=pw, minlength=nb)
    pcnt = np.bincount(kbin, minlength=nb).clip(min=1)
    return np.arange(nb), psum / pcnt


def panel_spectrum(ax, mesh, device: str, grid: int, bound: float) -> None:
    g = np.linspace(-bound, bound, grid, dtype=np.float32)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    print(f"  computing GT SDF on {grid}^3 grid ...")
    gt = signed_distance_repo_convention(mesh, pts, chunk=200_000, label="gt-grid")
    gt = gt.reshape(grid, grid, grid)
    for label, sub, col in RUNS:
        f = load_field(sub, device)
        pred = eval_sdf(f, pts, device).reshape(grid, grid, grid)
        k, psd = radial_psd(pred - gt)
        # spatial frequency in cycles per unit length
        freq = k / (2.0 * bound)
        m = k > 0
        ax.loglog(freq[m], psd[m], color=col, lw=1.8, label=label)
    ax.set_xlabel("spatial frequency  (cycles / unit)")
    ax.set_ylabel("error power  $|\\widehat{f-d}|^2$  (radial avg)")
    ax.set_title("(C) SDF error spectrum  [empirical]", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--n-grad", type=int, default=200_000)
    ap.add_argument("--near-std", type=float, default=0.01)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--bound", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("figs/pe_norm_failure.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    mesh = load_mesh(GT_MESH)

    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    print("panel A: per-band budget (analytic)")
    panel_budget(axes[0], args.L)
    print("panel B: trained |grad f|")
    panel_gradnorm(axes[1], mesh, device, args.n_grad, args.near_std, args.seed)
    print("panel C: SDF error spectrum")
    panel_spectrum(axes[2], mesh, device, args.grid, args.bound)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
