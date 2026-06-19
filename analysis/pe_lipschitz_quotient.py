"""Feasibility of the PE-normalized 1-Lipschitz SDF: the quotient q(x,y).

The construction  pred(x) = model(gamma_L(x)) / lambda_L  with `model` 1-Lipschitz
in encoded space obeys, for every pair,

    |pred(x) - pred(y)| <= ||gamma_L(x) - gamma_L(y)|| / lambda_L .

To fit the ground-truth SDF d we need |pred(x)-pred(y)| = |d(x)-d(y)|, i.e.

    q(x,y) = lambda_L * |d(x) - d(y)| / ||gamma_L(x) - gamma_L(y)||  <=  1 .

Wherever q>1 NO such model exists -- the encoding has placed x,y too close
(relative to the distance their predictions must span). This script samples many
pairs, computes q, and plots it against the world separation r=||x-y||. The
message: q<=1 for fine pairs (r small) but q grows to ~lambda_L for coarse pairs,
so a PE-normalized 1-Lipschitz field can represent detail finer than ~1/2^(L-1)
but NOT the coarse shape -- exactly why the normalized regression degrades.

Training-free; runs in seconds on CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from fit_gt_sdf import load_mesh, signed_distance_repo_convention


def pe_encode(x: np.ndarray, L: int) -> np.ndarray:
    """gamma_L(x) = [x, sin(2^k x_j), cos(2^k x_j)], k=0..L-1 (repo convention)."""
    freq = 2.0 ** np.arange(L)                      # (L,)
    xb = x[:, None, :] * freq[None, :, None]        # (N, L, 3)
    return np.concatenate(
        [x, np.sin(xb).reshape(len(x), -1), np.cos(xb).reshape(len(x), -1)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, default=Path("data/gt_meshes/happy_buddha_norm.ply"))
    ap.add_argument("--L", type=int, default=6, help="PE bands (k=0..L-1)")
    ap.add_argument("--n-pairs", type=int, default=300_000)
    ap.add_argument("--bound", type=float, default=1.1)
    ap.add_argument("--r-min", type=float, default=1e-3)
    ap.add_argument("--r-max", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("figs/pe_lipschitz_quotient.png"))
    args = ap.parse_args()

    L = args.L
    lam = float(np.sqrt((4.0 ** L + 2.0) / 3.0))     # lambda_L = sqrt((4^L+2)/3)
    rng = np.random.default_rng(args.seed)
    mesh = load_mesh(args.mesh)

    # Base points: half near the surface (multi-scale offset), half uniform volume.
    n = args.n_pairs
    n_surf = n // 2
    surf, fidx = mesh.sample(n_surf, return_index=True)
    surf = surf.astype(np.float32) + (rng.standard_normal((n_surf, 3)).astype(np.float32)
                                      * rng.uniform(0.0, 0.1, (n_surf, 1)).astype(np.float32))
    volp = rng.uniform(-args.bound, args.bound, (n - n_surf, 3)).astype(np.float32)
    x = np.concatenate([surf, volp], 0)

    # Partner at controlled, log-uniform separation r in a random direction.
    r = np.exp(rng.uniform(np.log(args.r_min), np.log(args.r_max), n)).astype(np.float32)
    u = rng.standard_normal((n, 3)).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    y = x + r[:, None] * u

    print(f"L={L}  lambda_L=sqrt((4^{L}+2)/3)={lam:.3f}  pairs={n:,}")
    dx = signed_distance_repo_convention(mesh, x, chunk=100_000, label="d(x)")
    dy = signed_distance_repo_convention(mesh, y, chunk=100_000, label="d(y)")

    gdiff = np.linalg.norm(pe_encode(x, L) - pe_encode(y, L), axis=1)
    q = lam * np.abs(dx - dy) / np.maximum(gdiff, 1e-12)
    # Drop non-finite SDF (degenerate triangles) and exact-zero q (tangent pairs)
    # so the log-log hexbin/quantiles stay clean.
    ok = np.isfinite(q) & (q > 0) & np.isfinite(r)
    r, q = r[ok], q[ok]

    # ---- median / p95 of q in log-r bins ----
    rb = np.geomspace(args.r_min, args.r_max, 41)
    ctr = np.sqrt(rb[:-1] * rb[1:])
    idx = np.digitize(r, rb) - 1
    med = np.array([np.median(q[idx == b]) if np.any(idx == b) else np.nan
                    for b in range(len(ctr))])
    p95 = np.array([np.quantile(q[idx == b], 0.95) if np.any(idx == b) else np.nan
                    for b in range(len(ctr))])
    frac_infeasible = float((q > 1).mean())
    print(f"fraction of pairs with q>1 (infeasible): {frac_infeasible:.3f}")

    # ================= figure (ICLR style) =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.dpi": 200,
        "axes.axisbelow": True,
    })

    y_lo, y_hi = 1.5e-2, max(lam * 1.6, 60.0)
    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    hb = ax.hexbin(r, q, xscale="log", yscale="log", gridsize=70,
                   bins="log", cmap="Blues", mincnt=1)
    ax.axhspan(1.0, y_hi, color="tab:red", alpha=0.06)
    ax.axhline(1.0, color="tab:red", lw=1.6)
    ax.text(args.r_min * 1.2, 1.2, "infeasibility threshold  $q=1$",
            color="tab:red", fontsize=11, va="bottom")
    ax.text(args.r_max * 0.95, y_hi * 0.6,
            rf"$\mathbb{{P}}[\,q(x,y)>1\,]={frac_infeasible:.2f}$",
            color="tab:red", fontsize=12, ha="right", va="top")
    ax.plot(ctr, med, "-", color="tab:orange", lw=2.4, label="median $q$")
    ax.plot(ctr, p95, "--", color="tab:orange", lw=1.4, alpha=0.8, label="95th pct")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(args.r_min, args.r_max)

    ax.set_xlabel(r"pair separation  $r=\|x-y\|$")
    ax.set_ylabel(r"$q(x,y)=\lambda_L\,|d(x)-d(y)|\,/\,\|\gamma_L(x)-\gamma_L(y)\|$")
    ax.legend(loc="upper left")
    cb = fig.colorbar(hb, ax=ax, pad=0.02); cb.set_label("pair count (log)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
