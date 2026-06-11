"""ICLR-quality figure from bench_trace_compaction_train.npz.

Two panels:
  A) speedup (dense / compacted) vs batch size B, one line per K. Log-x.
     Dashed hline at speedup=1; shaded "compaction harmful" region below.
     Shows the crossover B at a glance.
  B) wall-clock per train-step trace (ms) vs K at B=B_focus, two lines
     (dense, compacted). Highlights both the K-cut savings AND the
     compaction win in the regime that actually matters for training.

Usage:
    python plot_compaction_iclr.py --npz outputs/bench_compact_train_<job>.npz \\
        --out outputs/compaction_iclr.png --B-focus 16384
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B-focus", type=int, default=16384,
                    help="batch size for the right panel's K-sweep")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    Bs = z["Bs"].astype(int)
    Ks = z["Ks"].astype(int)
    dense_s   = z["dense_s"]    # shape (len(Bs), len(Ks))
    compact_s = z["compact_s"]
    speedup   = z["speedup"]    # dense / compact
    gpu       = str(z["gpu"])

    # ICLR-friendly typography
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

    # --- Panel A: speedup vs B for each K ---
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(len(Ks) - 1, 1)) for i in range(len(Ks))]
    for ki, K in enumerate(Ks):
        ax.plot(Bs, speedup[:, ki], "o-", color=colors[ki],
                markersize=4, label=f"K={K}")
    ylo = min(0.7, float(speedup.min()) * 0.95)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(int(b)) for b in Bs], rotation=0)
    ax.set_xlabel("batch size  $B$  (rays / step)")
    ax.set_ylabel(r"speedup  $t_{\mathrm{dense}}\,/\,t_{\mathrm{compact}}$")
    ax.set_ylim(ylo, max(speedup.max() * 1.05, 1.05))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.4,
              borderpad=0.3)

    # --- Panel B: trace cost vs K at B_focus ---
    ax = axes[1]
    if args.B_focus in Bs.tolist():
        bi = list(Bs).index(args.B_focus)
    else:
        bi = int(np.argmin(np.abs(Bs - args.B_focus)))
        print(f"warning: B={args.B_focus} not in sweep; using B={Bs[bi]}")
    d_ms = dense_s[bi] * 1e3
    c_ms = compact_s[bi] * 1e3
    ax.plot(Ks, d_ms, "s-", color="C3", label="dense", markersize=5)
    ax.plot(Ks, c_ms, "o-", color="C0", label="compacted", markersize=5)
    ax.set_ylim(0, max(d_ms.max(), c_ms.max()) * 1.10)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(int(k)) for k in Ks])
    ax.set_xlabel(r"sphere-tracing iteration budget  $K$")
    ax.set_ylabel(f"trace cost per step  (ms)   [B={Bs[bi]}]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)

    fig.tight_layout(pad=0.6, w_pad=2.2)
    out = Path(args.out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)
    print(f"\n[caption draft]")
    print(f"Sphere-trace cost on {gpu}. (a) Speedup of active-ray compaction "
          f"over a dense full-batch loop, as a function of training batch size $B$. "
          f"Below the crossover compaction is launch-bound and slightly harmful; above, "
          f"its f-eval savings dominate. (b) Per-step trace cost vs iteration budget "
          f"$K$ at $B={Bs[bi]}$: cutting $K$ from 256 to 64 yields ~"
          f"{d_ms[-1]/d_ms[Ks.tolist().index(64)]:.1f}x dense / "
          f"~{c_ms[-1]/c_ms[Ks.tolist().index(64)]:.1f}x compacted reduction, "
          f"justified by the fact that essentially all rays converge by $k\\approx 50$ "
          f"on a trained model.")


if __name__ == "__main__":
    main()
