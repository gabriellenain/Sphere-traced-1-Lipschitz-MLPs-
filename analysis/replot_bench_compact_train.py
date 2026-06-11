"""Re-plot bench_compact_train_*.npz with ICLR styling:
  - left panel: active rays per iter (same as before, no annotation chrome)
  - right panel: ms-per-train-step vs batch size B, one line per K
                 (compact solid / dense dashed). Log-log; a slope-1 reference
                 makes the fixed-cost vs compute-bound regimes visually obvious.
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
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    Bs = z["Bs"]
    Ks = z["Ks"]
    dense = z["dense_s"]       # (len(Bs), len(Ks)), seconds
    compact = z["compact_s"]   # (len(Bs), len(Ks)), seconds
    speedup = dense / np.maximum(compact, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    cmap = plt.get_cmap("viridis")

    # ---- left: speedup ratio (dense / compact) vs B, one line per K ----
    ax = axes[0]
    markers = ["o", "s", "^", "D"]
    for ki, K in enumerate(Ks):
        c = cmap(0.15 + 0.7 * (ki / max(len(Ks) - 1, 1)))
        m = markers[ki % len(markers)]
        # thin the topmost line so overlapping curves underneath remain visible
        lw = 1.2 if ki == len(Ks) - 1 else 1.7
        ms = 4.5 if ki == len(Ks) - 1 else 5.5
        ax.plot(Bs, speedup[:, ki], marker=m, ls="-", color=c, lw=lw, ms=ms,
                mfc="white", mew=1.3, label=f"K={K}")
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size $B$ (rays / step)")
    ax.set_ylabel(r"speedup $\;t_{\mathrm{dense}}/t_{\mathrm{compact}}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # ---- right: ms vs B per K, log-log ----
    ax = axes[1]
    for ki, K in enumerate(Ks):
        c = cmap(0.15 + 0.7 * (ki / max(len(Ks) - 1, 1)))
        m = markers[ki % len(markers)]
        lw_c = 1.2 if ki == len(Ks) - 1 else 1.7
        lw_d = 0.9 if ki == len(Ks) - 1 else 1.1
        ms = 4.5 if ki == len(Ks) - 1 else 5.5
        ax.plot(Bs, compact[:, ki] * 1e3, marker=m, ls="-", color=c,
                lw=lw_c, ms=ms, mfc="white", mew=1.3,
                label=f"compact K={K}")
        ax.plot(Bs, dense[:, ki] * 1e3, marker=m, ls="--", color=c,
                lw=lw_d, ms=ms - 0.5, mfc=c, mew=0, alpha=0.85,
                label=f"dense K={K}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size $B$ (rays / step)")
    ax.set_ylabel("wall-clock per train-step trace (ms)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False,
              handlelength=2.4, columnspacing=1.0)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
