"""Re-plot trace_tail_*.npz with ICLR-friendly panels:
  left  : active rays per iter (linear)
  right : cumulative cost — full-batch vs compacted — as a function of max-iter K
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
    apc = z["active_per_iter"]
    B = int(z["B"])
    iters = int(z["iters"])
    views = z["views"].tolist()
    K = np.arange(1, iters + 1)
    cum_compact = np.cumsum(apc)
    cum_full = K * B
    speedup = cum_full / np.maximum(cum_compact, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- left : active rays per iter ----
    ax = axes[0]
    ax.plot(np.arange(iters), apc, lw=1.6, color="C0")
    ax.set_xlabel("sphere-tracing iteration $k$")
    ax.set_ylabel("active rays")
    ax.set_title(f"Active rays per iteration (B = {B:,})")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, iters - 1)
    ax.set_ylim(0, B * 1.02)
    # annotate inflection points
    for thr, color in zip((0.5, 0.9, 0.99), ("C2", "C1", "C3")):
        target = B * (1 - thr)
        idx = np.where(apc <= target)[0]
        if len(idx):
            k = int(idx[0])
            ax.axvline(k, color=color, lw=0.8, ls="--", alpha=0.7)
            ax.text(k + 2, B * 0.92, f"{int(thr*100)}% done @ k={k}",
                    color=color, fontsize=8.5)

    # ---- right : cumulative MLP evaluations + speedup ----
    ax = axes[1]
    ax.plot(K, cum_full / 1e6, lw=1.8, color="C3", label="full-batch ($K \\cdot B$)")
    ax.fill_between(K, cum_compact / 1e6, cum_full / 1e6,
                    color="C3", alpha=0.15, label="wasted MLP evals")
    ax.plot(K, cum_compact / 1e6, lw=1.8, color="C0",
            label=r"compacted ($\sum_k |\mathrm{active}_k|$)")
    ax.set_xlabel("max iterations $K$")
    ax.set_ylabel("cumulative MLP evaluations  (millions)")
    ax.set_title("Cost vs trace budget")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, iters - 1)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # twin axis: speedup ratio
    ax2 = ax.twinx()
    ax2.plot(K, speedup, lw=1.3, color="0.35", ls=":", label="speedup")
    ax2.set_ylabel("speedup = full / compacted", color="0.35")
    ax2.tick_params(axis="y", colors="0.35")
    final = speedup[-1]
    ax2.annotate(f"{final:.1f}× at K={iters}",
                 xy=(iters - 1, final), xytext=(iters * 0.55, final * 0.55),
                 color="0.25", fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))

    fig.suptitle(f"DTU scan65 — checkpoint 270k — views {views}", y=1.02, fontsize=11)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
