"""Re-plot bench_compact_*.npz with cleaner ICLR styling:
  - left panel: active rays per iter, no dashed annotations, no title chrome
  - right panel: dense vs compact bars with speedup labels neatly aligned
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
    Ks = z["Ks"]
    dense_t = z["dense_s"]
    compact_t = z["compact_s"]
    speedup = z["speedup"]

    iters_axis = np.arange(len(apc))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # ---- left: active-rays curve, no annotations ----
    ax = axes[0]
    ax.plot(iters_axis, apc, lw=1.7, color="C0")
    ax.set_xlabel("sphere-tracing iteration $k$")
    ax.set_ylabel("active rays")
    ax.set_title(f"Active rays per iteration (B = {B:,})")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(apc) - 1)
    ax.set_ylim(0, B * 1.02)

    # ---- right: bars + neatly aligned speedup labels ----
    ax = axes[1]
    width = 0.36
    x = np.arange(len(Ks))
    ax.bar(x - width/2, dense_t * 1e3, width, color="C3", label="dense")
    ax.bar(x + width/2, compact_t * 1e3, width, color="C0", label="compacted")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel("wall-clock per pass (ms)")
    ax.set_title("Measured trace cost vs iteration budget")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # Speedup labels: anchored at the top-right corner of each red (dense) bar,
    # extending slightly right + up so the label sits in the whitespace just
    # outside the bar without overlapping the compact bar next to it.
    max_ms = (dense_t * 1e3).max()
    ax.set_ylim(0, max_ms * 1.10)
    for xi, dt, sp in zip(x, dense_t * 1e3, speedup):
        ax.text(xi - width/2 + width/2 + 0.005,  # right edge of red bar = xi (small nudge for breathing room)
                dt, f"{sp:.1f}×",
                ha="left", va="bottom", fontsize=11,
                fontweight="bold", color="0.15")

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
