"""ICLR-quality per-step time breakdown from a run's profile/phases.csv.

Aggregates mean ms per profiled phase, computes the residual (unprofiled "other"
= step_wall - sum(phases)), and renders a horizontal stacked bar + a side table
of absolute ms and percentages. Default scope: the last 20% of logged steps,
so transient init/warmup doesn't skew the headline numbers.

Usage:
    python plot_step_breakdown.py \\
        --csv outputs/<run>/profile/phases.csv \\
        --out outputs/step_breakdown.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
    cols = rows[0].keys()
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}

PHASE_LABELS = {
    "trace_ms":     "sphere trace",
    "photo_ncc_ms": "photo + NCC",
    "backward_ms":  "backward",
    "opt_step_ms":  "optimizer step",
    "idr_mask_ms":  "IDR mask",
    "eikonal_ms":   "eikonal",
    "mvs_sdf_ms":   "MVS-SDF",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to profile/phases.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--last-frac", type=float, default=0.2,
                    help="fraction of (final) rows to average over (default 0.2)")
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="phases below this %% of step time are merged into 'other'")
    args = ap.parse_args()

    df = load_csv(args.csv)
    n = len(df["step"])
    cut = max(1, int(n * (1 - args.last_frac)))
    df = {k: v[cut:] for k, v in df.items()}
    print(f"averaging over {len(df['step'])} rows (steps {int(df['step'][0])}"
          f" – {int(df['step'][-1])})")

    phases_present = [c for c in PHASE_LABELS if c in df]
    means = {PHASE_LABELS[c]: float(df[c].mean()) for c in phases_present}
    wall = float(df["step_wall_ms"].mean())
    profiled = sum(means.values())
    other = max(wall - profiled, 0.0)

    parts = list(means.items()) + [("other (data, mask, sfm, …)", other)]
    # merge tiny slices
    big   = [(k, v) for k, v in parts if 100 * v / wall >= args.min_share]
    small = [(k, v) for k, v in parts if 100 * v / wall <  args.min_share]
    if small:
        big.append(("other (data, mask, sfm, …)",
                    sum(v for _, v in small) +
                    next((v for k, v in big if k.startswith("other")), 0.0)))
        big = [(k, v) for k, v in big if not (k.startswith("other") and v == 0)]
        # dedupe — keep one 'other' entry summing all "other" contributors
        seen_other = False
        merged = []
        other_sum = 0.0
        for k, v in big:
            if k.startswith("other"):
                other_sum += v
                seen_other = True
            else:
                merged.append((k, v))
        if seen_other:
            merged.append(("other (data, mask, sfm, …)", other_sum))
        big = merged

    big.sort(key=lambda kv: -kv[1])
    labels = [k for k, _ in big]
    values = np.array([v for _, v in big])
    pct = 100 * values / wall

    print(f"\n  phase                              ms     %")
    print(f"  --------------------------------------------")
    for k, v, p in zip(labels, values, pct):
        print(f"  {k:<32s} {v:7.2f}  {p:5.1f}")
    print(f"  --------------------------------------------")
    print(f"  step wall (mean)                 {wall:7.2f}  100.0")

    # ICLR typography
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9, "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7.8, 2.2))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]

    left = 0.0
    for lbl, v, p, c in zip(labels, values, pct, colors):
        ax.barh(0, v, left=left, color=c, edgecolor="white", linewidth=0.8)
        # in-bar text if slice wide enough, else outside
        if p >= 4.0:
            ax.text(left + v / 2, 0, f"{lbl}\n{p:.1f}%",
                    ha="center", va="center", fontsize=8.8, color="white",
                    fontweight="bold")
        left += v

    ax.set_xlim(0, wall)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"time per training step  (ms)   —   mean wall = {wall:.1f} ms")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # legend below for any slice that didn't get inline label
    handles, lbls = [], []
    for lbl, v, p, c in zip(labels, values, pct, colors):
        if p < 4.0:
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=c, ec="white"))
            lbls.append(f"{lbl}  ({p:.1f}%, {v:.2f} ms)")
    if handles:
        ax.legend(handles, lbls, loc="upper center",
                  bbox_to_anchor=(0.5, -1.4), ncol=min(len(lbls), 3),
                  frameon=False, handlelength=1.4)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.45)
    out = Path(args.out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)
    print(f"saved {out.with_suffix('.pdf')}", flush=True)

    # caption draft
    trace_pct = next((p for k, p in zip(labels, pct) if "trace" in k.lower()), 0)
    print(f"\n[caption draft] "
          f"Per-step wall-clock breakdown averaged over the final "
          f"{int(args.last_frac*100)}% of training steps "
          f"({int(df['step'][0])}–{int(df['step'][-1])}). "
          f"Sphere tracing accounts for {trace_pct:.1f}% of step time; "
          f"photo+NCC dominates the remainder.")


if __name__ == "__main__":
    main()
