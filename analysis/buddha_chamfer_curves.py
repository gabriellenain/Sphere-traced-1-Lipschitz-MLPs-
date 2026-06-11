#!/usr/bin/env python3
"""Chamfer vs width / depth on the Buddha sweep, PE vs no-PE.

Seven panels, PE and no-PE overlaid in each:
  top row    -- 3 panels, Chamfer vs width at fixed depth (4, 8, 16)
  bottom row -- 4 panels, Chamfer vs depth at fixed width (64, 128, 256, 512)

Shows the headline result: no-PE Chamfer improves monotonically with capacity,
while PE Chamfer *degrades* as width grows (the high-frequency encoding overfits
the SDF values and corrupts the zero-level-set).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("outputs/buddha_sweep")
C_NOPE, C_PE = "#3b6fb5", "#d2691e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="chamfer",
                    help="results.csv column to plot")
    ap.add_argument("--results", type=Path, default=SWEEP / "results.csv")
    ap.add_argument("--out", type=Path, default=Path("figs/buddha_chamfer_curves.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.results)))
    # table[encoding][(width, depth)] = metric
    table: dict[str, dict[tuple[int, int], float]] = {"none": {}, "pe": {}}
    for r in rows:
        table[r["encoding"]][(int(r["width"]), int(r["depth"]))] = float(r[args.metric])
    widths, depths = [64, 128, 256, 512], [4, 8, 16]

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.25, which="both")

    def curve(ax, xs, get):
        for enc, col, mark, ls, lab in (("none", C_NOPE, "o", "--", "no-PE"),
                                        ("pe", C_PE, "s", "-", "PE")):
            ys = [table[enc].get(get(x), np.nan) for x in xs]
            ax.plot(xs, ys, ls, color=col, marker=mark, ms=6, lw=1.8, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.legend(fontsize=8, frameon=False)

    # top row: Chamfer vs width, one panel per fixed depth
    for j, d in enumerate(depths):
        ax = axes[0, j]
        curve(ax, widths, lambda w, d=d: (w, d))
        ax.set_title(f"fixed depth = {d}", fontsize=10)
        ax.set_xlabel("width"); ax.set_ylabel(args.metric)
    axes[0, 3].axis("off")

    # bottom row: Chamfer vs depth, one panel per fixed width
    for j, w in enumerate(widths):
        ax = axes[1, j]
        curve(ax, depths, lambda d, w=w: (w, d))
        ax.set_title(f"fixed width = {w}", fontsize=10)
        ax.set_xlabel("depth"); ax.set_ylabel(args.metric)

    # shared y-axis across every panel so heights are comparable at a glance
    allv = [float(r[args.metric]) for r in rows]
    ylo, yhi = min(allv) * 0.92, max(allv) * 1.08
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], *axes[1, :]]:
        ax.set_ylim(ylo, yhi)

    fig.suptitle(f"Happy Buddha SDF regression -- {args.metric} vs capacity, "
                 "PE vs no-PE  (lower is better)", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
