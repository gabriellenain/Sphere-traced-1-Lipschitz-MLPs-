#!/usr/bin/env python
"""Per-scene occupancy: visual-hull %% (as used at training init) vs the
"perfect" %% — the interior fraction (f<0) of the *converged* SDF, evaluated on
the IDENTICAL grid the hull uses (linspace(-bound, bound, res), default 256³ in
a 3×3×3 box).

DTU ships only a holey GT *point cloud* (no watertight volume), so the converged
reconstruction's own solid volume is the tightest available proxy for the true
object. The hull is always an upper bound: hull%% ≥ perfect%% ≥ true%%. A healthy
hull sits modestly above the recon; hull ≫ recon means the hull is bloated.

Outputs a grouped bar chart PNG.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.model import make_model

GRID_TOTAL = None  # filled per res


def discover_runs(roots: list[Path]) -> dict[str, Path]:
    """Latest run per scan that has a checkpoint + an 'occupied voxels' log line."""
    best: dict[str, Path] = {}
    for root in roots:
        for d in sorted(root.glob("run_*scan*"), reverse=True):
            m = re.search(r"(scan\d+)", d.name)
            if not m:
                continue
            scan = m.group(1)
            if scan in best:
                continue
            if not list((d / "ckpt").glob("*.pt")):
                continue
            log = d / "train.log"
            if not log.exists() or "occupied voxels:" not in log.read_text(errors="ignore"):
                continue
            best[scan] = d
    return best


def hull_pct(run: Path, total: int) -> float:
    for line in (run / "train.log").read_text(errors="ignore").splitlines():
        if "occupied voxels:" in line:
            n = int(re.search(r"(\d+)", line.split("occupied voxels:")[1]).group(1))
            return 100.0 * n / total
    return float("nan")


@torch.no_grad()
def perfect_pct(run: Path, res: int, bound: float, device: str, chunk: int) -> float:
    cfg = json.loads((run / "config.json").read_text())["model"]
    ckpts = sorted((run / "ckpt").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    ckpt = torch.load(ckpts[-1], map_location=device)
    f = make_model(**cfg).to(device).eval()
    f.load_state_dict(ckpt["f"])

    lin = torch.linspace(-bound, bound, res, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    inside = 0
    for i in range(0, pts.shape[0], chunk):
        v = f(pts[i:i + chunk].to(device)).squeeze(-1)
        inside += int((v < 0).sum().item())
    step = ckpt.get("step", "?")
    print(f"    {run.name}: step={step} inside={inside} ({100*inside/pts.shape[0]:.3f}%)")
    return 100.0 * inside / pts.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "/scratch/_projets_/willow/1-lip-tracer-new/outputs"])
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--out", default="hull_vs_perfect_occupancy.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))
    total = args.res ** 3

    runs = discover_runs([Path(r) for r in args.roots])
    scans = sorted(runs, key=lambda s: int(re.search(r"\d+", s).group()))
    print(f"device={device}  res={args.res}  scenes={scans}")

    hull, perfect = [], []
    for s in scans:
        print(f"  [{s}] {runs[s].name}")
        hull.append(hull_pct(runs[s], total))
        perfect.append(perfect_pct(runs[s], args.res, args.bound, device, args.chunk))

    # ---- plot ----
    x = np.arange(len(scans)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(scans)), 5))
    b1 = ax.bar(x - w / 2, hull, w, label="Visual hull (training init)", color="#4C78A8")
    b2 = ax.bar(x + w / 2, perfect, w, label='"Perfect" — converged SDF (f<0)', color="#59A14F")
    for bars, vals in ((b1, hull), (b2, perfect)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}%",
                    ha="center", va="bottom", fontsize=8)
    for xi, (h, p) in enumerate(zip(hull, perfect)):
        if p > 0:
            ax.text(xi, max(h, p) * 1.10, f"×{h/p:.1f}", ha="center",
                    fontsize=8, color="#B22222", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(scans)
    ax.set_ylabel(f"% of {args.res}³ grid occupied  (box [±{args.bound}])")
    ax.set_title("DTU occupancy: visual hull (upper bound) vs converged reconstruction\n"
                 "red ×N = hull/recon ratio (≈1 tight, ≫1 bloated)")
    ax.legend(); ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
