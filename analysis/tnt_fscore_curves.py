#!/usr/bin/env python3
"""Before-vs-after-BA F1(delta) and distance-CDF curves for a TnT scene.

Reuses the distance arrays dumped by tnt_eval/evaluation.py (a *single* fixed
registration per checkpoint). The threshold delta is swept analytically over
those frozen arrays -- alignment does NOT move with delta -- so:

    Precision(delta) = mean(d_rec->gt  < delta)   ==  Pr[d(x, X_gt) < delta]
    Recall(delta)    = mean(d_gt->rec  < delta)
    F1(delta)        = 2 P R / (P + R)

The single-number official F-score is exactly F1(dTau); the curves pass through
your existing before/after points by construction.

Usage:
    python analysis/tnt_fscore_curves.py \
        --before <before_ba_eval_out_dir> \
        --after  <after_ba_eval_out_dir> \
        --scene  Ignatius --dtau <dTau> --out fig_ba_fscore_curves.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pair(d: Path, scene: str):
    """Per-point distances for one eval dir.

    Prefer the raw .npy arrays (dumped by EvaluateHisto). If absent, recompute
    them from the cropped+aligned+downsampled clouds the toolbox already wrote
    (<scene>.precision.ply = reconstruction s, <scene>.recall.ply = GT t) with
    the identical open3d call -- no re-registration, same numbers as the run.
    """
    np_p, np_r = d / f"{scene}.precision.npy", d / f"{scene}.recall.npy"
    if np_p.exists() and np_r.exists():
        return np.load(np_p), np.load(np_r)

    import open3d as o3d
    s = o3d.io.read_point_cloud(str(d / f"{scene}.precision.ply"))  # reconstruction
    t = o3d.io.read_point_cloud(str(d / f"{scene}.recall.ply"))      # GT
    p = np.asarray(s.compute_point_cloud_distance(t))   # rec -> gt (precision)
    r = np.asarray(t.compute_point_cloud_distance(s))   # gt  -> rec (recall)
    return p, r


def curves(p, r, deltas):
    P = (p[None, :] < deltas[:, None]).mean(1)
    R = (r[None, :] < deltas[:, None]).mean(1)
    F = np.where(P + R > 0, 2 * P * R / (P + R), 0.0)
    return P, R, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dtau", type=float, required=True,
                    help="official threshold (marks the operating point)")
    ap.add_argument("--dmax", type=float, default=None,
                    help="max delta on x-axis (default 4*dtau)")
    ap.add_argument("--out", type=Path, default=Path("ba_fscore_curves.pdf"))
    a = ap.parse_args()

    dmax = a.dmax or 4 * a.dtau
    deltas = np.linspace(0.0, dmax, 400)

    pb, rb = load_pair(a.before, a.scene)
    pa, ra = load_pair(a.after, a.scene)
    Pb, Rb, Fb = curves(pb, rb, deltas)
    Pa, Ra, Fa = curves(pa, ra, deltas)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.4))

    # left: F1(delta)
    ax0.plot(deltas, Fb, "--", color="tab:gray", lw=2, label="before BA")
    ax0.plot(deltas, Fa, "-", color="tab:red", lw=2, label="after BA")
    ax0.axvline(a.dtau, color="k", lw=0.8, ls=":")
    ax0.set_xlabel(r"threshold $\delta$")
    ax0.set_ylabel(r"$F_1(\delta)$")
    ax0.set_xlim(0, dmax)
    ax0.set_ylim(0, 1)
    ax0.legend(frameon=False, loc="lower right")

    # right: distance CDF  Pr[d(x,X_gt) < delta]  (precision side)
    ax1.plot(deltas, Pb, "--", color="tab:gray", lw=2, label="before BA")
    ax1.plot(deltas, Pa, "-", color="tab:red", lw=2, label="after BA")
    ax1.axvline(a.dtau, color="k", lw=0.8, ls=":")
    ax1.set_xlabel(r"threshold $\delta$")
    ax1.set_ylabel(r"$\Pr[\,d(x,X_{\mathrm{gt}}) < \delta\,]$")
    ax1.set_xlim(0, dmax)
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")
    i = int(np.argmin(np.abs(deltas - a.dtau)))
    print(f"  @dTau={a.dtau:g}:  F1 before={Fb[i]:.4f}  after={Fa[i]:.4f}")


if __name__ == "__main__":
    main()
