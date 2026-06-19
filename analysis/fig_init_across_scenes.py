#!/usr/bin/env python3
"""ICLR figure: two ways to seed the 1-Lipschitz network on DTU, across scenes.

Per scene, side by side:
  (left)  COLMAP *sparse* cloud  ->  eps-ball@floor seed  (ours; no dense MVS).
          The winning method from fig_colmap_init_compare: union of eps-balls
          (eps = 3x voxel, <=4 components) -> border flood-fill -> EDT -> regress.
          Input = sparse_sfm_points.txt only, so it runs on every scene.
  (right) Sphere + MVSFormer++ depth-carve  (dense-MVS init).
          A bounding sphere voxel-carved by MVSFormer++ depth votes. Needs the
          dense per-view depth maps precomputed (heavy GPU); only scan40 & scan122
          have them, so the other scenes show the seed the sparse route still gives.

All meshes are rendered with the same CPU clay renderer (analysis/clay_render.py)
for an apples-to-apples look. Stats come from each run's fit.json / summary.json.
Output: analysis/figures/init_across_scenes.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clay_render import clay_render

OUTROOT = Path("outputs")  # -> /scratch .../Sphere-traced-outputs
DIAG = Path("_diagnostics")
OUT = Path("analysis/figures/init_across_scenes")

EPS_GREEN = "#1b7837"
MVS_BLUE = "#2166ac"
GREY = "#9a9a9a"

# scene -> (sparse point count filled at runtime, MVSFormer carve dir or None)
SCENES = [
    dict(scan="scan122",
         mvs=DIAG / "mvsformer_scan122_sphere/sphere_carve_r060_res256_nomask_clip2_close2"),
    dict(scan="scan40",
         mvs=DIAG / "mvsformer_scan40_sphere/sphere_carve_auto_res256_nomask_clip2_close2"),
    dict(scan="scan24", mvs=None),
    dict(scan="scan65", mvs=None),
    dict(scan="scan118", mvs=None),
]


def fmt_count(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    return str(int(n))


def n_points(scan):
    p = Path("data/dtu_idr") / scan / "sparse_sfm_points.txt"
    if not p.exists():
        return None
    with p.open() as fh:
        return sum(1 for _ in fh)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nrow = len(SCENES)
    fig = plt.figure(figsize=(7.2, 2.05 * nrow), dpi=200)
    gs = fig.add_gridspec(
        nrow, 2, hspace=0.10, wspace=0.04,
        left=0.16, right=0.985, top=0.905, bottom=0.025)

    fig.suptitle("Two routes to a warm-start surface on DTU", x=0.575, y=0.975,
                 fontsize=13, fontweight="bold")
    # column headers
    fig.text(0.365, 0.928,
             "COLMAP sparse  →  $\\epsilon$-ball seed",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=EPS_GREEN)
    fig.text(0.365, 0.913, "ours · sparse points only", ha="center", va="top",
             fontsize=8.2, color="#555")
    fig.text(0.785, 0.928, "Sphere + MVSFormer++ carve",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=MVS_BLUE)
    fig.text(0.785, 0.913, "dense-MVS init · needs precomputed depths", ha="center",
             va="top", fontsize=8.2, color="#555")

    for r, sc in enumerate(SCENES):
        scan = sc["scan"]
        npts = n_points(scan)
        # row label (centered on the gridspec band [bottom, top])
        row_y = 0.905 - (0.905 - 0.025) * (r + 0.5) / nrow
        fig.text(0.025, row_y,
                 f"{scan}\n{fmt_count(npts)} pts" if npts else scan,
                 ha="left", va="center", fontsize=10, fontweight="bold",
                 transform=fig.transFigure)

        # ---- left: eps-ball seed --------------------------------------------
        axL = fig.add_subplot(gs[r, 0])
        eps_dir = OUTROOT / f"points_sdf_fit_{scan}_eps_opt"
        ply = eps_dir / "pred_mesh.ply"
        if ply.exists():
            stats = clay_render(axL, ply, az=25, el=18,
                                base=(0.80, 0.86, 0.80))
            fit = json.loads((eps_dir / "fit.json").read_text())
            wt = "✓" if fit.get("watertight") else "✗"
            cap = (f"{fmt_count(fit['n_faces'])} f · {fit['n_components']} comp · "
                   f"watertight {wt}")
            axL.text(0.5, -0.02, cap, transform=axL.transAxes, ha="center",
                     va="top", fontsize=8.0, color="#333")
        else:
            axL.text(0.5, 0.5, "fit running…", transform=axL.transAxes,
                     ha="center", va="center", fontsize=9, color=GREY)
            axL.set_xticks([]); axL.set_yticks([])
        for s in axL.spines.values():
            s.set_edgecolor(EPS_GREEN); s.set_linewidth(1.4)

        # ---- right: MVSFormer carve -----------------------------------------
        axR = fig.add_subplot(gs[r, 1])
        if sc["mvs"] is not None:
            cply = sc["mvs"] / "sphere_mvsformer_carved.ply"
            stats = clay_render(axR, cply, az=25, el=18,
                                base=(0.78, 0.83, 0.90))
            summ = json.loads((sc["mvs"] / "summary.json").read_text())
            cap = (f"{fmt_count(stats['faces'])} f · "
                   f"{summ['pct_removed_total']:.0f}% of sphere carved")
            axR.text(0.5, -0.02, cap, transform=axR.transAxes, ha="center",
                     va="top", fontsize=8.0, color="#333")
            for s in axR.spines.values():
                s.set_edgecolor(MVS_BLUE); s.set_linewidth(1.4)
        else:
            axR.text(0.5, 0.5, "dense MVS depths\nnot precomputed",
                     transform=axR.transAxes, ha="center", va="center",
                     fontsize=8.6, color=GREY, fontstyle="italic")
            axR.set_xticks([]); axR.set_yticks([])
            axR.set_facecolor("#f4f4f4")
            for s in axR.spines.values():
                s.set_edgecolor("#d6d6d6"); s.set_linewidth(1.0); s.set_linestyle((0, (3, 3)))

    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
