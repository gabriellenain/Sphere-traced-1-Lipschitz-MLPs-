#!/usr/bin/env python3
"""ICLR figure: seeding the 1-Lipschitz FTheta from a COLMAP *sparse* cloud.

Every method below is fed the identical input — `scan122/sparse_sfm_points.txt`
(~14k SfM points, already in the normalized training frame) — and asked to emit a
warm-start surface. The figure contrasts the two families we tried:

  A. Implicit objectives that must *learn* the sign of f from the points alone.
     They all fail on a sparse cloud: with no closed shell the inside/outside
     decision is unconstrained in the gaps, so the fit either collapses onto the
     f==0 global optimum (marching-cubes carves the PE ripple into the whole
     cube) or shatters into specks around point clusters.

  B. Sign-by-construction: build an exact SDF first (union of eps-balls ->
     border flood-fill decides interior -> EDT), then regress f onto it. The
     target is a real, strongly-positive-far SDF, so it cannot collapse and is
     watertight by construction. Tuning eps to the cloud's natural connected-
     component floor (3x voxel, max_components=4) gives the cleanest seed.

Inputs are the pred_mesh_hq.png 4-view renders + fit.json already produced by
fit_points_sdf.py on scan122. No re-running; this only composites + tabulates.
Output: analysis/figures/colmap_init_compare.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path("/scratch/_projets_/willow/1-lip-tracer-new/Sphere-traced-outputs")
OUT = Path("analysis/figures/colmap_init_compare")

# ---- the methods we tried on DTU scan122, in narrative order ----------------
# (dir, short title, multi-line subtitle, family, verdict tag)
METHODS = [
    dict(dir="points_sdf_fit_scan122_surfonly", fam="A",
         title="Unsigned fit",
         sub=r"$\min\,|f(p)|^2$", verdict="collapse"),
    dict(dir="points_sdf_fit_scan122_plain", fam="A",
         title="SAL push-away",
         sub=r"$+\,\lambda\,e^{-\alpha|f|}$ (full cube)", verdict="collapse"),
    dict(dir="points_sdf_fit_scan122", fam="A",
         title="SAL + ROI",
         sub=r"push-away in SfM bbox", verdict="fragments"),
    dict(dir="points_sdf_fit_scan122_closed", fam="B",
         title="$\\epsilon$-ball, force 1 comp",
         sub=r"$\epsilon=2{\times}$voxel", verdict="watertight"),
    dict(dir="points_sdf_fit_scan122_eps_opt", fam="B",
         title="$\\epsilon$-ball @ comp. floor",
         sub=r"$\epsilon=3{\times}$voxel, $\leq\!4$ comp", verdict="BEST"),
]

BEST_GREEN = "#1b7837"
FAIL_RED = "#b2182b"
WARN_AMBER = "#d98c00"
FAMA = "#8c510a"
FAMB = "#01665e"


def front_crop(hq_png: Path, pad: int = 10) -> np.ndarray:
    """Tight content crop of the leftmost ('front') panel of a 4-view render."""
    im = np.asarray(Image.open(hq_png).convert("RGB"))
    H, W, _ = im.shape
    panel = im[:, : W // 4]
    body = panel[45:]  # drop the black title text strip
    mask = body.min(axis=2) < 222  # object pixels (render bg ~234, paper 255)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return panel
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, body.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, body.shape[1])
    crop = body[y0:y1, x0:x1]
    # pad to square on white so all panels share an aspect ratio
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 255, np.uint8)
    sq[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = crop
    return sq


def load_fit(d: Path) -> dict:
    return json.loads((d / "fit.json").read_text())


def fmt_count(n: int) -> str:
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}k"
    return str(n)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fits = [load_fit(ROOT / m["dir"]) for m in METHODS]
    crops = [front_crop(ROOT / m["dir"] / "pred_mesh_hq.png") for m in METHODS]

    n = len(METHODS)
    fig = plt.figure(figsize=(2.05 * n, 6.4), dpi=200)
    # rows: family bracket / render / title+sub / metrics table
    gs = fig.add_gridspec(
        3, n, height_ratios=[0.16, 1.0, 0.62],
        hspace=0.06, wspace=0.05,
        left=0.085, right=0.985, top=0.93, bottom=0.045)

    fig.suptitle(
        "Seeding the 1-Lipschitz network from a sparse COLMAP cloud  (DTU scan122, "
        "~14k SfM points, identical input)",
        fontsize=12.5, fontweight="bold", y=0.985)

    # ---- family brackets across the top -------------------------------------
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    ax_top.set_xlim(0, n)
    ax_top.set_ylim(0, 1)

    def bracket(c0, c1, label, color):
        x0, x1 = c0 + 0.04, c1 + 0.96
        y = 0.42
        ax_top.plot([x0, x0, x1, x1], [y + 0.22, y, y, y + 0.22],
                    color=color, lw=1.6, clip_on=False)
        ax_top.text((x0 + x1) / 2, y - 0.34, label, ha="center", va="top",
                    color=color, fontsize=10.2, fontweight="bold")

    bracket(0, 2, "A.  Sign learned from points  →  fails on a sparse cloud", FAMA)
    bracket(3, 4, "B.  Sign by construction  ($\\epsilon$-ball $\\to$ flood-fill $\\to$ EDT)", FAMB)

    # ---- render row + titles ------------------------------------------------
    for j, (m, fit, crop) in enumerate(zip(METHODS, fits, crops)):
        is_best = m["verdict"] == "BEST"
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(crop)
        ax.set_xticks([]); ax.set_yticks([])
        edge = BEST_GREEN if is_best else ("#cfcfcf")
        lw = 3.2 if is_best else 0.8
        for s in ax.spines.values():
            s.set_edgecolor(edge); s.set_linewidth(lw)
        col = FAMA if m["fam"] == "A" else FAMB
        title = ("★ " + m["title"] + " ★") if is_best else m["title"]
        ax.set_title(title, fontsize=10.6,
                     fontweight="bold" if is_best else "normal",
                     color=BEST_GREEN if is_best else col, pad=4)
        ax.text(0.5, -0.055, m["sub"], transform=ax.transAxes,
                ha="center", va="top", fontsize=8.6, color="#444")

    # ---- metrics table ------------------------------------------------------
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis("off")
    ax_t.set_xlim(0, n)
    ax_t.set_ylim(0, 1)

    rows = [
        ("triangles",   lambda f: fmt_count(int(f["n_faces"]))),
        ("components",  lambda f: fmt_count(int(f["n_components"])) if "n_components" in f else "$10^3{+}$"),
        ("watertight",  lambda f: ("✓" if f.get("watertight") else "✗")),
        ("pt$\\rightarrow$surf (mean)", lambda f: f"{f['point_to_surface_mean']:.3f}"),
    ]
    row_label_x = -0.02
    n_rows = len(rows) + 1  # +1 for verdict
    ytop, ybot = 0.97, 0.05
    yh = (ytop - ybot) / n_rows
    ys = [ytop - (i + 0.5) * yh for i in range(n_rows)]

    # row labels
    for i, (lbl, _) in enumerate(rows):
        ax_t.text(row_label_x, ys[i], lbl, ha="right", va="center",
                  fontsize=8.6, color="#222", transform=ax_t.transData,
                  fontstyle="italic")
    ax_t.text(row_label_x, ys[-1], "verdict", ha="right", va="center",
              fontsize=8.6, color="#222", fontstyle="italic")

    verdict_color = {"collapse": FAIL_RED, "fragments": FAIL_RED,
                     "watertight": WARN_AMBER, "BEST": BEST_GREEN}
    verdict_txt = {"collapse": "collapse", "fragments": "fragments",
                   "watertight": "frag. / bloat", "BEST": "clean seed"}

    for j, (m, fit) in enumerate(zip(METHODS, fits)):
        cx = j + 0.5
        is_best = m["verdict"] == "BEST"
        if is_best:
            ax_t.add_patch(FancyBboxPatch(
                (j + 0.04, ybot), 0.92, ytop - ybot,
                boxstyle="round,pad=0,rounding_size=0.04",
                linewidth=2.0, edgecolor=BEST_GREEN,
                facecolor=(0.86, 0.94, 0.88), zorder=0))
        for i, (_, fn) in enumerate(rows):
            val = fn(fit)
            color = "#111"
            weight = "normal"
            if i == 2:  # watertight glyph
                color = BEST_GREEN if val == "✓" else FAIL_RED
                weight = "bold"
            if is_best:
                weight = "bold"
            ax_t.text(cx, ys[i], val, ha="center", va="center",
                      fontsize=9.4, color=color, fontweight=weight)
        vc = verdict_color[m["verdict"]]
        ax_t.text(cx, ys[-1], verdict_txt[m["verdict"]], ha="center", va="center",
                  fontsize=9.0, color=vc, fontweight="bold")

    # faint row separators
    for i in range(n_rows + 1):
        yline = ytop - i * yh
        ax_t.plot([-0.0, n], [yline, yline], color="#e3e3e3", lw=0.6, zorder=-1)

    # footnote on the p2s caveat
    fig.text(0.5, 0.012,
             "Note: low pt$\\rightarrow$surf for the collapsed fits is pathological "
             "(the surface fills the cube, so every point is near it); the chosen seed "
             "sits $\\approx\\epsilon$ outside the cloud by design and training carves it back.",
             ha="center", va="bottom", fontsize=7.6, color="#666", fontstyle="italic")

    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", OUT.with_suffix(".png"), "and .pdf")


if __name__ == "__main__":
    main()
