"""Combined SSAA-invariance figure for DTU scans 118, 97, and 110.

This reuses the same normal-buffer analysis as fig_stripes_ssaa.py, but presents
the three scenes in one paper figure: image evidence per scan, then shared
spectral/band plots.

Output: figures/fig_dtu_stripes_ssaa_multi.png (+ .pdf)
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


PX_MM = 0.3798
OUT = Path("figures/fig_dtu_stripes_ssaa_multi")


@dataclass(frozen=True)
class ScanSpec:
    label: str
    run: Path
    crop: tuple[int, int, int, int]
    color: str


SCANS = [
    ScanSpec(
        "scan 118",
        Path("outputs/run_20260602_220736_scan118_occ_4954645/st_buffers_view50"),
        (220, 560, 476, 816),
        "#8f1d21",
    ),
    ScanSpec(
        "scan 97",
        Path("outputs/run_20260613_034249_scan97_sphere_nomask_init_5012955/st_buffers_view50_it256_gpu"),
        (460, 455, 652, 647),
        "#1d6f9f",
    ),
    ScanSpec(
        "scan 110",
        Path("outputs/run_20260612_134452_scan110_sphere_nomask_init_5010815/st_buffers_view50_it256_gpu"),
        (600, 575, 856, 831),
        "#4f7f2a",
    ),
]


def load_patch(scan: ScanSpec, ss: int) -> np.ndarray:
    img = Image.open(scan.run / f"view50_ss{ss}_normals.png").convert("RGB")
    a = np.asarray(img, np.float32) / 255.0
    x0, y0, x1, y1 = scan.crop
    return a[y0:y1, x0:x1]


def shade(rgb: np.ndarray) -> np.ndarray:
    n = rgb * 2.0 - 1.0
    bg = (rgb > 0.98).all(-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    nm = n[~bg].mean(0)
    nm /= np.linalg.norm(nm)
    t = np.cross(nm, [0.0, 0.0, 1.0])
    t /= np.linalg.norm(t) + 1e-9
    l = nm * np.cos(np.deg2rad(45)) + t * np.sin(np.deg2rad(45))
    s = np.clip(n @ l, 0, 1)
    s = 0.10 + 0.90 * s
    s[bg] = 1.0
    return s


def radial_spectrum(gray: np.ndarray) -> np.ndarray:
    p = gray - gray.mean()
    h, w0 = p.shape
    win_y = np.hanning(h)
    win_x = np.hanning(w0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(p * win_y[:, None] * win_x[None, :])))
    yy, xx = np.mgrid[-h // 2 : h // 2, -w0 // 2 : w0 // 2]
    r = np.hypot(xx, yy).astype(int)
    rad = np.bincount(r.ravel(), F.ravel()) / np.bincount(r.ravel())
    return rad / (h * w0)


def band_rms(gray: np.ndarray, lo: float, hi: float) -> float:
    p = gray - gray.mean()
    h, w0 = p.shape
    F = np.abs(np.fft.fftshift(np.fft.fft2(p)))
    yy, xx = np.mgrid[-h // 2 : h // 2, -w0 // 2 : w0 // 2]
    r = np.hypot(xx, yy)
    m = (r >= lo) & (r < hi)
    return np.sqrt((F[m] ** 2).sum()) / (h * w0)


def add_scale_bar(ax, h: int) -> None:
    bar = 50
    y = h - 14
    ax.plot([12, 12 + bar], [y, y], "w-", lw=1.8,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="k"), pe.Normal()])
    ax.text(12 + bar / 2, y - 8, f"{bar * PX_MM:.0f} mm", ha="center",
            va="bottom", fontsize=5.8, color="w",
            path_effects=[pe.Stroke(linewidth=1.4, foreground="k"), pe.Normal()])


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "legend.fontsize": 6.3,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6,
        "savefig.bbox": "tight",
    })

    rows = []
    for scan in SCANS:
        p1 = load_patch(scan, 1)
        p8 = load_patch(scan, 8)
        g1 = p1.mean(-1)
        g8 = p8.mean(-1)
        h = g1.shape[0]
        stripe = (h / 50, h / 10)
        speckle = (h / 6, h / 2)
        rows.append({
            "scan": scan,
            "p1": p1,
            "p8": p8,
            "g1": g1,
            "g8": g8,
            "sh1": shade(p1),
            "sh8": shade(p8),
            "diff": 1.0 - np.clip(np.abs(g1 - g8) * 8, 0, 1),
            "h": h,
            "stripe": [band_rms(g1, *stripe), band_rms(g8, *stripe)],
            "speckle": [band_rms(g1, *speckle), band_rms(g8, *speckle)],
            "spec1": radial_spectrum(g1),
            "spec8": radial_spectrum(g8),
        })

    fig = plt.figure(figsize=(6.95, 7.05))
    gs = fig.add_gridspec(
        4, 6,
        height_ratios=[1.0, 1.0, 1.0, 1.34],
        left=0.075, right=0.99, bottom=0.08, top=0.965,
        hspace=0.27, wspace=0.62,
    )

    col_titles = ["1 spp", "64 spp", r"difference $|1-64|\times 8$"]
    for r, row in enumerate(rows):
        images = [row["sh1"], row["sh8"], row["diff"]]
        for c, im in enumerate(images):
            ax = fig.add_subplot(gs[r, 2 * c : 2 * c + 2])
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if r == 0:
                ax.set_title(col_titles[c], pad=2.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            if c == 0:
                ax.set_ylabel(row["scan"].label, rotation=0, labelpad=24,
                              va="center", ha="right", fontsize=7.2)
                add_scale_bar(ax, row["h"])

    spp = np.array([1, 64])
    ax_band = fig.add_subplot(gs[3, 0:2])
    ax_spec = fig.add_subplot(gs[3, 2:6])

    for row in rows:
        scan = row["scan"]
        ax_band.plot(spp, row["stripe"], "o-", color=scan.color, lw=1.15, ms=3.0)
        ax_band.plot(spp, row["speckle"], "s--", color=scan.color, lw=1.05,
                     ms=2.7, alpha=0.72)
    ax_band.set_xscale("log")
    ax_band.set_xticks(spp, ["1", "64"])
    ax_band.set_xlabel("samples / pixel", labelpad=1.5)
    ax_band.set_ylabel("band rms", labelpad=1.5)
    ax_band.set_ylim(0, 0.055)
    ax_band.margins(x=0.08)

    for row in rows:
        scan = row["scan"]
        h = row["h"]
        rbins = np.arange(len(row["spec1"]))
        lam = h / np.maximum(rbins, 1e-9)
        sel = (rbins >= 2) & (lam >= 2)
        ax_spec.loglog(lam[sel], row["spec1"][sel], color=scan.color, lw=1.0)
        ax_spec.loglog(lam[sel], row["spec8"][sel], color=scan.color, lw=1.0,
                       ls="--")

    ax_spec.axvspan(10, 50, color="#c1272d", alpha=0.10, lw=0)
    ax_spec.axvspan(2, 6, color="#0072bd", alpha=0.10, lw=0)
    ax_spec.set_xlim(230, 2)
    ax_spec.set_ylim(1.8e-5, 5.5e-3)
    ax_spec.set_xlabel("wavelength (px)", labelpad=1.5)
    ax_spec.set_ylabel("amplitude", labelpad=1.0)
    ax_spec.yaxis.set_label_coords(-0.07, 0.5)
    secax = ax_spec.secondary_xaxis(
        "top", functions=(lambda l: l * PX_MM, lambda m: m / PX_MM))
    secax.set_xlabel("wavelength (mm)", labelpad=1.5, fontsize=6.5)
    secax.tick_params(labelsize=6)

    scan_handles = [
        Line2D([0], [0], color=s.color, lw=1.5, label=s.label)
        for s in SCANS
    ]
    style_handles = [
        Line2D([0], [0], color="0.1", lw=1.25, ls="-", label="1 spp"),
        Line2D([0], [0], color="0.1", lw=1.25, ls="--", label="64 spp"),
    ]
    leg1 = ax_spec.legend(handles=scan_handles, frameon=False, loc="upper right",
                          handlelength=1.6, borderaxespad=0.2)
    ax_spec.add_artist(leg1)
    ax_spec.legend(handles=style_handles, frameon=False, loc="lower left",
                   handlelength=1.8, borderaxespad=0.2)

    ax_band.legend(handles=[
        Line2D([0], [0], color="0.1", marker="o", lw=1.2, label=r"stripes ($\lambda$ 10-50 px)"),
        Line2D([0], [0], color="0.1", marker="s", lw=1.1, ls="--", label=r"speckle ($\lambda<6$ px)"),
    ], frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.03),
       handlelength=1.5, borderaxespad=0.0, labelspacing=0.25)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=600)
    print(f"wrote {OUT}.png")
    for row in rows:
        print(row["scan"].label,
              "stripe", np.round(row["stripe"], 4),
              "speckle", np.round(row["speckle"], 4))


if __name__ == "__main__":
    main()
