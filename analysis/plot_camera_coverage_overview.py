#!/usr/bin/env python3
"""Create a paper-style camera-coverage overview from NCC diagnostic arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_array(directory: Path, name: str) -> np.ndarray:
    return np.load(directory / f"{name}.npy")


def masked(values: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    shown = values.astype(np.float32).copy()
    shown[~foreground] = np.nan
    return shown


def add_map(fig, ax, values, foreground, title, subtitle, cmap, vmin, vmax):
    image = ax.imshow(masked(values, foreground), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13, fontweight="semibold", pad=8)
    ax.text(0.5, -0.035, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#444444")
    ax.set_axis_off()
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    bar.ax.tick_params(labelsize=8)
    return image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="directory written by analysis/ncc_supervision_coverage.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <input-dir>/camera_coverage_overview.png)")
    args = ap.parse_args()

    directory = args.input_dir
    out = args.out or directory / "camera_coverage_overview.png"
    meta = json.loads((directory / "metadata.json").read_text())
    foreground = load_array(directory, "foreground_hit").astype(bool)
    rgb = np.asarray(Image.open(directory / "reference_rgb.png"))

    n_current = max(len(meta["current_cameras"]), 1)
    n_all = max(len(meta["all_cameras"]), 1)
    center_current = load_array(directory, "n_center_current")
    patch_current = load_array(directory, "n_patch_current")
    center_all = load_array(directory, "n_center_all")
    patch_all = load_array(directory, "n_patch_all")
    rho_max = load_array(directory, "rho_max_all")
    patch_gain = patch_all / n_all - patch_current / n_current

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 12.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Cross-view camera coverage | reference view {meta['reference_view']} | "
        f"{n_current} nearest vs. {n_all} all alternatives",
        fontsize=18, fontweight="bold",
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Reference RGB", fontsize=13, fontweight="semibold", pad=8)
    axes[0, 0].text(
        0.5, -0.035, "sphere-traced foreground is evaluated pixelwise",
        transform=axes[0, 0].transAxes, ha="center", va="top",
        fontsize=9, color="#444444")
    axes[0, 0].set_axis_off()

    add_map(fig, axes[0, 1], center_current / n_current, foreground,
            "Centre coverage | nearest cameras",
            "fraction passing visibility, mask and normal-angle gates",
            "viridis", 0.0, 1.0)
    add_map(fig, axes[0, 2], patch_current / n_current, foreground,
            "Full-patch coverage | nearest cameras",
            "fraction also keeping the complete tangent patch in frame",
            "viridis", 0.0, 1.0)
    add_map(fig, axes[1, 0], center_all / n_all, foreground,
            "Centre coverage | all cameras",
            "fraction passing visibility, mask and normal-angle gates",
            "viridis", 0.0, 1.0)
    add_map(fig, axes[1, 1], patch_all / n_all, foreground,
            "Full-patch coverage | all cameras",
            "fraction also keeping the complete tangent patch in frame",
            "viridis", 0.0, 1.0)
    limit = max(float(np.nanmax(np.abs(masked(patch_gain, foreground)))), 1e-6)
    add_map(fig, axes[1, 2], patch_gain, foreground,
            "Coverage gain | all minus nearest",
            "difference between normalized full-patch coverage fractions",
            "coolwarm", -limit, limit)

    fg_rho = rho_max[foreground]
    summary = (
        f"checkpoint step {meta['checkpoint_step']}   |   "
        f"patch {meta['ncc_patch']}x{meta['ncc_patch']}, half-width {meta['ncc_half_pix']:g}px   |   "
        f"foreground hits {int(foreground.sum()):,}   |   "
        f"median best joint patch support {float(np.median(fg_rho)):.2f}"
    )
    fig.text(0.5, 0.006, summary, ha="center", va="bottom", fontsize=10,
             color="#333333")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
