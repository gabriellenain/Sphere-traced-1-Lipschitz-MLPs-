#!/usr/bin/env python3
"""Minimal per-view foreground sampling histogram diagnostic.

For foreground-ray sampling from the pooled deterministic ray table, the view
marginal is h_i = |F_i| / sum_j |F_j|. This script computes that quantity from
per-view masks and compares it to the uniform view marginal 1 / N_views.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ) if bold else (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _draw_dashed_line(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float],
                      fill: str, width: int = 2, dash: int = 8, gap: int = 6) -> None:
    x0, y0, x1, y1 = xy
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    dx = (x1 - x0) / length
    dy = (y1 - y0) / length
    t = 0.0
    while t < length:
        a = t
        b = min(t + dash, length)
        draw.line((x0 + dx * a, y0 + dy * a, x0 + dx * b, y0 + dy * b),
                  fill=fill, width=width)
        t += dash + gap


def _nice_ticks(vmax: float, n_ticks: int = 4) -> list[float]:
    if vmax <= 0:
        return [0.0]
    raw = vmax / max(n_ticks, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda m: abs(m * mag - raw)) * mag
    top = math.ceil(vmax / step) * step
    ticks = [i * step for i in range(int(round(top / step)) + 1)]
    return ticks


def _plot_histogram(rows: list[dict[str, float | int | str]], summary: dict[str, float | int | str],
                    out_png: Path) -> None:
    n = len(rows)
    h = np.array([float(r["h_i"]) for r in rows], dtype=np.float64)
    ratio = np.array([float(r["h_over_u"]) for r in rows], dtype=np.float64)
    u = float(summary["u_i"])

    w, hpx = 1360, 650
    margin_l, margin_r = 92, 40
    top_y0, top_y1 = 96, 354
    bot_y0, bot_y1 = 432, 585
    plot_w = w - margin_l - margin_r
    bg = "white"
    axis = "#202020"
    grid = "#e7e7e7"
    blue = "#2c7fb8"
    orange = "#d95f02"
    black = "#111111"
    muted = "#555555"

    img = Image.new("RGB", (w, hpx), bg)
    draw = ImageDraw.Draw(img)
    font_title = _font(24, bold=True)
    font = _font(15)
    font_small = _font(12)
    font_tiny = _font(11)

    title = f"{summary['label']} per-view foreground sampling histogram"
    subtitle = (
        f"N={n} views  uniform={100.0 * u:.3f}%  "
        f"ESS={summary['ess_views']:.1f}/{n}  "
        f"max/min={summary['h_over_u_max']:.2f}x/{summary['h_over_u_min']:.2f}x"
    )
    draw.text((margin_l, 28), title, fill=axis, font=font_title)
    draw.text((margin_l, 60), subtitle, fill=muted, font=font)

    def x_for(i: int) -> float:
        if n == 1:
            return margin_l + plot_w / 2
        return margin_l + i * plot_w / (n - 1)

    def draw_x_axis(y: int) -> None:
        draw.line((margin_l, y, margin_l + plot_w, y), fill=axis, width=2)
        for i in range(0, n, 50):
            x = x_for(i)
            draw.line((x, y, x, y + 5), fill=axis, width=1)
            draw.text((x - 10, y + 8), str(i), fill=muted, font=font_tiny)
        if (n - 1) % 50:
            x = x_for(n - 1)
            draw.line((x, y, x, y + 5), fill=axis, width=1)
            draw.text((x - 16, y + 8), str(n - 1), fill=muted, font=font_tiny)

    def y_map(v: float, ymin: float, ymax: float, y0: int, y1: int) -> float:
        return y1 - (v - ymin) / max(ymax - ymin, 1e-12) * (y1 - y0)

    top_max = max(float(h.max()), u) * 1.12
    top_ticks = _nice_ticks(top_max, 4)
    top_max = max(top_ticks[-1], top_max)
    for t in top_ticks:
        y = y_map(t, 0.0, top_max, top_y0, top_y1)
        draw.line((margin_l, y, margin_l + plot_w, y), fill=grid, width=1)
        draw.text((16, y - 8), f"{100.0 * t:.2f}%", fill=muted, font=font_small)
    draw.line((margin_l, top_y0, margin_l, top_y1), fill=axis, width=2)
    draw_x_axis(top_y1)
    bar_w = max(1, int(plot_w / max(n, 1)) - 1)
    for i, val in enumerate(h):
        x = x_for(i)
        y = y_map(float(val), 0.0, top_max, top_y0, top_y1)
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, top_y1), fill=blue)
    uy = y_map(u, 0.0, top_max, top_y0, top_y1)
    _draw_dashed_line(draw, (margin_l, uy, margin_l + plot_w, uy), black, width=2)
    draw.text((margin_l + 6, top_y0 + 8), "h_i = |F_i| / sum_j |F_j|", fill=blue, font=font)
    draw.text((margin_l + 6, top_y0 + 30), "dashed: uniform u_i = 1 / N_views",
              fill=black, font=font)

    ratio_max = max(1.0, float(ratio.max())) * 1.15
    ratio_ticks = _nice_ticks(ratio_max, 3)
    ratio_max = max(ratio_ticks[-1], ratio_max)
    for t in ratio_ticks:
        y = y_map(t, 0.0, ratio_max, bot_y0, bot_y1)
        draw.line((margin_l, y, margin_l + plot_w, y), fill=grid, width=1)
        draw.text((34, y - 8), f"{t:.1f}x", fill=muted, font=font_small)
    draw.line((margin_l, bot_y0, margin_l, bot_y1), fill=axis, width=2)
    draw_x_axis(bot_y1)
    one_y = y_map(1.0, 0.0, ratio_max, bot_y0, bot_y1)
    _draw_dashed_line(draw, (margin_l, one_y, margin_l + plot_w, one_y), black, width=2)
    for i, val in enumerate(ratio):
        x = x_for(i)
        y = y_map(float(val), 0.0, ratio_max, bot_y0, bot_y1)
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bot_y1), fill=orange)
    draw.text((margin_l + 6, bot_y0 + 8), "relative exposure: h_i / u_i",
              fill=orange, font=font)
    draw.text((margin_l + plot_w - 72, bot_y1 + 36), "view id", fill=muted, font=font_small)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def _pose_paths(scene: Path) -> list[Path]:
    pose_dir = scene / "pose"
    paths = sorted(p for p in pose_dir.glob("*.txt") if not p.name.startswith("._"))
    if not paths:
        raise FileNotFoundError(f"no pose txt files under {pose_dir}")
    return paths


def _resolve_mask_dir(scene: Path, mask_dir: Path | None) -> Path:
    candidates = []
    if mask_dir is not None:
        candidates.append(mask_dir)
    candidates.extend([scene / "mask", scene / "mask_truck"])
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "no mask directory found; pass --mask-dir or create scene/mask/"
    )


def _load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 127


def _sampled_foreground_count(mask: np.ndarray, down: int) -> tuple[int, int]:
    if down <= 0:
        raise ValueError("--down must be >= 1")
    height, width = mask.shape[:2]
    h_d, w_d = height // down, width // down
    ys, xs = np.meshgrid(np.arange(h_d), np.arange(w_d), indexing="ij")
    xs_f = (xs + 0.5) * down - 0.5
    ys_f = (ys + 0.5) * down - 0.5
    yi = np.clip((ys_f + 0.5).astype(np.int64).ravel(), 0, height - 1)
    xi = np.clip((xs_f + 0.5).astype(np.int64).ravel(), 0, width - 1)
    sampled = mask[yi, xi]
    return int(sampled.sum()), int(sampled.size)


def compute(scene: Path, mask_dir: Path, label: str, down: int) -> tuple[list[dict], dict]:
    poses = _pose_paths(scene)
    records = []
    missing = []
    for view_id, pose_path in enumerate(poses):
        mask_path = mask_dir / f"{pose_path.stem}.png"
        if not mask_path.exists():
            missing.append(mask_path.name)
            continue
        mask = _load_mask(mask_path)
        fg, total = _sampled_foreground_count(mask, down)
        records.append({
            "view_id": view_id,
            "pose_stem": pose_path.stem,
            "mask": str(mask_path),
            "fg_pixels": fg,
            "total_pixels": total,
            "coverage": fg / max(total, 1),
        })
    if missing:
        preview = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"missing masks matching poses: {preview}{extra}")
    total_fg = sum(int(r["fg_pixels"]) for r in records)
    if total_fg <= 0:
        raise ValueError(f"all foreground counts are zero in {mask_dir}")
    n = len(records)
    u = 1.0 / n
    h_vals = []
    ratios = []
    for r in records:
        h_i = int(r["fg_pixels"]) / total_fg
        ratio = h_i / u
        r["h_i"] = h_i
        r["u_i"] = u
        r["h_over_u"] = ratio
        h_vals.append(h_i)
        ratios.append(ratio)
    h_arr = np.array(h_vals, dtype=np.float64)
    ratio_arr = np.array(ratios, dtype=np.float64)
    coverage = np.array([float(r["coverage"]) for r in records], dtype=np.float64)
    ess = 1.0 / float(np.sum(h_arr ** 2))
    summary = {
        "label": label,
        "scene": str(scene),
        "mask_dir": str(mask_dir),
        "down": down,
        "n_views": n,
        "total_foreground_pixels": int(total_fg),
        "total_sampled_pixels": int(sum(int(r["total_pixels"]) for r in records)),
        "u_i": u,
        "h_i_min": float(h_arr.min()),
        "h_i_mean": float(h_arr.mean()),
        "h_i_max": float(h_arr.max()),
        "h_over_u_min": float(ratio_arr.min()),
        "h_over_u_mean": float(ratio_arr.mean()),
        "h_over_u_max": float(ratio_arr.max()),
        "coverage_min": float(coverage.min()),
        "coverage_mean": float(coverage.mean()),
        "coverage_max": float(coverage.max()),
        "ess_views": float(ess),
        "ess_fraction": float(ess / n),
    }
    return records, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("data/tnt/Truck"))
    ap.add_argument("--mask-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("_diagnostics/truck_sampling_histogram"))
    ap.add_argument("--label", default="Truck TnT")
    ap.add_argument("--down", type=int, default=1,
                    help="deterministic ray-grid downsample factor used in training")
    args = ap.parse_args()

    mask_dir = _resolve_mask_dir(args.scene, args.mask_dir)
    rows, summary = compute(args.scene, mask_dir, args.label, args.down)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "view_sampling_histogram.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "view_id", "pose_stem", "mask", "fg_pixels", "total_pixels",
            "coverage", "h_i", "u_i", "h_over_u",
        ])
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out_dir / "view_sampling_histogram_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    png_path = args.out_dir / "view_sampling_histogram.png"
    _plot_histogram(rows, summary, png_path)

    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(
        "summary: "
        f"N={summary['n_views']} u={summary['u_i']:.6f} "
        f"h[min/mean/max]={summary['h_i_min']:.6f}/"
        f"{summary['h_i_mean']:.6f}/{summary['h_i_max']:.6f} "
        f"ratio[min/mean/max]={summary['h_over_u_min']:.3f}/"
        f"{summary['h_over_u_mean']:.3f}/{summary['h_over_u_max']:.3f} "
        f"ESS={summary['ess_views']:.1f}/{summary['n_views']}"
    )


if __name__ == "__main__":
    main()
