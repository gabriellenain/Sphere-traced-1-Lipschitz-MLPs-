#!/usr/bin/env python3
"""CVPR-style summary of nearest source views.

The training selector with ``view_selection="nearest"`` uses the six closest
camera centres per reference view. This script reports exactly that choice for
selected reference views, plus aggregate statistics over every camera.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # keep this runnable on lean cluster login envs
    plt = None


def _load_tnt_cameras(scene: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Return normalized c2w matrices, camera centers and matching RGB paths."""
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float64)
    bb_min, bb_max = bbox[:3], bbox[3:6]
    center = 0.5 * (bb_min + bb_max)
    scale = float(np.max(0.5 * (bb_max - bb_min)))

    pose_paths = sorted((scene / "pose").glob("0_*.txt"))
    if not pose_paths:
        raise FileNotFoundError(f"no training poses 0_*.txt under {scene / 'pose'}")

    c2ws, rgb_paths = [], []
    for pp in pose_paths:
        rgb = scene / "rgb" / (pp.stem + ".png")
        if not rgb.exists():
            for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                alt = scene / "rgb" / (pp.stem + suffix)
                if alt.exists():
                    rgb = alt
                    break
        if not rgb.exists():
            continue
        c2w = np.loadtxt(pp, dtype=np.float64).reshape(4, 4)
        c2w[:3, 3] = (c2w[:3, 3] - center) / scale
        c2ws.append(c2w)
        rgb_paths.append(rgb)

    if not c2ws:
        raise FileNotFoundError(f"no RGB images matching poses under {scene}")
    c2w_arr = np.stack(c2ws)
    return c2w_arr, c2w_arr[:, :3, 3].copy(), rgb_paths


def _load_dtu_idr_cameras(scene: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Return normalized c2w matrices from IDR/DTU cameras.npz."""
    from scipy.linalg import rq

    cam_path = scene / "cameras.npz"
    if not cam_path.exists():
        cam_path = scene / "cameras_sphere.npz"
    if not cam_path.exists():
        raise FileNotFoundError(f"no cameras.npz/cameras_sphere.npz under {scene}")
    cam_dict = np.load(cam_path)
    img_paths = sorted(
        p for p in (scene / "image").iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and not p.name.startswith("._")
    )
    if not img_paths:
        raise FileNotFoundError(f"no images under {scene / 'image'}")

    c2ws = []
    for i, _img_path in enumerate(img_paths):
        P = cam_dict[f"world_mat_{i}"][:3, :4].astype(np.float64)
        M = P[:, :3]
        K, R = rq(M)
        sign = np.sign(np.diag(K))
        sign[sign == 0] = 1.0
        T = np.diag(sign)
        K = K @ T
        R = T @ R
        if np.linalg.det(R) < 0:
            K[:, 2] *= -1.0
            R[2, :] *= -1.0
        K = K / K[2, 2]
        t = np.linalg.solve(K, P[:, 3])
        cam_center = -R.T @ t
        cam_center_h = np.concatenate([cam_center, [1.0]], axis=0)
        key_inv = f"scale_mat_inv_{i}"
        scale_mat_inv = cam_dict[key_inv] if key_inv in cam_dict else np.linalg.inv(cam_dict[f"scale_mat_{i}"])
        cam_center = (scale_mat_inv @ cam_center_h)[:3]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = cam_center
        c2ws.append(c2w)

    c2w_arr = np.stack(c2ws)
    return c2w_arr, c2w_arr[:, :3, 3].copy(), img_paths


def _load_scene_cameras(scene: Path) -> tuple[str, np.ndarray, np.ndarray, list[Path]]:
    if (scene / "intrinsics.txt").exists() and (scene / "pose").is_dir():
        c2ws, centers, rgb_paths = _load_tnt_cameras(scene)
        return "tnt", c2ws, centers, rgb_paths
    if ((scene / "cameras.npz").exists() or (scene / "cameras_sphere.npz").exists()) and (scene / "image").is_dir():
        c2ws, centers, rgb_paths = _load_dtu_idr_cameras(scene)
        return "dtu_idr", c2ws, centers, rgb_paths
    raise FileNotFoundError(f"could not detect TnT or DTU/IDR scene layout under {scene}")


def _object_center(scene: Path, centers: np.ndarray) -> np.ndarray:
    sfm_path = scene / "sparse_sfm_points.txt"
    if sfm_path.exists():
        pts = np.loadtxt(sfm_path, dtype=np.float64).reshape(-1, 3)
        if len(pts):
            return pts.mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def _object_axis_xz(scene: Path) -> dict[str, object] | None:
    """Rough object orientation from PCA of sparse SfM points, projected to x-z."""
    sfm_path = scene / "sparse_sfm_points.txt"
    if not sfm_path.exists():
        return None
    pts = np.loadtxt(sfm_path, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return None
    xz = pts[:, [0, 2]]
    center = xz.mean(axis=0)
    xc = xz - center
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    axis = eigvec[:, order[0]]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    # Sign is arbitrary for an orientation axis; make output deterministic.
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    total = float(np.maximum(eigval.sum(), 1e-12))
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return {
        "center_xz": center.tolist(),
        "axis_xz": axis.tolist(),
        "angle_deg_from_x": angle,
        "variance_ratio": float(eigval[order[0]] / total),
    }


def _parse_ref_views(spec: str, n_views: int) -> list[int]:
    spec = spec.strip().lower()
    if spec in {"paper6", "six", "default"}:
        return [int(round(i * (n_views - 1) / 5)) for i in range(6)]
    if spec in {"many", "paper24", "twentyfour"}:
        return [int(round(i * (n_views - 1) / 23)) for i in range(24)]
    if spec.startswith("evenly"):
        n = int(spec.removeprefix("evenly"))
        if n <= 1:
            raise ValueError("--ref-views evenlyN requires N > 1")
        return [int(round(i * (n_views - 1) / (n - 1))) for i in range(n)]
    if spec in {"all", "*"}:
        return list(range(n_views))
    refs = [int(tok) for tok in spec.replace(",", " ").split()]
    if not refs:
        raise ValueError("empty --ref-views")
    bad = [r for r in refs if r < 0 or r >= n_views]
    if bad:
        raise ValueError(f"reference views out of range for {n_views} views: {bad}")
    return refs


def _nearest_sources(centers: np.ndarray, n_src: int) -> np.ndarray:
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :n_src]


def _parallax_angles_deg(centers: np.ndarray, obj: np.ndarray) -> np.ndarray:
    vec = centers - obj[None, :]
    vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
    cos = np.clip(vec @ vec.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _select_sources_by_distance_and_parallax(
    centers: np.ndarray,
    obj: np.ndarray,
    n_src: int,
    min_parallax_deg: float,
) -> tuple[np.ndarray, dict[str, object]]:
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if min_parallax_deg <= 0.0:
        return np.argsort(d, axis=1)[:, :n_src], {
            "selector": "nearest camera centers",
            "min_parallax_deg": 0.0,
            "min_valid_candidates": int(centers.shape[0] - 1),
            "refs_with_too_few_candidates": [],
        }

    parallax = _parallax_angles_deg(centers, obj)
    selected = np.empty((len(centers), n_src), dtype=np.int64)
    valid_counts: list[int] = []
    too_few: list[dict[str, int]] = []
    for r in range(len(centers)):
        valid = np.flatnonzero((np.arange(len(centers)) != r) & (parallax[r] >= min_parallax_deg))
        valid_counts.append(int(len(valid)))
        if len(valid) < n_src:
            too_few.append({"ref": int(r), "valid_candidates": int(len(valid))})
            continue
        ranked = valid[np.argsort(d[r, valid])]
        selected[r] = ranked[:n_src]
    if too_few:
        first = too_few[:8]
        raise ValueError(
            f"min_parallax_deg={min_parallax_deg:g} leaves fewer than n_src={n_src} "
            f"candidates for {len(too_few)} refs; first: {first}"
        )
    return selected, {
        "selector": "nearest camera centers subject to object-center parallax",
        "min_parallax_deg": float(min_parallax_deg),
        "min_valid_candidates": int(min(valid_counts)),
        "median_valid_candidates": float(np.median(valid_counts)),
        "max_valid_candidates": int(max(valid_counts)),
        "refs_with_too_few_candidates": too_few,
    }


def _pair_stats(c2ws: np.ndarray, centers: np.ndarray, src: np.ndarray,
                obj: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    V, n_src = src.shape
    rows = []
    for ref in range(V):
        ref_center = centers[ref]
        ref_vec = ref_center - obj
        ref_obj_dist = max(float(np.linalg.norm(ref_vec)), 1e-12)
        ref_axis = c2ws[ref, :3, 2]
        ref_axis = ref_axis / max(float(np.linalg.norm(ref_axis)), 1e-12)
        for rank, s in enumerate(src[ref], start=1):
            src_center = centers[s]
            src_vec = src_center - obj
            src_obj_dist = max(float(np.linalg.norm(src_vec)), 1e-12)
            baseline = float(np.linalg.norm(src_center - ref_center))
            cos = float(np.clip(np.dot(ref_vec, src_vec) / (ref_obj_dist * src_obj_dist), -1.0, 1.0))
            parallax = math.degrees(math.acos(cos))
            src_axis = c2ws[s, :3, 2]
            src_axis = src_axis / max(float(np.linalg.norm(src_axis)), 1e-12)
            axis_cos = float(np.clip(np.dot(ref_axis, src_axis), -1.0, 1.0))
            axis_angle = math.degrees(math.acos(axis_cos))
            rows.append((
                ref,
                int(s),
                rank,
                baseline,
                100.0 * baseline / ref_obj_dist,
                parallax,
                axis_angle,
                ref_obj_dist,
                src_obj_dist,
            ))

    arr = np.asarray(rows, dtype=np.float64)
    summary = {
        "num_reference_views": int(V),
        "sources_per_reference": int(n_src),
        "baseline_mean": float(arr[:, 3].mean()),
        "baseline_median": float(np.median(arr[:, 3])),
        "baseline_min": float(arr[:, 3].min()),
        "baseline_max": float(arr[:, 3].max()),
        "baseline_pct_ref_depth_mean": float(arr[:, 4].mean()),
        "parallax_deg_mean": float(arr[:, 5].mean()),
        "parallax_deg_median": float(np.median(arr[:, 5])),
        "parallax_deg_min": float(arr[:, 5].min()),
        "parallax_deg_max": float(arr[:, 5].max()),
        "axis_angle_deg_mean": float(arr[:, 6].mean()),
        "axis_angle_deg_median": float(np.median(arr[:, 6])),
    }
    return arr, summary


def _write_csv(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ref_view",
            "src_view",
            "rank",
            "camera_distance_norm",
            "baseline_pct_ref_object_distance",
            "center_parallax_deg",
            "optical_axis_angle_deg",
            "ref_object_distance_norm",
            "src_object_distance_norm",
        ])
        for row in rows:
            writer.writerow([
                int(row[0]),
                int(row[1]),
                int(row[2]),
                f"{row[3]:.6f}",
                f"{row[4]:.3f}",
                f"{row[5]:.3f}",
                f"{row[6]:.3f}",
                f"{row[7]:.6f}",
                f"{row[8]:.6f}",
            ])


def _reference_summary(rows: np.ndarray, refs: list[int]) -> list[dict[str, float | str | int]]:
    out: list[dict[str, float | str | int]] = []
    ref_mask = np.isin(rows[:, 0].astype(int), refs)
    shown = rows[ref_mask]
    for r in refs:
        rr = rows[rows[:, 0] == r]
        out.append({
            "ref_view": int(r),
            "mean_camera_distance_norm": float(rr[:, 3].mean()),
            "min_camera_distance_norm": float(rr[:, 3].min()),
            "max_camera_distance_norm": float(rr[:, 3].max()),
            "mean_baseline_pct_ref_object_distance": float(rr[:, 4].mean()),
            "mean_center_parallax_deg": float(rr[:, 5].mean()),
            "min_center_parallax_deg": float(rr[:, 5].min()),
            "max_center_parallax_deg": float(rr[:, 5].max()),
            "mean_optical_axis_angle_deg": float(rr[:, 6].mean()),
        })
    out.append({
        "ref_view": "mean",
        "mean_camera_distance_norm": float(shown[:, 3].mean()),
        "min_camera_distance_norm": float(shown[:, 3].min()),
        "max_camera_distance_norm": float(shown[:, 3].max()),
        "mean_baseline_pct_ref_object_distance": float(shown[:, 4].mean()),
        "mean_center_parallax_deg": float(shown[:, 5].mean()),
        "min_center_parallax_deg": float(shown[:, 5].min()),
        "max_center_parallax_deg": float(shown[:, 5].max()),
        "mean_optical_axis_angle_deg": float(shown[:, 6].mean()),
    })
    return out


def _write_reference_summary_csv(path: Path, rows: np.ndarray, refs: list[int]) -> None:
    summary = _reference_summary(rows, refs)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })


def _thumb(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - im.width) // 2
    y = (size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, size[0], 20), fill=(0, 0, 0, 150))
    draw.text((5, 4), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return canvas


def _save_thumbnail_strip(path: Path, rgb_paths: list[Path], refs: list[int],
                          src: np.ndarray) -> None:
    cell = (118, 66)
    gap = 5
    rows = []
    for r in refs:
        ids = [r, *src[r].tolist()]
        thumbs = [_thumb(rgb_paths[i], ("ref " if i == r else "src ") + str(i), cell) for i in ids]
        row = Image.new("RGB", ((cell[0] + gap) * len(thumbs) - gap, cell[1]), "white")
        x = 0
        for im in thumbs:
            row.paste(im, (x, 0))
            x += cell[0] + gap
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, (cell[1] + gap) * len(rows) - gap), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += cell[1] + gap
    out.save(path)


def _save_figure(path: Path, scene_label: str, selector_label: str,
                 centers: np.ndarray, refs: list[int], src: np.ndarray,
                 rows: np.ndarray, summary: dict[str, float],
                 obj_axis: dict[str, object] | None = None) -> None:
    if plt is None:
        _save_figure_pil(path, scene_label, selector_label, centers, refs, src, rows, summary, obj_axis)
        return

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 180,
    })
    n_labels = len(refs) + 1
    fig_w = max(7.0, 3.8 + 0.22 * n_labels)
    fig = plt.figure(figsize=(fig_w, 3.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])

    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.scatter(centers[:, 0], centers[:, 2], s=5, c="#c5cbd3", edgecolors="none", label="all train")
    palette = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
    for k, r in enumerate(refs):
        color = palette[k % len(palette)]
        for s in src[r]:
            ax_cam.plot([centers[r, 0], centers[s, 0]], [centers[r, 2], centers[s, 2]],
                        lw=0.55, alpha=0.65, c=color)
        ax_cam.scatter(centers[r, 0], centers[r, 2], s=22, c=color, edgecolors="black", linewidths=0.35)
        ax_cam.text(centers[r, 0], centers[r, 2], f" {r}", color=color, fontsize=7, va="center")
    ax_cam.scatter([0.0], [0.0], marker="+", s=45, c="black", linewidths=0.8)
    if obj_axis is not None:
        center_xz = np.asarray(obj_axis["center_xz"], dtype=np.float64)
        axis_xz = np.asarray(obj_axis["axis_xz"], dtype=np.float64)
        xz = centers[:, [0, 2]]
        length = 0.20 * float(np.max(np.ptp(xz, axis=0)))
        p0 = center_xz - 0.5 * length * axis_xz
        p1 = center_xz + 0.5 * length * axis_xz
        ax_cam.plot([p0[0], p1[0]], [p0[1], p1[1]], c="#111111", lw=2.0, solid_capstyle="round")
        ax_cam.scatter([center_xz[0]], [center_xz[1]], marker="x", s=26, c="#111111", linewidths=0.9)
        ax_cam.text(center_xz[0], center_xz[1], " rough object axis", fontsize=7, va="bottom", color="#111111")
    ax_cam.set_aspect("equal", adjustable="box")
    ax_cam.set_title(f"{scene_label} camera layout")
    ax_cam.set_xlabel("x")
    ax_cam.set_ylabel("z")
    ax_cam.spines[["top", "right"]].set_visible(False)

    ref_mask = np.isin(rows[:, 0].astype(int), refs)
    ref_rows = rows[ref_mask]
    for ax, col, title, color in [
        (fig.add_subplot(gs[0, 1]), 5, "object-center parallax", "#0072b2"),
        (fig.add_subplot(gs[0, 2]), 3, "camera-center distance", "#d55e00"),
        (fig.add_subplot(gs[1, 1]), 4, "baseline / ref distance", "#009e73"),
        (fig.add_subplot(gs[1, 2]), 6, "optical-axis angle", "#cc79a7"),
    ]:
        vals = [ref_rows[ref_rows[:, 0] == r, col] for r in refs]
        vals.append(ref_rows[:, col])
        labels = [str(r) for r in refs] + ["mean"]
        pos = np.arange(len(labels))
        means = [float(v.mean()) for v in vals]
        lo = [float(v.min()) for v in vals]
        hi = [float(v.max()) for v in vals]
        colors = [color] * (len(labels) - 1) + ["#59616b"]
        ax.bar(pos, means, width=0.62, color=colors, alpha=0.82)
        ax.vlines(pos, lo, hi, color="#20242a", lw=0.9)
        ax.set_xticks(pos, labels, rotation=90 if len(labels) > 12 else 0)
        ax.set_title(title)
        ax.set_xlabel("reference view")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8eaed", lw=0.6)
        if col == 4:
            ax.set_ylabel("%")
        elif col in {5, 6}:
            ax.set_ylabel("deg")

    fig.suptitle(
        f"{selector_label} "
        f"(all-pair median parallax {summary['parallax_deg_median']:.1f} deg)",
        y=1.02,
        fontsize=9,
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
               fill: str = "#20242a") -> None:
    draw.text(xy, text, fill=fill, font=ImageFont.load_default())


def _save_figure_pil(path: Path, scene_label: str, selector_label: str,
                     centers: np.ndarray, refs: list[int], src: np.ndarray,
                     rows: np.ndarray, summary: dict[str, float],
                     obj_axis: dict[str, object] | None = None) -> None:
    """Small no-matplotlib fallback for login nodes without plotting deps."""
    n_labels = len(refs) + 1
    W, H = max(1260, 520 + 38 * n_labels), 610
    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im, "RGBA")
    palette = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]

    _draw_text(
        draw,
        (22, 14),
        f"{selector_label} for {scene_label} "
        f"(all-pair median parallax {summary['parallax_deg_median']:.1f} deg)",
        "#111111",
    )

    # Camera layout, projected to x-z.
    cam_box = (34, 55, 420, 560)
    draw.rectangle(cam_box, outline="#d7dce2")
    _draw_text(draw, (cam_box[0], cam_box[1] - 18), "camera layout (x-z)")
    xz = centers[:, [0, 2]]
    lo = xz.min(axis=0)
    hi = xz.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        x = cam_box[0] + 12 + int((p[0] - lo[0]) / span[0] * (cam_box[2] - cam_box[0] - 24))
        y = cam_box[3] - 12 - int((p[1] - lo[1]) / span[1] * (cam_box[3] - cam_box[1] - 24))
        return x, y

    for p in xz:
        x, y = to_px(p)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#b8c0ca")
    ox, oy = to_px(np.array([0.0, 0.0]))
    draw.line((ox - 7, oy, ox + 7, oy), fill="#20242a", width=1)
    draw.line((ox, oy - 7, ox, oy + 7), fill="#20242a", width=1)
    if obj_axis is not None:
        center_xz = np.asarray(obj_axis["center_xz"], dtype=np.float64)
        axis_xz = np.asarray(obj_axis["axis_xz"], dtype=np.float64)
        length = 0.20 * float(np.max(np.ptp(xz, axis=0)))
        p0 = center_xz - 0.5 * length * axis_xz
        p1 = center_xz + 0.5 * length * axis_xz
        x0, y0 = to_px(p0)
        x1, y1 = to_px(p1)
        cx, cy = to_px(center_xz)
        draw.line((x0, y0, x1, y1), fill="#111111", width=3)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline="#111111", width=2)
        _draw_text(draw, (cx + 6, cy - 8), "rough object axis", "#111111")
    for k, r in enumerate(refs):
        color = palette[k % len(palette)]
        rx, ry = to_px(xz[r])
        for s in src[r]:
            sx, sy = to_px(xz[s])
            draw.line((rx, ry, sx, sy), fill=color + "90", width=1)
        draw.ellipse((rx - 5, ry - 5, rx + 5, ry + 5), fill=color, outline="#111111")
        _draw_text(draw, (rx + 6, ry - 5), str(r), color)

    ref_mask = np.isin(rows[:, 0].astype(int), refs)
    ref_rows = rows[ref_mask]
    charts = [
        (465, 70, W - 45, 172, 5, "object-center parallax", "deg", "#0072b2"),
        (465, 205, W - 45, 307, 3, "camera-center distance", "norm.", "#d55e00"),
        (465, 340, W - 45, 442, 4, "baseline / ref distance", "%", "#009e73"),
        (465, 475, W - 45, 577, 6, "optical-axis angle", "deg", "#cc79a7"),
    ]
    for x0, y0, x1, y1, col, title, unit, color in charts:
        draw.rectangle((x0, y0, x1, y1), outline="#d7dce2")
        _draw_text(draw, (x0, y0 - 18), f"{title} ({unit})")
        vals = [ref_rows[ref_rows[:, 0] == r, col] for r in refs]
        vals.append(ref_rows[:, col])
        labels = [str(r) for r in refs] + ["mean"]
        ymax = max(float(max(v.max() for v in vals)), 1e-6)
        ymax *= 1.12
        draw.line((x0 + 34, y1 - 22, x1 - 12, y1 - 22), fill="#cdd3da")
        draw.line((x0 + 34, y0 + 12, x0 + 34, y1 - 22), fill="#cdd3da")
        _draw_text(draw, (x0 + 2, y0 + 10), f"{ymax:.1f}")
        plot_w = x1 - x0 - 58
        step = plot_w / max(len(labels), 1)
        bar_half = max(4, min(18, int(step * 0.28)))
        for i, (label, v) in enumerate(zip(labels, vals)):
            mean = float(v.mean())
            lo_v = float(v.min())
            hi_v = float(v.max())
            cx = x0 + 45 + int(i * step + step * 0.5)
            base = y1 - 22
            bar_h = int(mean / ymax * (y1 - y0 - 44))
            lo_y = base - int(lo_v / ymax * (y1 - y0 - 44))
            hi_y = base - int(hi_v / ymax * (y1 - y0 - 44))
            fill = "#59616bd8" if label == "mean" else color + "d8"
            draw.rectangle((cx - bar_half, base - bar_h, cx + bar_half, base), fill=fill)
            draw.line((cx, lo_y, cx, hi_y), fill="#20242a", width=2)
            _draw_text(draw, (cx - 10, y1 - 18), str(label))
            if step >= 28 or label == "mean":
                _draw_text(draw, (cx - 14, base - bar_h - 13), f"{mean:.1f}", "#30343b")

    im.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("data/tnt/Ignatius"))
    ap.add_argument("--out-dir", type=Path, default=Path("_diagnostics/ignatius_closest_view_stats"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--ref-views", default="paper6",
                    help="'paper6', 'many'/'paper24', 'evenlyN', 'all', or comma/space ids.")
    ap.add_argument("--n-src", type=int, default=6)
    ap.add_argument("--min-parallax-deg", type=float, default=0.0,
                    help="If >0, pick the closest source views with theta_rs(x0) above this threshold.")
    args = ap.parse_args()

    dataset_kind, c2ws, centers, rgb_paths = _load_scene_cameras(args.scene)
    scene_label = args.label or args.scene.name
    prefix = scene_label.lower().replace(" ", "_")
    refs = _parse_ref_views(args.ref_views, len(c2ws))
    obj = _object_center(args.scene, centers)
    obj_axis = _object_axis_xz(args.scene)
    src, selection_info = _select_sources_by_distance_and_parallax(
        centers,
        obj,
        args.n_src,
        args.min_parallax_deg,
    )
    if args.min_parallax_deg > 0.0:
        selector_label = f"{args.n_src} closest source views, parallax >= {args.min_parallax_deg:g} deg"
    else:
        selector_label = f"{args.n_src} nearest source views per reference"
    rows, summary = _pair_stats(c2ws, centers, src, obj)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / f"{prefix}_closest_view_stats.csv", rows)
    _write_reference_summary_csv(
        args.out_dir / f"{prefix}_closest_view_reference_summary.csv",
        rows,
        refs,
    )
    (args.out_dir / f"{prefix}_closest_view_summary.json").write_text(
        json.dumps({
            "scene": str(args.scene),
            "dataset_kind": dataset_kind,
            "scene_label": scene_label,
            "view_selection": selection_info,
            "reference_views_shown": refs,
            "sources_per_shown_ref": {str(r): src[r].astype(int).tolist() for r in refs},
            "shown_reference_summary_with_mean": _reference_summary(rows, refs),
            "object_center_norm": obj.tolist(),
            "rough_object_axis_xz": obj_axis,
            "summary_all_refs": summary,
        }, indent=2) + "\n"
    )
    _save_figure(
        args.out_dir / f"{prefix}_closest_view_stats.png",
        scene_label,
        selector_label,
        centers,
        refs,
        src,
        rows,
        summary,
        obj_axis,
    )
    _save_thumbnail_strip(args.out_dir / f"{prefix}_closest_view_thumbs.png", rgb_paths, refs, src)

    print(f"loaded {len(c2ws)} {scene_label} train views ({dataset_kind})")
    print(f"reference views shown: {refs}")
    print(f"wrote {args.out_dir / f'{prefix}_closest_view_stats.png'}")
    print(f"wrote {args.out_dir / f'{prefix}_closest_view_thumbs.png'}")
    print(f"all refs: median parallax={summary['parallax_deg_median']:.2f} deg, "
          f"median baseline={summary['baseline_median']:.3f} normalized units")


if __name__ == "__main__":
    main()
