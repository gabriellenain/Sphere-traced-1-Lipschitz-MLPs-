#!/usr/bin/env python3
"""Visualise MASt3R depth quality for an IDR-format DTU scan.

Produces:
  - outputs/depth_quality_scan<N>/contact_vis.png   — coloured depth maps for 12 evenly-spaced views
  - outputs/depth_quality_scan<N>/contact_align.png — aligned depth vs raw, with SFM residuals
  - outputs/depth_quality_scan<N>/alignment_stats.png — per-view scale / shift / residual plot
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

REPO = Path(__file__).parent


def _load_rgb(img_dir: Path, idx: int) -> np.ndarray | None:
    candidates = sorted(img_dir.glob("*.png"))
    if idx >= len(candidates):
        return None
    import cv2
    bgr = cv2.imread(str(candidates[idx]))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path,
                    default=Path("/scratch/_projets_/willow/1-lip-tracer/data/dtu_idr/scan122"))
    ap.add_argument("--depth-dir", type=Path, default=None,
                    help="depth dir (default: scene/mast3r_depth_1600x1200)")
    ap.add_argument("--n-views", type=int, default=12, help="views shown in contact sheets")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    depth_dir = args.depth_dir or (args.scene / "mast3r_depth_1600x1200")
    scan_name = args.scene.name
    out_dir   = args.out_dir or (REPO / "outputs" / f"depth_quality_{scan_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    import json
    manifest_path = depth_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.json in {depth_dir}")
    manifest = json.loads(manifest_path.read_text())
    frames   = manifest["frames"]

    V        = len(frames)
    img_dir  = args.scene / "image"
    view_ids = [int(round(i * (V - 1) / (args.n_views - 1))) for i in range(args.n_views)]

    # ---- alignment stats per view (recompute to get residuals) -----------
    from scipy.linalg import rq

    scales, shifts, residuals, valid_fracs = [], [], [], []
    raw_medians, conf_medians = [], []
    for i in range(V):
        depth  = np.load(depth_dir / frames[i]["depth_npy"]).astype(np.float32)
        valid_path = depth_dir / frames[i].get("valid_mask", "")
        if valid_path.exists():
            import cv2
            valid = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE) > 0
        else:
            valid = np.isfinite(depth) & (depth > 0)
        valid_fracs.append(float(valid.mean()))
        raw_medians.append(float(np.median(depth[valid])) if valid.any() else np.nan)
        conf_path = depth_dir / frames[i].get("confidence_npy", "")
        if conf_path.exists():
            conf = np.load(conf_path).astype(np.float32)
            conf_medians.append(float(np.median(conf[valid])) if valid.any() else np.nan)
        else:
            conf_medians.append(np.nan)

    sfm_path = args.scene / "sparse_sfm_points.txt"
    has_sfm = sfm_path.exists()
    if has_sfm:
        cam_dict = np.load(args.scene / "cameras.npz")
        sfm_pts  = np.loadtxt(sfm_path, dtype=np.float64)
        if sfm_pts.ndim == 1:
            sfm_pts = sfm_pts[None]

        for i in range(V):
            P     = cam_dict[f"world_mat_{i}"][:3, :4].astype(np.float64)
            M     = P[:, :3]
            K_cam, R_cam = rq(M)
            sign  = np.sign(np.diag(K_cam)); sign[sign == 0] = 1.0
            T     = np.diag(sign)
            K_cam = K_cam @ T; R_cam = T @ R_cam
            if np.linalg.det(R_cam) < 0:
                K_cam[:, 2] *= -1.0; R_cam[2, :] *= -1.0
            K_cam = K_cam / K_cam[2, 2]
            t_metric = np.linalg.solve(K_cam, P[:, 3])
            cam_center_metric = -R_cam.T @ t_metric
            cam_center_h = np.concatenate([cam_center_metric, [1.0]])
            key_inv = f"scale_mat_inv_{i}"
            scale_mat_inv = (cam_dict[key_inv] if key_inv in cam_dict
                             else np.linalg.inv(cam_dict[f"scale_mat_{i}"]))
            cam_center_norm = (scale_mat_inv @ cam_center_h)[:3]
            t_norm = (-R_cam @ cam_center_norm).astype(np.float32)
            K_f    = K_cam.astype(np.float32)
            R_f    = R_cam.astype(np.float32)

            depth  = np.load(depth_dir / frames[i]["depth_npy"]).astype(np.float32)
            H_d, W_d = depth.shape
            pts_cam = (R_f @ sfm_pts.T).T + t_norm
            z_norm  = pts_cam[:, 2].astype(np.float32)
            u = K_f[0, 0] * pts_cam[:, 0] / (z_norm + 1e-6) + K_f[0, 2]
            v = K_f[1, 1] * pts_cam[:, 1] / (z_norm + 1e-6) + K_f[1, 2]
            ok = (z_norm > 1e-3) & (u >= 0) & (u < W_d) & (v >= 0) & (v < H_d)
            if ok.sum() < 3:
                scales.append(np.nan); shifts.append(np.nan)
                residuals.append(np.nan)
                continue
            ui = np.clip(u[ok].astype(int), 0, W_d - 1)
            vi_arr = np.clip(v[ok].astype(int), 0, H_d - 1)
            m_at = depth[vi_arr, ui].astype(np.float64)
            z_gt = z_norm[ok].astype(np.float64)
            A    = np.stack([m_at, np.ones_like(m_at)], axis=1)
            (a, b), *_ = np.linalg.lstsq(A, z_gt, rcond=None)
            pred = a * m_at + b
            rmse = float(np.sqrt(((pred - z_gt) ** 2).mean()))
            scales.append(float(a)); shifts.append(float(b))
            residuals.append(rmse)

    # ------------------------------------------------------------------ plot 1: contact vis
    n_cols = 4
    n_rows = (args.n_views + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = np.array(axes).reshape(-1)
    for j, vi in enumerate(view_ids):
        import cv2
        vis_path = depth_dir / f"{frames[vi]['image'].split('/')[1].replace('.png', '')}_depth_vis.png"
        # fallback: try stem from depth_npy name
        stem = frames[vi]["depth_npy"].replace("_depth.npy", "")
        vis_path = depth_dir / f"{stem}_depth_vis.png"
        if vis_path.exists():
            bgr = cv2.imread(str(vis_path))
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb = _load_rgb(img_dir, vi)
        if rgb is not None:
            thumb_h = img.shape[0] // 4
            thumb_w = int(thumb_h * rgb.shape[1] / rgb.shape[0])
            thumb   = (cv2.resize(rgb, (thumb_w, thumb_h)) * 255).astype(np.uint8)
            img[:thumb_h, :thumb_w] = thumb
        axes[j].imshow(img)
        axes[j].set_title(f"view {vi}  valid={valid_fracs[vi]:.0%}", fontsize=9)
        axes[j].axis("off")
    for ax in axes[len(view_ids):]:
        ax.axis("off")
    fig.suptitle(f"MASt3R depth maps — {scan_name}  ({depth_dir.name})", fontsize=12)
    fig.tight_layout()
    p1 = out_dir / "contact_vis.png"
    fig.savefig(p1, dpi=120)
    print(f"saved → {p1}")
    plt.close(fig)

    # ------------------------------------------------------------------ plot 2: alignment/raw stats
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    xs = list(range(V))
    if has_sfm:
        axes[0].plot(xs, scales, "o-", ms=3, lw=1, color="steelblue")
        axes[0].set_ylabel("scale (a)"); axes[0].set_title("Per-view LS alignment: scale")
        axes[0].axhline(1.0, color="gray", lw=0.8, ls="--")
    else:
        axes[0].plot(xs, valid_fracs, "o-", ms=3, lw=1, color="seagreen")
        axes[0].set_ylabel("valid fraction"); axes[0].set_title("Per-view valid-mask coverage")
    axes[0].grid(True, alpha=0.3)

    if has_sfm:
        axes[1].plot(xs, shifts, "o-", ms=3, lw=1, color="darkorange")
        axes[1].set_ylabel("shift (b)"); axes[1].set_title("shift")
        axes[1].axhline(0.0, color="gray", lw=0.8, ls="--")
    else:
        axes[1].plot(xs, raw_medians, "o-", ms=3, lw=1, color="darkorange")
        axes[1].set_ylabel("raw depth median"); axes[1].set_title("Raw MASt3R depth median over valid pixels")
    axes[1].grid(True, alpha=0.3)

    if has_sfm:
        axes[2].plot(xs, residuals, "o-", ms=3, lw=1, color="crimson")
        axes[2].set_ylabel("RMSE (normalised units)"); axes[2].set_title("SFM residual after alignment")
    else:
        axes[2].plot(xs, conf_medians, "o-", ms=3, lw=1, color="crimson")
        axes[2].set_ylabel("confidence median"); axes[2].set_title("MASt3R confidence median over valid pixels")
    axes[2].set_xlabel("view index")
    axes[2].grid(True, alpha=0.3)

    title_kind = "alignment quality" if has_sfm else "raw depth quality (no SFM alignment reference)"
    fig.suptitle(f"Depth {title_kind} — {scan_name}", fontsize=12)
    fig.tight_layout()
    p2 = out_dir / "alignment_stats.png"
    fig.savefig(p2, dpi=120)
    print(f"saved → {p2}")
    plt.close(fig)

    # ------------------------------------------------------------------ summary
    import datetime
    res_arr = np.array([r for r in residuals if not np.isnan(r)])
    vf_arr  = np.array(valid_fracs)

    lines = []
    lines.append(f"=== alignment summary ({scan_name}) ===")
    lines.append(f"  date:           {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  depth_dir:      {depth_dir}")
    lines.append(f"  views:          {V}")
    lines.append(f"  valid_frac:     mean={vf_arr.mean():.1%}  min={vf_arr.min():.1%}")
    if has_sfm and len(res_arr):
        lines.append(f"  SFM RMSE:       mean={res_arr.mean():.4f}  p50={np.median(res_arr):.4f}  p90={np.percentile(res_arr,90):.4f}")
        lines.append(f"  scale:          mean={np.nanmean(scales):.4f}  std={np.nanstd(scales):.4f}")
        lines.append(f"  shift:          mean={np.nanmean(shifts):.4f}  std={np.nanstd(shifts):.4f}")
        lines.append("")
        lines.append("  per-view  [idx  valid%   scale    shift    rmse]")
        for i in range(V):
            sc = scales[i] if i < len(scales) else float("nan")
            sh = shifts[i] if i < len(shifts) else float("nan")
            rs = residuals[i] if i < len(residuals) else float("nan")
            lines.append(f"    {i:4d}  {valid_fracs[i]:.1%}  {sc:8.4f}  {sh:8.4f}  {rs:.4f}")
    else:
        lines.append(f"  SFM RMSE:       skipped (no {sfm_path})")
        lines.append(f"  raw depth med:  mean={np.nanmean(raw_medians):.4f}  std={np.nanstd(raw_medians):.4f}")
        lines.append(f"  conf med:       mean={np.nanmean(conf_medians):.4f}  std={np.nanstd(conf_medians):.4f}")
        lines.append("")
        lines.append("  per-view  [idx  valid%   depth_med  conf_med]")
        for i in range(V):
            lines.append(f"    {i:4d}  {valid_fracs[i]:.1%}  {raw_medians[i]:10.4f}  {conf_medians[i]:.4f}")
    lines.append(f"\noutputs in {out_dir}")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    log_path = out_dir / "quality_summary.log"
    log_path.write_text(summary + "\n")
    print(f"saved → {log_path}")


if __name__ == "__main__":
    main()
