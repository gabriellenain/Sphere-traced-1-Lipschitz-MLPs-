"""Post-hoc Chamfer evaluation for analysis/refine_depths.py point clouds."""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_dtu_chamfer import (
    _crop_to_bbox, _in_obs, _infer_scan_id, _load_dtu_gt, _mask_filter_gt,
    _nn_metrics, _render_error_png, _to_world,
)
from refine_depths import _chamfer_from_pts, _load_ply_verts, save_ply
from lip_tracer.train import load_config_json


def _load_points(out_dir: Path, down: int, kind: str, views: list[int] | None) -> np.ndarray | None:
    tag = f"down{down}"
    if views is None:
        paths = sorted(out_dir.glob(f"view*_{tag}_points_{kind}.ply"))
        if not paths:
            paths = sorted(out_dir.glob(f"view*_{tag}_{kind}.ply"))
    else:
        paths = []
        for vi in views:
            p = out_dir / f"view{vi:02d}_{tag}_points_{kind}.ply"
            if not p.exists():
                p = out_dir / f"view{vi:02d}_{tag}_{kind}.ply"
            if p.exists():
                paths.append(p)
    if not paths:
        return None
    return np.concatenate([_load_ply_verts(p) for p in paths], axis=0)


def _chamfer_depth_cloud(pts_world: np.ndarray, gt_pts: np.ndarray, gt_info: dict,
                         scene_path: Path, device: str, label: str,
                         n_sample: int, max_dist: float = 20.0,
                         gt_crop_pred_bbox: bool = True,
                         gt_crop_bbox_padding: float = 0.02,
                         gt_max_dist_to_pred: float | None = 20.0,
                         out_png: Path | None = None,
                         png_vmax: float | None = None) -> dict | None:
    rng = np.random.default_rng(0)
    if len(pts_world) > n_sample:
        pts_world = pts_world[rng.choice(len(pts_world), n_sample, replace=False)]

    pred = pts_world[_in_obs(pts_world, gt_info["ObsMask"], gt_info["BB"], gt_info["Res"])]
    if len(pred) < 100:
        print(f"  [chamfer {label}] only {len(pred)} pts in ObsMask -- skipping", flush=True)
        return None

    keep_gt, _ = _mask_filter_gt(gt_pts, scene_path)
    gt_eval = gt_pts[keep_gt]

    if gt_crop_pred_bbox:
        keep_bb, _ = _crop_to_bbox(gt_eval, pred, gt_crop_bbox_padding)
        gt_eval = gt_eval[keep_bb]
        print(f"  [chamfer {label}] pred-bbox crop => {len(gt_eval):,} GT pts", flush=True)

    if gt_max_dist_to_pred is not None:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pred).query(gt_eval, k=1, workers=-1)
        gt_eval = gt_eval[d <= gt_max_dist_to_pred]
        print(f"  [chamfer {label}] max-dist-to-pred={gt_max_dist_to_pred} => {len(gt_eval):,} GT pts", flush=True)

    metrics, pred_to_gt, gt_to_pred = _nn_metrics(pred, gt_eval, max_dist=max_dist, device=device)
    print(f"  [chamfer {label}]  accuracy={metrics['accuracy']:.4f}  "
          f"completeness={metrics['completeness']:.4f}  chamfer={metrics['chamfer']:.4f}  "
          f"(pred={len(pred):,}  gt={len(gt_eval):,})", flush=True)
    if out_png is not None:
        _render_error_png(pred, pred_to_gt, gt_eval, gt_to_pred,
                          out_png, vmax=png_vmax, align_pca=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="outputs/depth_refine_JOBID directory")
    ap.add_argument("--run", default="outputs/run_20260510_094943_scan65")
    ap.add_argument("--dtu-eval-dir",
                    default="/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS Data")
    ap.add_argument("--down", type=int, default=2)
    ap.add_argument("--views", type=int, nargs="+")
    ap.add_argument("--n-chamfer-pts", type=int, default=1_000_000)
    ap.add_argument("--legacy-crop", action="store_true",
                    help="use refine_depths' old pred-bbox GT crop for backwards comparison")
    ap.add_argument("--no-gt-crop-pred-bbox", action="store_true",
                    help="disable pred-bbox GT crop (enabled by default to match compare_dtu_chamfer)")
    ap.add_argument("--gt-crop-bbox-padding", type=float, default=0.02,
                    help="world-unit padding around pred bbox for GT crop (default 0.02)")
    ap.add_argument("--no-gt-max-dist-to-pred", action="store_true",
                    help="disable max-dist-to-pred GT filter (enabled at 20mm by default)")
    ap.add_argument("--gt-max-dist-to-pred", type=float, default=20.0,
                    help="keep only GT pts within this distance of pred (default 20.0 mm)")
    ap.add_argument("--no-png", action="store_true",
                    help="do not save Chamfer error PNGs")
    ap.add_argument("--png-vmax", type=float, default=None,
                    help="colour scale maximum in mm; default uses 95th percentile")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cfg = load_config_json(Path(args.run) / "config.json")
    scene_path = Path(cfg.scene)
    scan_id = _infer_scan_id(scene_path)

    cam_dict = np.load(scene_path / "cameras.npz")
    scale_mat = cam_dict["scale_mat_0"].astype(np.float64)
    gt_pts, gt_info = _load_dtu_gt(Path(args.dtu_eval_dir), scan_id)

    print(f"[chamfer] out_dir={out_dir}")
    print(f"[chamfer] scan{scan_id}  GT={len(gt_pts):,} pts in ObsMask")

    summary = {}
    for kind in ("before", "after"):
        pts_obj = _load_points(out_dir, args.down, kind, args.views)
        if pts_obj is None:
            print(f"[chamfer] no {kind} point files found")
            continue
        pts_world = _to_world(pts_obj, scale_mat)
        tag = f"down{args.down}"
        save_ply(out_dir / f"depth_points_{kind}_{tag}.ply",
                 pts_obj.astype(np.float32),
                 np.zeros((0, 3), dtype=np.int32))
        save_ply(out_dir / f"depth_points_{kind}_{tag}_world.ply",
                 pts_world.astype(np.float32),
                 np.zeros((0, 3), dtype=np.int32))
        if args.legacy_crop:
            summary[kind] = _chamfer_from_pts(
                pts_world, gt_pts, gt_info, scene_path, "cuda",
                label=f"depth {kind:6s}", n_sample=args.n_chamfer_pts)
        else:
            summary[kind] = _chamfer_depth_cloud(
                pts_world, gt_pts, gt_info, scene_path, "cuda",
                label=f"depth {kind:6s}", n_sample=args.n_chamfer_pts,
                gt_crop_pred_bbox=not args.no_gt_crop_pred_bbox,
                gt_crop_bbox_padding=args.gt_crop_bbox_padding,
                gt_max_dist_to_pred=None if args.no_gt_max_dist_to_pred else args.gt_max_dist_to_pred,
                out_png=None if args.no_png else out_dir / f"chamfer_error_{kind}.png",
                png_vmax=args.png_vmax)

    if "before" in summary and "after" in summary:
        summary["delta"] = summary["after"]["chamfer"] - summary["before"]["chamfer"]
        print(f"[chamfer] before={summary['before']['chamfer']:.4f}  "
              f"after={summary['after']['chamfer']:.4f}  "
              f"delta={summary['delta']:+.4f}")

    (out_dir / "chamfer_metrics_posthoc.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
