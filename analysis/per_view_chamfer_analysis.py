"""Per-view Chamfer accuracy analysis: correlates NCC quality with geometry degradation.

Produces:
  - per_view_chamfer.json   raw numbers
  - per_view_chamfer.png    scatter zncc_before vs Δaccuracy  (for paper)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_dtu_chamfer import (
    _crop_to_bbox, _in_obs, _infer_scan_id, _load_dtu_gt,
    _mask_filter_gt, _nn_metrics, _to_world,
)
from refine_depths import _load_ply_verts
from lip_tracer.train import load_config_json


def _accuracy(pred_world, gt_eval, gt_info, max_dist=20.0, device="cuda"):
    pred = pred_world[_in_obs(pred_world, gt_info["ObsMask"], gt_info["BB"], gt_info["Res"])]
    if len(pred) < 10:
        return None, None
    keep_bb, _ = _crop_to_bbox(gt_eval, pred, 0.02)
    gt_v = gt_eval[keep_bb]
    if len(gt_v) < 10:
        return None, None
    metrics, pred_to_gt, _ = _nn_metrics(pred, gt_v, max_dist=max_dist, device=device)
    return metrics["accuracy"], pred_to_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run", default="outputs/run_20260510_094943_scan65")
    ap.add_argument("--dtu-eval-dir",
                    default="/scratch/_projets_/willow/1-lip-tracer/data/dtu_eval/SampleSet/MVS Data")
    ap.add_argument("--down", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cfg = load_config_json(Path(args.run) / "config.json")
    scene_path = Path(cfg.scene)
    scan_id = _infer_scan_id(scene_path)
    scale_mat = np.load(scene_path / "cameras.npz")["scale_mat_0"].astype(np.float64)

    gt_pts_obs, gt_info = _load_dtu_gt(Path(args.dtu_eval_dir), scan_id)
    keep_mf, _ = _mask_filter_gt(gt_pts_obs, scene_path)
    gt_eval = gt_pts_obs[keep_mf]
    print(f"GT: {len(gt_eval):,} pts after mask filter", flush=True)

    tag = f"down{args.down}"
    ncc_files = sorted(out_dir.glob(f"view*_{tag}_metrics.json"))
    rows = []
    for ncc_f in ncc_files:
        ncc = json.loads(ncc_f.read_text())
        vi = ncc["view"]
        zncc_b = ncc["train_before"]["zncc_mean"]
        zncc_a = ncc["train_after"]["zncc_mean"]

        accs = {}
        for kind in ("before", "after"):
            ply = out_dir / f"view{vi:02d}_{tag}_points_{kind}.ply"
            if not ply.exists():
                ply = out_dir / f"view{vi:02d}_{tag}_{kind}.ply"
            if not ply.exists():
                accs[kind] = None
                continue
            pts_world = _to_world(_load_ply_verts(ply), scale_mat)
            acc, _ = _accuracy(pts_world, gt_eval, gt_info, device=args.device)
            accs[kind] = acc

        if accs["before"] is None or accs["after"] is None:
            continue
        delta = accs["after"] - accs["before"]
        rows.append({"view": vi, "zncc_before": zncc_b, "zncc_after": zncc_a,
                     "acc_before": accs["before"], "acc_after": accs["after"],
                     "delta_acc": delta})
        print(f"  view {vi:02d}  zncc_b={zncc_b:.3f}  acc_b={accs['before']:.3f}"
              f"  acc_a={accs['after']:.3f}  Δ={delta:+.3f}", flush=True)

    out = out_dir / "per_view_chamfer.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")

    # ── scatter plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        zb  = np.array([r["zncc_before"] for r in rows])
        da  = np.array([r["delta_acc"]   for r in rows])
        vis = [r["view"] for r in rows]

        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(zb, da, c=da, cmap="RdYlGn_r", vmin=da.min(), vmax=max(da.max(), 0.1),
                        s=40, zorder=3)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        for v, x, y in zip(vis, zb, da):
            if y > np.percentile(da, 85) or y < np.percentile(da, 15):
                ax.annotate(str(v), (x, y), fontsize=6, ha="left", va="bottom")
        fig.colorbar(sc, ax=ax, label="Δ accuracy (mm, after−before)")
        ax.set_xlabel("NCC score before refinement (zncc)")
        ax.set_ylabel("Δ accuracy (mm)")
        ax.set_title("Per-view: NCC quality vs geometry degradation")
        r = np.corrcoef(zb, da)[0, 1]
        ax.text(0.97, 0.97, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        fig.tight_layout()
        png = out_dir / "per_view_chamfer.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"saved -> {png}")
        print(f"\ncorrelation zncc_before vs Δaccuracy: r={r:.3f}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
