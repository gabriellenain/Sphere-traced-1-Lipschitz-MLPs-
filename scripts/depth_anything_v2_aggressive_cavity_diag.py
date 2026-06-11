"""Aggressive single-view DA-V2 carve diagnostic for the DTU scan118 cavity.

Uses the existing normalized DA-V2 proposal q for frontal view 23, thresholds
it into a binary mask, and carves selected rays over

    [t_BA - h, t_BA + lambda]

for lambda in {0, 16h, 32h, 64h, 128h}. The untouched H_BA occupancy is kept
as a separate reference because the literal lambda=0 case still removes the
current first-hit voxel layer.

This is intentionally unsafe and visual-only. It does not use multi-view
voting, neural fitting, training, or photometric verification.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
from scipy.ndimage import distance_transform_edt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views
from lip_tracer.visual_hull import carve
from scripts.depth_anything_v2_hull_diag import (
    DEFAULT_SCENE,
    _make_rays,
    _sample_first_hit,
    _shade_hits,
)


DEFAULT_PROPOSALS = Path(
    "artifacts/depth_anything_v2_scan118_carve_ablation/"
    "scan118_views20-26_carve_ablation.npz"
)
DEFAULT_DEPTHS = Path(
    "artifacts/depth_anything_v2_scan118_front_grid/"
    "scan118_views20-26_depth_arrays.npz"
)
DEFAULT_OUT = Path("artifacts/depth_anything_v2_scan118_aggressive_cavity")


def _view_row(view_ids: np.ndarray, view: int) -> int:
    rows = np.flatnonzero(view_ids == view)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one cached row for view {view}, got {len(rows)}")
    return int(rows[0])


def _cast_single_view_removals(
    occ: np.ndarray,
    views: dict,
    view: int,
    t_ba: np.ndarray,
    selected_mask: np.ndarray,
    *,
    bound: float,
    lambda_voxels: tuple[int, ...],
    ray_step_voxels: float,
    ray_chunk: int,
) -> np.ndarray:
    """Return occupied-voxel removal masks for the requested amplitudes."""
    res = occ.shape[0]
    voxel = 2.0 * bound / max(res - 1, 1)
    proposal_down = views["H"] // t_ba.shape[0]
    if views["W"] // proposal_down != t_ba.shape[1]:
        raise ValueError("cached t_BA size is incompatible with scene image size")
    origins, dirs, _, _ = _make_rays(
        views["c2w"][view].numpy(),
        views["K"][view].numpy(),
        views["H"],
        views["W"],
        proposal_down,
    )
    t_flat = t_ba.reshape(-1)
    selected = np.flatnonzero(selected_mask.reshape(-1) & np.isfinite(t_flat))
    print(
        f"[carve] frontal view={view}, selected rays={len(selected):,} "
        f"({100.0 * len(selected) / t_flat.size:.2f}%)",
        flush=True,
    )
    removals = np.zeros((len(lambda_voxels), *occ.shape), dtype=bool)
    for start in range(0, len(selected), ray_chunk):
        ridx = selected[start:start + ray_chunk]
        o, d = origins[ridx], dirs[ridx]
        t0 = t_flat[ridx]
        for level, mult in enumerate(lambda_voxels):
            offsets = (
                np.arange(
                    -1.0,
                    mult + ray_step_voxels,
                    ray_step_voxels,
                    dtype=np.float32,
                )
                * voxel
            )
            xyz = o[:, None, :] + (t0[:, None] + offsets[None, :])[..., None] * d[:, None, :]
            grid = np.rint((xyz + bound) / voxel).astype(np.int32)
            in_bounds = np.all((grid >= 0) & (grid < res), axis=-1)
            grid = np.clip(grid, 0, res - 1)
            ix, iy, iz = grid[..., 0], grid[..., 1], grid[..., 2]
            active = in_bounds & occ[iz, iy, ix]
            removals[level, iz[active], iy[active], ix[active]] = True
    print(
        "[carve] removed occupied voxels: "
        + ", ".join(
            f"{mult}h={int(removals[level].sum()):,}"
            for level, mult in enumerate(lambda_voxels)
        ),
        flush=True,
    )
    return removals


def _render_occupancy(
    occ: np.ndarray,
    views: dict,
    render_views: tuple[int, ...],
    *,
    bound: float,
    render_down: int,
    ray_step_voxels: float,
    ray_chunk: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Phong images, hit maps, and first-hit depths for one occupancy."""
    voxel = 2.0 * bound / max(occ.shape[0] - 1, 1)
    print(f"[render] {label}: computing EDT SDF", flush=True)
    sdf = ((distance_transform_edt(~occ) - distance_transform_edt(occ)) * voxel).astype(
        np.float32
    )
    gradients = np.gradient(sdf, voxel, voxel, voxel)
    phongs, hits, depths = [], [], []
    for view in render_views:
        print(f"[render] {label}: view {view}", flush=True)
        origins, dirs, Hr, Wr = _make_rays(
            views["c2w"][view].numpy(),
            views["K"][view].numpy(),
            views["H"],
            views["W"],
            render_down,
        )
        depth, hit = _sample_first_hit(
            occ,
            origins,
            dirs,
            bound,
            step_voxels=ray_step_voxels,
            chunk=ray_chunk,
        )
        phong = _shade_hits(
            occ,
            origins,
            dirs,
            depth,
            hit,
            bound,
            gradients=gradients,
        )
        phongs.append(phong.reshape(Hr, Wr, 3))
        hits.append(hit.reshape(Hr, Wr))
        depths.append(depth.reshape(Hr, Wr))
    del sdf, gradients
    gc.collect()
    return np.stack(phongs), np.stack(hits), np.stack(depths)


def _save_binary_mask(
    out: Path,
    rgb: np.ndarray,
    q: np.ndarray,
    selected: np.ndarray,
    *,
    view: int,
    tau_q: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB view {view}")
    im = axes[1].imshow(q, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].contour(selected, levels=[0.5], colors=["#00e5ff"], linewidths=0.8)
    axes[1].set_title(r"Existing proposal $q$")
    axes[2].imshow(selected, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(rf"Binary mask $B=[q>{tau_q:g}]$")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[1], shrink=0.75, pad=0.02)
    cb.set_label("normalized proposal q")
    fig.suptitle("Aggressive DA-V2 frontal cavity selection")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _save_comparison(
    out: Path,
    row_labels: tuple[str, ...],
    render_views: tuple[int, ...],
    phongs: np.ndarray,
    hits: np.ndarray,
    depth_delta: np.ndarray,
    silhouette_iou: np.ndarray,
) -> None:
    valid_delta = depth_delta[1:][np.isfinite(depth_delta[1:])]
    vmax = float(np.percentile(valid_delta, 99)) if valid_delta.size else 1.0
    vmax = max(vmax, 1e-6)
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(
        len(row_labels),
        3 * len(render_views),
        figsize=(12.0, 2.5 * len(row_labels)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, label in enumerate(row_labels):
        for group, view in enumerate(render_views):
            col = 3 * group
            axes[row, col].imshow(phongs[row, group])
            axes[row, col + 1].imshow(hits[row, group], cmap="gray_r", vmin=0, vmax=1)
            im = axes[row, col + 2].imshow(
                depth_delta[row, group],
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
            )
            axes[row, col].set_title(
                f"view {view}: Phong\nhit IoU={silhouette_iou[row, group]:.4f}",
                fontsize=9,
            )
            axes[row, col + 1].set_title("silhouette / hit", fontsize=9)
            axes[row, col + 2].set_title(r"$t_{\rm hit}-t_{\rm BA}$", fontsize=9)
            axes[row, col].set_ylabel(label)
            for ax in axes[row, col:col + 3]:
                ax.set_xticks([])
                ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[:, 2::3], shrink=0.65, pad=0.012)
    cb.set_label("first-hit depth increase from H_BA (world units)")
    fig.suptitle(
        "DTU scan118 aggressive DA-V2-selected frontal carve\n"
        "reference plus literal [t_BA-h, t_BA+lambda] segments; no safety constraints",
        fontsize=12,
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    ap.add_argument("--depths", type=Path, default=DEFAULT_DEPTHS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--front-view", type=int, default=23)
    ap.add_argument("--tau-q", type=float, default=0.15)
    ap.add_argument("--hull-res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--render-down", type=int, default=2)
    ap.add_argument("--ray-step-voxels", type=float, default=0.5)
    ap.add_argument("--ray-chunk", type=int, default=2048)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    proposal_data = np.load(args.proposals)
    depth_data = np.load(args.depths)
    proposal_row = _view_row(proposal_data["views"], args.front_view)
    depth_row = _view_row(depth_data["views"], args.front_view)
    q = proposal_data["q"][proposal_row]
    selected = q > args.tau_q
    print(
        f"[setup] q source={args.proposals}, front view={args.front_view}, "
        f"tau_q={args.tau_q:g}, selected={int(selected.sum()):,}/{selected.size:,}",
        flush=True,
    )
    mask_path = args.out_dir / "scan118_view23_binary_proposal.png"
    _save_binary_mask(
        mask_path,
        depth_data["rgb"][depth_row],
        q,
        selected,
        view=args.front_view,
        tau_q=args.tau_q,
    )
    print(f"[save] binary proposal: {mask_path}", flush=True)

    print(
        f"[hull] carving current border-aware hull: res={args.hull_res}, "
        f"bound={args.bound:g}",
        flush=True,
    )
    occ_ba = carve(
        scene=args.scene,
        res=args.hull_res,
        bound=args.bound,
        border_aware=True,
    )
    n_ba = int(occ_ba.sum())
    print(f"[hull] H_BA occupied={n_ba:,}/{occ_ba.size:,}", flush=True)
    views = load_views(args.scene)
    lambda_voxels = (0, 16, 32, 64, 128)
    removals = _cast_single_view_removals(
        occ_ba,
        views,
        args.front_view,
        depth_data["t_ba_raw"][depth_row],
        selected,
        bound=args.bound,
        lambda_voxels=lambda_voxels,
        ray_step_voxels=args.ray_step_voxels,
        ray_chunk=args.ray_chunk,
    )
    occupancies = np.stack([occ_ba, *(occ_ba & ~remove for remove in removals)])
    removed_counts = [0, *(int(remove.sum()) for remove in removals)]
    removed_percent = [100.0 * count / n_ba for count in removed_counts]
    print(
        "[carve] reference and removal percentages: "
        + ", ".join(
            f"{label}={count:,} ({percent:.3f}%)"
            for label, count, percent in zip(
                ("H_BA ref", "0h", "16h", "32h", "64h", "128h"),
                removed_counts,
                removed_percent,
            )
        ),
        flush=True,
    )

    render_views = (23, 20, 26)
    row_labels = ("H_BA ref", "lambda=0h", "lambda=16h", "lambda=32h", "lambda=64h", "lambda=128h")
    phong_rows, hit_rows, depth_rows = [], [], []
    for label, occ in zip(row_labels, occupancies):
        phong, hit, depth = _render_occupancy(
            occ,
            views,
            render_views,
            bound=args.bound,
            render_down=args.render_down,
            ray_step_voxels=args.ray_step_voxels,
            ray_chunk=args.ray_chunk,
            label=label,
        )
        phong_rows.append(phong)
        hit_rows.append(hit)
        depth_rows.append(depth)
    phongs = np.stack(phong_rows)
    hits = np.stack(hit_rows)
    depths = np.stack(depth_rows)
    base_hits = hits[0]
    base_depth = depths[0]
    both_hit = hits & base_hits[None]
    depth_delta = np.where(both_hit, depths - base_depth[None], np.nan)
    intersection = np.logical_and(hits, base_hits[None]).sum(axis=(2, 3))
    union = np.logical_or(hits, base_hits[None]).sum(axis=(2, 3))
    silhouette_iou = intersection / np.maximum(union, 1)

    comparison_path = args.out_dir / "scan118_view23_aggressive_carve_comparison.png"
    _save_comparison(
        comparison_path,
        row_labels,
        render_views,
        phongs,
        hits,
        depth_delta,
        silhouette_iou,
    )
    print(f"[save] comparison figure: {comparison_path}", flush=True)

    arrays_path = args.out_dir / "scan118_view23_aggressive_carve.npz"
    np.savez_compressed(
        arrays_path,
        front_view=np.asarray(args.front_view),
        render_views=np.asarray(render_views),
        tau_q=np.asarray(args.tau_q),
        q=q,
        binary_proposal=selected,
        lambda_voxels=np.asarray(lambda_voxels),
        row_labels=np.asarray(row_labels),
        hull_res=np.asarray(args.hull_res),
        hull_bound=np.asarray(args.bound),
        occupancy=occupancies,
        removal_mask=removals,
        removed_voxels=np.asarray(removed_counts),
        removed_percent=np.asarray(removed_percent),
        hit=hits,
        depth=depths,
        depth_delta_vs_ba=depth_delta,
        silhouette_iou_vs_ba=silhouette_iou,
    )
    delta_stats = []
    for row, label in enumerate(row_labels):
        by_view = []
        for group, view in enumerate(render_views):
            values = depth_delta[row, group][np.isfinite(depth_delta[row, group])]
            by_view.append(
                {
                    "view": view,
                    "mean": float(values.mean()) if values.size else None,
                    "p95": float(np.percentile(values, 95)) if values.size else None,
                    "max": float(values.max()) if values.size else None,
                }
            )
        delta_stats.append({"label": label, "by_view": by_view})
    summary = {
        "front_view": args.front_view,
        "render_views": list(render_views),
        "tau_q": args.tau_q,
        "selected_rays": int(selected.sum()),
        "hull_res": args.hull_res,
        "bound": args.bound,
        "row_labels": list(row_labels),
        "lambda_voxels": ["H_BA reference", *lambda_voxels],
        "removed_voxels": removed_counts,
        "removed_percent": removed_percent,
        "occupied_voxels": [int(occ.sum()) for occ in occupancies],
        "silhouette_iou_vs_ba": silhouette_iou.tolist(),
        "first_hit_delta_vs_ba": delta_stats,
        "safety_constraints": "none; aggressive visual diagnostic only",
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[save] arrays: {arrays_path}", flush=True)
    print(f"[save] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
