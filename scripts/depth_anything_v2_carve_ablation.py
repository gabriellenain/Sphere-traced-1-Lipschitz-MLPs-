"""Offline visual ablation for DA-V2-guided border-aware hull carving.

Uses cached DA-V2 and border-aware-hull first-hit maps for DTU scan118 views
20..26. It forms a rough per-view cavity proposal

    q = max(norm(D) - norm(t_BA) - tau, 0)

erodes proposal support slightly, casts short carve segments from t_BA, and
removes an occupied voxel only when at least two views vote to carve it.

This is intentionally a visual-only experiment. It does not modify the
training pipeline, hull initialization, metric depth alignment, patch/NCC
verification, or neural fitting.
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
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views
from lip_tracer.visual_hull import carve
from scripts.depth_anything_v2_hull_diag import (
    DEFAULT_SCENE,
    _make_rays,
    _normalize_farther,
    _sample_first_hit,
    _shade_hits,
)


DEFAULT_INPUT = Path(
    "artifacts/depth_anything_v2_scan118_front_grid/"
    "scan118_views20-26_depth_arrays.npz"
)
DEFAULT_OUT = Path("artifacts/depth_anything_v2_scan118_carve_ablation")


def _resize_float(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(x.astype(np.float32)).resize(
            (shape[1], shape[0]),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )


def _resize_bool(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(x).resize(
            (shape[1], shape[0]),
            Image.Resampling.NEAREST,
        ),
        dtype=bool,
    )


def _proposal_maps(
    da_inverse_raw: np.ndarray,
    masks_full: np.ndarray,
    t_ba_norm: np.ndarray,
    t_ba_hit: np.ndarray,
    *,
    tau: float,
    erode_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return globally normalized q maps, resized DA depth, and resized masks."""
    target_shape = t_ba_norm.shape[1:]
    da_norms, masks, proposals = [], [], []
    for raw_full, mask_full, t_norm, hit in zip(
        da_inverse_raw,
        masks_full,
        t_ba_norm,
        t_ba_hit,
    ):
        raw = _resize_float(raw_full, target_shape)
        mask = _resize_bool(mask_full, target_shape)
        da_norm = _normalize_farther(raw, mask, raw_larger_is_farther=False)
        valid = mask & hit & np.isfinite(da_norm) & np.isfinite(t_norm)
        q = np.where(valid, np.maximum(da_norm - t_norm - tau, 0.0), 0.0)
        if erode_pixels > 0:
            q *= binary_erosion(q > 0.0, iterations=erode_pixels)
        da_norms.append(da_norm)
        masks.append(mask)
        proposals.append(q.astype(np.float32))

    proposals = np.stack(proposals)
    q_max = float(proposals.max())
    if q_max > 0.0:
        proposals /= q_max
    return proposals, np.stack(da_norms), np.stack(masks)


def _cast_carve_votes(
    occ: np.ndarray,
    views: dict,
    view_ids: np.ndarray,
    t_ba_raw: np.ndarray,
    t_ba_hit: np.ndarray,
    proposals: np.ndarray,
    *,
    bound: float,
    lambda_voxels: tuple[int, ...],
    ray_step_voxels: float,
    ray_chunk: int,
) -> np.ndarray:
    """Return per-amplitude carve vote volumes with shape (L, z, y, x)."""
    res = occ.shape[0]
    voxel = 2.0 * bound / max(res - 1, 1)
    proposal_down = views["H"] // t_ba_raw.shape[1]
    if views["W"] // proposal_down != t_ba_raw.shape[2]:
        raise ValueError("cached t_BA size is incompatible with scene image size")
    votes = np.zeros((len(lambda_voxels), *occ.shape), dtype=np.uint8)

    for row, view in enumerate(view_ids):
        q_flat = proposals[row].reshape(-1)
        t_flat = t_ba_raw[row].reshape(-1)
        hit_flat = t_ba_hit[row].reshape(-1)
        selected = np.flatnonzero(hit_flat & np.isfinite(t_flat) & (q_flat > 0.0))
        print(
            f"[votes] view {int(view)}: proposal rays={len(selected):,} "
            f"({100.0 * len(selected) / len(q_flat):.2f}%)",
            flush=True,
        )
        if len(selected) == 0:
            continue
        origins, dirs, _, _ = _make_rays(
            views["c2w"][view].numpy(),
            views["K"][view].numpy(),
            views["H"],
            views["W"],
            proposal_down,
        )
        marks = np.zeros((len(lambda_voxels), *occ.shape), dtype=bool)
        for start in range(0, len(selected), ray_chunk):
            ridx = selected[start:start + ray_chunk]
            o, d = origins[ridx], dirs[ridx]
            t0, q = t_flat[ridx], q_flat[ridx]
            for level, mult in enumerate(lambda_voxels):
                max_dist = mult * voxel * q
                n_steps = int(np.ceil(max_dist.max() / (ray_step_voxels * voxel))) + 1
                offsets = (
                    np.arange(n_steps, dtype=np.float32)[None, :]
                    * ray_step_voxels
                    * voxel
                )
                active = offsets <= max_dist[:, None]
                xyz = o[:, None, :] + (t0[:, None] + offsets)[..., None] * d[:, None, :]
                grid = np.rint((xyz + bound) / voxel).astype(np.int32)
                grid = np.clip(grid, 0, res - 1)
                ix, iy, iz = grid[..., 0], grid[..., 1], grid[..., 2]
                active &= occ[iz, iy, ix]
                marks[level, iz[active], iy[active], ix[active]] = True
        votes += marks
        print(
            "         occupied voxel marks: "
            + ", ".join(
                f"{mult}h={int(marks[level].sum()):,}"
                for level, mult in enumerate(lambda_voxels)
            ),
            flush=True,
        )
        del marks, origins, dirs
        gc.collect()
    return votes


def _render_occupancy(
    occ: np.ndarray,
    views: dict,
    view_ids: np.ndarray,
    *,
    bound: float,
    render_down: int,
    ray_step_voxels: float,
    ray_chunk: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """EDT-render one occupancy from the requested camera views."""
    voxel = 2.0 * bound / max(occ.shape[0] - 1, 1)
    print(f"[render] {label}: computing EDT SDF", flush=True)
    sdf = ((distance_transform_edt(~occ) - distance_transform_edt(occ)) * voxel).astype(
        np.float32
    )
    gradients = np.gradient(sdf, voxel, voxel, voxel)
    phongs, hits = [], []
    for view in view_ids:
        print(f"[render] {label}: view {int(view)}", flush=True)
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
    del sdf, gradients
    gc.collect()
    return np.stack(phongs), np.stack(hits)


def _save_proposals(
    out: Path,
    view_ids: np.ndarray,
    rgbs: np.ndarray,
    proposals: np.ndarray,
    tau: float,
) -> None:
    fig, axes = plt.subplots(
        len(view_ids),
        2,
        figsize=(8, 3.0 * len(view_ids)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, view in enumerate(view_ids):
        axes[row, 0].imshow(rgbs[row])
        im = axes[row, 1].imshow(proposals[row], cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(f"view {int(view)}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("RGB")
    axes[0, 1].set_title(r"Eroded proposal $q$")
    cb = fig.colorbar(im, ax=axes[:, 1], shrink=0.65, pad=0.02)
    cb.set_label("normalized carve proposal")
    fig.suptitle(rf"DA-V2 cavity proposal: $q=[norm(D)-norm(t_{{BA}})-{tau:g}]_+$")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _save_comparison(
    out: Path,
    view_ids: np.ndarray,
    lambda_all: tuple[int, ...],
    phongs: np.ndarray,
    hits: np.ndarray,
    silhouette_iou: np.ndarray,
) -> None:
    n_views = len(view_ids)
    n_variants = len(lambda_all)
    fig, axes = plt.subplots(
        n_views,
        2 * n_variants,
        figsize=(4.0 * n_variants, 2.6 * n_views),
        constrained_layout=True,
        squeeze=False,
    )
    for row, view in enumerate(view_ids):
        for col, mult in enumerate(lambda_all):
            axes[row, 2 * col].imshow(phongs[col, row])
            axes[row, 2 * col + 1].imshow(hits[col, row], cmap="gray_r", vmin=0, vmax=1)
            axes[row, 2 * col].set_title(
                (r"$H_{\rm BA}$" if mult == 0 else rf"$\lambda={mult}h$")
                + f"\nPhong, hit IoU={silhouette_iou[col, row]:.4f}",
                fontsize=9,
            )
            axes[row, 2 * col + 1].set_title("silhouette / hit", fontsize=9)
            axes[row, 2 * col].set_ylabel(f"view {int(view)}")
            for ax in axes[row, 2 * col:2 * col + 2]:
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(
        "DTU scan118 DA-V2-guided inward-carve ablation\n"
        "two-view vote threshold; EDT step-0 surfaces; no neural fitting",
        fontsize=12,
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--erode-pixels", type=int, default=3)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--hull-res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--render-down", type=int, default=4)
    ap.add_argument("--ray-step-voxels", type=float, default=0.5)
    ap.add_argument("--ray-chunk", type=int, default=2048)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cached = np.load(args.input)
    view_ids = cached["views"].astype(int)
    lambda_voxels = (4, 8, 16)
    lambda_all = (0, *lambda_voxels)
    print(f"[setup] views={view_ids.tolist()}", flush=True)
    print(
        f"[setup] tau={args.tau:g}, erosion={args.erode_pixels}px, "
        f"vote threshold={args.min_votes}, lambdas={lambda_all} * h",
        flush=True,
    )

    proposals, da_norms, masks = _proposal_maps(
        cached["da_inverse_depth_raw"],
        cached["mask"],
        cached["t_ba_depth_farther_norm"],
        cached["t_ba_hit"],
        tau=args.tau,
        erode_pixels=args.erode_pixels,
    )
    proposal_path = args.out_dir / "scan118_views20-26_proposals.png"
    _save_proposals(proposal_path, view_ids, cached["rgb"], proposals, args.tau)
    print(
        f"[proposal] nonzero pixels per view: "
        f"{[int((q > 0).sum()) for q in proposals]}",
        flush=True,
    )
    print(f"[save] proposal figure: {proposal_path}", flush=True)

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
    print(f"[hull] H_BA occupied={int(occ_ba.sum()):,}/{occ_ba.size:,}", flush=True)
    views = load_views(args.scene)
    votes = _cast_carve_votes(
        occ_ba,
        views,
        view_ids,
        cached["t_ba_raw"],
        cached["t_ba_hit"],
        proposals,
        bound=args.bound,
        lambda_voxels=lambda_voxels,
        ray_step_voxels=args.ray_step_voxels,
        ray_chunk=args.ray_chunk,
    )

    removals = votes >= args.min_votes
    occupancies = np.stack([occ_ba, *(occ_ba & ~remove for remove in removals)])
    removed_counts = [0, *(int((occ_ba & remove).sum()) for remove in removals)]
    print(
        "[carve] removed occupied voxels: "
        + ", ".join(f"{mult}h={count:,}" for mult, count in zip(lambda_all, removed_counts)),
        flush=True,
    )

    phong_variants, hit_variants = [], []
    for mult, occ in zip(lambda_all, occupancies):
        phong, hit = _render_occupancy(
            occ,
            views,
            view_ids,
            bound=args.bound,
            render_down=args.render_down,
            ray_step_voxels=args.ray_step_voxels,
            ray_chunk=args.ray_chunk,
            label="H_BA" if mult == 0 else f"lambda={mult}h",
        )
        phong_variants.append(phong)
        hit_variants.append(hit)
    phongs = np.stack(phong_variants)
    hits = np.stack(hit_variants)
    base_hits = hits[0]
    intersection = np.logical_and(hits, base_hits[None]).sum(axis=(2, 3))
    union = np.logical_or(hits, base_hits[None]).sum(axis=(2, 3))
    silhouette_iou = intersection / np.maximum(union, 1)

    comparison_path = args.out_dir / "scan118_views20-26_carve_comparison.png"
    _save_comparison(comparison_path, view_ids, lambda_all, phongs, hits, silhouette_iou)
    print(f"[save] comparison figure: {comparison_path}", flush=True)

    arrays_path = args.out_dir / "scan118_views20-26_carve_ablation.npz"
    np.savez_compressed(
        arrays_path,
        views=view_ids,
        lambda_voxels=np.asarray(lambda_all),
        tau=np.asarray(args.tau),
        erode_pixels=np.asarray(args.erode_pixels),
        min_votes=np.asarray(args.min_votes),
        hull_res=np.asarray(args.hull_res),
        hull_bound=np.asarray(args.bound),
        render_down=np.asarray(args.render_down),
        q=proposals,
        da_depth_farther_norm=da_norms,
        foreground_mask=masks,
        carve_votes=votes,
        occupancy=occupancies,
        removed_voxels=np.asarray(removed_counts),
        hit=hits,
        silhouette_iou_vs_ba=silhouette_iou,
        depth_polarity=np.asarray("D and t_BA normalized maps: larger=farther"),
    )
    summary = {
        "views": view_ids.tolist(),
        "lambda_voxels": list(lambda_all),
        "tau": args.tau,
        "erode_pixels": args.erode_pixels,
        "min_votes": args.min_votes,
        "hull_res": args.hull_res,
        "bound": args.bound,
        "removed_voxels": removed_counts,
        "occupied_voxels": [int(occ.sum()) for occ in occupancies],
        "silhouette_iou_vs_ba": silhouette_iou.tolist(),
        "neural_fit": "not run by this visual-only ablation",
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[save] arrays: {arrays_path}", flush=True)
    print(f"[save] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
