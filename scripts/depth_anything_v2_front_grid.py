"""Render a read-only DA-V2 versus border-aware-hull grid around scan118 front.

The default neighborhood is scan118 views 20..26 (front view 23, radius 3).
Each row shows RGB, border-aware hull Phong, hull first-hit depth t_BA, and
Depth Anything V2 relative depth. All displayed depth maps use the same
polarity: larger values mean farther from the camera.

This diagnostic carves the current border-aware hull once for visualization.
It does not modify carving, initialization, or training.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views
from lip_tracer.visual_hull import carve
from scripts.depth_anything_v2_hull_diag import (
    DEFAULT_MODEL,
    DEFAULT_SCENE,
    _make_rays,
    _normalize_farther,
    _sample_first_hit,
    _shade_hits,
)


def _load_rgb_and_mask(
    image_paths: list[Path],
    mask_paths: list[Path],
    view: int,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = imageio.imread(image_paths[view])[..., :3].astype(np.uint8)
    mask_raw = imageio.imread(mask_paths[view])
    mask = (mask_raw[..., 0] if mask_raw.ndim == 3 else mask_raw) > 127
    return rgb, mask


def _predict_da(
    rgb: np.ndarray,
    processor,
    model,
    device: str,
) -> np.ndarray:
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        raw = model(**inputs).predicted_depth
        raw = F.interpolate(
            raw.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )[0, 0]
    return raw.float().cpu().numpy().astype(np.float32)


def _save_grid(
    out: Path,
    view_ids: list[int],
    rgbs: np.ndarray,
    phongs: np.ndarray,
    t_ba_norms: np.ndarray,
    da_norms: np.ndarray,
) -> None:
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(
        len(view_ids),
        4,
        figsize=(15, 4.0 * len(view_ids)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, view in enumerate(view_ids):
        axes[row, 0].imshow(rgbs[row])
        axes[row, 1].imshow(phongs[row])
        axes[row, 2].imshow(t_ba_norms[row], cmap=cmap, vmin=0.0, vmax=1.0)
        im = axes[row, 3].imshow(da_norms[row], cmap=cmap, vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(f"view {view}", fontsize=11)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    titles = [
        "RGB",
        "Border-aware hull\nPhong first hit",
        r"Border-aware hull $t_{\rm BA}$",
        "Depth Anything V2\nrelative depth",
    ]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=12)
    cb = fig.colorbar(im, ax=axes[:, 2:], shrink=0.6, pad=0.015)
    cb.set_label("normalized depth (larger = farther from camera)")
    fig.suptitle(
        "DTU scan118 front neighborhood: DA-V2 raw output is inverse-depth-like; "
        "displayed DA-V2 depth is inverted",
        fontsize=12,
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--front-view", type=int, default=23)
    ap.add_argument("--radius", type=int, default=3, help="render front-view +/- radius")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/depth_anything_v2_scan118_front_grid"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="'auto', 'cpu', or e.g. 'cuda'")
    ap.add_argument("--hull-res", type=int, default=256, help="match current hull init")
    ap.add_argument("--bound", type=float, default=1.5, help="match current DTU hull init")
    ap.add_argument("--render-down", type=int, default=2)
    ap.add_argument("--ray-step-voxels", type=float, default=0.5)
    ap.add_argument("--ray-chunk", type=int, default=2048)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    image_paths = sorted(p for p in (args.scene / "image").glob("*.png") if not p.name.startswith("._"))
    mask_paths = sorted(p for p in (args.scene / "mask").glob("*.png") if not p.name.startswith("._"))
    if len(mask_paths) < len(image_paths):
        raise FileNotFoundError(f"expected at least {len(image_paths)} masks under {args.scene / 'mask'}")
    lo = max(0, args.front_view - args.radius)
    hi = min(len(image_paths) - 1, args.front_view + args.radius)
    view_ids = list(range(lo, hi + 1))
    print(f"[grid] views: {view_ids}", flush=True)

    print(f"[da-v2] loading {args.model} on {device}", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        args.model,
        local_files_only=True,
    ).to(device).eval()

    print(
        f"[hull] carving current border-aware visual hull once: "
        f"res={args.hull_res}, bound={args.bound:g}",
        flush=True,
    )
    occ = carve(scene=args.scene, res=args.hull_res, bound=args.bound, border_aware=True)
    print(f"[hull] occupied voxels: {int(occ.sum()):,}/{occ.size:,}", flush=True)
    voxel = 2.0 * args.bound / max(args.hull_res - 1, 1)
    sdf = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * voxel
    gradients = np.gradient(sdf, voxel, voxel, voxel)
    views = load_views(args.scene)

    rgbs, masks, phongs = [], [], []
    da_raws, da_norms, t_ba_raws, t_ba_norms, t_ba_hits = [], [], [], [], []
    for view in view_ids:
        print(f"[grid] view {view}", flush=True)
        rgb, mask = _load_rgb_and_mask(image_paths, mask_paths, view)
        da_raw = _predict_da(rgb, processor, model, device)
        da_norm = _normalize_farther(da_raw, mask, raw_larger_is_farther=False)
        origins, dirs, Hr, Wr = _make_rays(
            views["c2w"][view].numpy(),
            views["K"][view].numpy(),
            views["H"],
            views["W"],
            args.render_down,
        )
        t_ba_raw, t_ba_hit = _sample_first_hit(
            occ,
            origins,
            dirs,
            args.bound,
            step_voxels=args.ray_step_voxels,
            chunk=args.ray_chunk,
        )
        phong = _shade_hits(
            occ,
            origins,
            dirs,
            t_ba_raw,
            t_ba_hit,
            args.bound,
            gradients=gradients,
        )
        t_ba_raw = t_ba_raw.reshape(Hr, Wr)
        t_ba_hit = t_ba_hit.reshape(Hr, Wr)
        rgbs.append(rgb)
        masks.append(mask)
        da_raws.append(da_raw)
        da_norms.append(da_norm)
        t_ba_raws.append(t_ba_raw)
        t_ba_hits.append(t_ba_hit)
        t_ba_norms.append(_normalize_farther(t_ba_raw, t_ba_hit, raw_larger_is_farther=True))
        phongs.append(phong.reshape(Hr, Wr, 3))

    arrays_path = args.out_dir / f"scan118_views{view_ids[0]:02d}-{view_ids[-1]:02d}_depth_arrays.npz"
    np.savez_compressed(
        arrays_path,
        views=np.asarray(view_ids),
        rgb=np.stack(rgbs),
        mask=np.stack(masks),
        da_inverse_depth_raw=np.stack(da_raws),
        da_depth_farther_norm=np.stack(da_norms),
        t_ba_raw=np.stack(t_ba_raws),
        t_ba_depth_farther_norm=np.stack(t_ba_norms),
        t_ba_hit=np.stack(t_ba_hits),
        t_ba_phong=np.stack(phongs),
        model=np.asarray(args.model),
        depth_polarity=np.asarray(
            "da_inverse_depth_raw: larger=nearer; "
            "da_depth_farther_norm and t_ba_depth_farther_norm: larger=farther"
        ),
    )
    grid_path = args.out_dir / f"scan118_views{view_ids[0]:02d}-{view_ids[-1]:02d}_grid.png"
    _save_grid(
        grid_path,
        view_ids,
        np.stack(rgbs),
        np.stack(phongs),
        np.stack(t_ba_norms),
        np.stack(da_norms),
    )
    print(f"[save] arrays: {arrays_path}", flush=True)
    print(f"[save] grid: {grid_path}", flush=True)
    print("[polarity] DA-V2 raw larger=nearer; normalized displays larger=farther", flush=True)


if __name__ == "__main__":
    main()
