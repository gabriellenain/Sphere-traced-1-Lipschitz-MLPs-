"""Compare Depth Anything V2 with border-aware visual-hull first-hit depth.

The default scene/view is DTU scan118 view 23: the frontal statue image where
the child is recessed behind the hands/shell. This is a read-only diagnostic:
it does not alter hull initialization, carving behavior, or training.

Depth Anything V2's relative checkpoint emits a disparity-like prediction:
larger raw values mean nearer to the camera. For all displayed/saved normalized
depth maps below, the polarity is inverted so larger values consistently mean
farther from the camera.

Example:
  source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
  python scripts/depth_anything_v2_hull_diag.py
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
from scipy.ndimage import distance_transform_edt, map_coordinates
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.data import load_views
from lip_tracer.visual_hull import carve


DEFAULT_SCENE = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan118")
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _normalize_farther(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    raw_larger_is_farther: bool,
) -> np.ndarray:
    """Normalize valid values to [0, 1], with larger output meaning farther."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    finite = valid & np.isfinite(values)
    if not finite.any():
        return out
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    scale = max(hi - lo, 1e-8)
    norm = (values[finite] - lo) / scale
    out[finite] = norm if raw_larger_is_farther else 1.0 - norm
    return out


def _run_depth_anything(
    rgb: np.ndarray,
    model_name: str,
    device: str,
) -> np.ndarray:
    """Return the model's raw relative inverse-depth prediction at RGB size."""
    print(f"[da-v2] loading {model_name} on {device}", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
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
    out = raw.float().cpu().numpy().astype(np.float32)
    print(
        f"[da-v2] raw relative inverse depth range: [{out.min():.5f}, {out.max():.5f}] "
        "(larger raw = nearer; displayed depth is inverted)",
        flush=True,
    )
    return out


def _make_rays(
    c2w: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    down: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    Hr, Wr = H // down, W // down
    K = K.copy()
    K[0] /= down
    K[1] /= down
    ys, xs = np.meshgrid(
        np.arange(Hr, dtype=np.float32),
        np.arange(Wr, dtype=np.float32),
        indexing="ij",
    )
    dcam = np.stack(
        [
            (xs - K[0, 2]) / K[0, 0],
            (ys - K[1, 2]) / K[1, 1],
            np.ones_like(xs),
        ],
        axis=-1,
    ).reshape(-1, 3)
    dirs = dcam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-8)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).copy()
    return origins, dirs, Hr, Wr


def _ray_box(
    origins: np.ndarray,
    dirs: np.ndarray,
    bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore"):
        inv = 1.0 / dirs
    t0 = (-bound - origins) * inv
    t1 = (bound - origins) * inv
    near = np.maximum(np.minimum(t0, t1).max(axis=1), 0.0)
    far = np.maximum(t0, t1).min(axis=1)
    return near, far


def _sample_first_hit(
    occ: np.ndarray,
    origins: np.ndarray,
    dirs: np.ndarray,
    bound: float,
    *,
    step_voxels: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-march the occupancy grid and return approximate first-hit distances."""
    res = occ.shape[0]
    voxel = 2.0 * bound / max(res - 1, 1)
    step = voxel * step_voxels
    near, far = _ray_box(origins, dirs, bound)
    hit = np.zeros(len(dirs), dtype=bool)
    depth = np.full(len(dirs), np.nan, dtype=np.float32)

    for start in range(0, len(dirs), chunk):
        stop = min(start + chunk, len(dirs))
        o, d = origins[start:stop], dirs[start:stop]
        tn, tf = near[start:stop], far[start:stop]
        crosses_box = tf >= tn
        if not crosses_box.any():
            continue
        n_steps = int(np.ceil(np.max((tf[crosses_box] - tn[crosses_box]) / step))) + 1
        ts = tn[:, None] + np.arange(n_steps, dtype=np.float32)[None, :] * step
        active = crosses_box[:, None] & (ts <= tf[:, None])
        xyz = o[:, None, :] + ts[..., None] * d[:, None, :]
        grid = np.rint((xyz + bound) / voxel).astype(np.int32)
        grid = np.clip(grid, 0, res - 1)
        inside = occ[grid[..., 2], grid[..., 1], grid[..., 0]] & active
        any_hit = inside.any(axis=1)
        first = inside.argmax(axis=1)
        rows = np.nonzero(any_hit)[0]
        hit[start + rows] = True
        depth[start + rows] = ts[rows, first[rows]]
        if start == 0 or stop == len(dirs) or start % (20 * chunk) == 0:
            print(f"[hull] first-hit rays {stop:,}/{len(dirs):,}", flush=True)
    return depth, hit


def _shade_hits(
    occ: np.ndarray,
    origins: np.ndarray,
    dirs: np.ndarray,
    depth: np.ndarray,
    hit: np.ndarray,
    bound: float,
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Phong-shade first-hit points using the occupancy distance-transform SDF."""
    voxel = 2.0 * bound / max(occ.shape[0] - 1, 1)
    if gradients is None:
        sdf = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * voxel
        grad_z, grad_y, grad_x = np.gradient(sdf, voxel, voxel, voxel)
    else:
        grad_z, grad_y, grad_x = gradients
    p = origins[hit] + depth[hit, None] * dirs[hit]
    coords = ((p + bound) / voxel)[:, [2, 1, 0]].T
    normals = np.stack(
        [
            map_coordinates(grad_x, coords, order=1, mode="nearest"),
            map_coordinates(grad_y, coords, order=1, mode="nearest"),
            map_coordinates(grad_z, coords, order=1, mode="nearest"),
        ],
        axis=1,
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-8)

    light = np.asarray([0.4, 0.5, 0.8], dtype=np.float32)
    light /= np.linalg.norm(light)
    view = -dirs[hit]
    half_vec = view + light[None]
    half_vec /= np.linalg.norm(half_vec, axis=1, keepdims=True).clip(min=1e-8)
    diffuse = np.clip(normals @ light, 0.0, 1.0)
    specular = np.clip(np.sum(normals * half_vec, axis=1), 0.0, 1.0) ** 32
    shade = np.clip(0.20 + 0.65 * diffuse + 0.15 * specular, 0.0, 1.0)

    phong = np.ones((len(dirs), 3), dtype=np.float32)
    phong[hit] = shade[:, None]
    return phong


def _render_border_aware_hull(
    scene: Path,
    view: int,
    *,
    hull_res: int,
    bound: float,
    render_down: int,
    ray_step_voxels: float,
    ray_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(
        f"[hull] carving current border-aware visual hull: "
        f"res={hull_res}, bound={bound:g}",
        flush=True,
    )
    occ = carve(scene=scene, res=hull_res, bound=bound, border_aware=True)
    print(f"[hull] occupied voxels: {int(occ.sum()):,}/{occ.size:,}", flush=True)
    views = load_views(scene)
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
    phong = _shade_hits(occ, origins, dirs, depth, hit, bound)
    return depth.reshape(Hr, Wr), hit.reshape(Hr, Wr), phong.reshape(Hr, Wr, 3)


def _save_figure(
    out: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    phong: np.ndarray,
    t_ba_norm: np.ndarray,
    da_norm: np.ndarray,
    view: int,
) -> None:
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].contour(mask, levels=[0.5], colors=["#00d8ff"], linewidths=0.6)
    axes[0].set_title(f"RGB + mask outline\nDTU scan118 view {view}")
    axes[1].imshow(phong)
    axes[1].set_title("Border-aware hull\nPhong first-hit render")
    axes[2].imshow(t_ba_norm, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[2].set_title(r"Border-aware hull $t_{\rm BA}$" "\nnormalized ray depth")
    im = axes[3].imshow(da_norm, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[3].set_title("Depth Anything V2\nnormalized relative depth")
    for ax in axes:
        ax.set_axis_off()
    cb = fig.colorbar(im, ax=axes[2:], shrink=0.72, pad=0.02)
    cb.set_label("normalized depth (larger = farther from camera)")
    fig.suptitle(
        "Depth Anything V2 polarity: raw relative output is inverse-depth-like "
        "(larger = nearer); displayed DA-V2 depth is inverted",
        fontsize=11,
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--view", type=int, default=23, help="frontal DTU view index")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/depth_anything_v2_scan118_view23"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="'auto', 'cpu', or e.g. 'cuda'")
    ap.add_argument("--hull-res", type=int, default=256, help="match current hull init")
    ap.add_argument("--bound", type=float, default=1.5, help="match current DTU hull init")
    ap.add_argument("--render-down", type=int, default=2, help="hull render downsampling factor")
    ap.add_argument("--ray-step-voxels", type=float, default=0.5)
    ap.add_argument("--ray-chunk", type=int, default=2048)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    image_paths = sorted(p for p in (args.scene / "image").glob("*.png") if not p.name.startswith("._"))
    mask_paths = sorted(p for p in (args.scene / "mask").glob("*.png") if not p.name.startswith("._"))
    if args.view < 0 or args.view >= len(image_paths):
        raise ValueError(f"--view {args.view} is outside [0, {len(image_paths) - 1}]")
    if len(mask_paths) < len(image_paths):
        raise FileNotFoundError(f"expected at least {len(image_paths)} masks under {args.scene / 'mask'}")
    rgb = imageio.imread(image_paths[args.view])[..., :3].astype(np.uint8)
    mask_raw = imageio.imread(mask_paths[args.view])
    mask = (mask_raw[..., 0] if mask_raw.ndim == 3 else mask_raw) > 127

    da_inverse_raw = _run_depth_anything(rgb, args.model, device)
    da_farther_norm = _normalize_farther(
        da_inverse_raw,
        mask,
        raw_larger_is_farther=False,
    )
    t_ba_raw, t_ba_hit, t_ba_phong = _render_border_aware_hull(
        args.scene,
        args.view,
        hull_res=args.hull_res,
        bound=args.bound,
        render_down=args.render_down,
        ray_step_voxels=args.ray_step_voxels,
        ray_chunk=args.ray_chunk,
    )
    t_ba_farther_norm = _normalize_farther(
        t_ba_raw,
        t_ba_hit,
        raw_larger_is_farther=True,
    )
    mask_render = np.asarray(
        Image.fromarray(mask).resize(
            (t_ba_raw.shape[1], t_ba_raw.shape[0]),
            Image.Resampling.NEAREST,
        )
    ).astype(bool)

    npz_path = args.out_dir / f"scan118_view{args.view:02d}_depth_arrays.npz"
    np.savez_compressed(
        npz_path,
        rgb=rgb,
        mask=mask,
        mask_render=mask_render,
        da_inverse_depth_raw=da_inverse_raw,
        da_depth_farther_norm=da_farther_norm,
        t_ba_raw=t_ba_raw,
        t_ba_depth_farther_norm=t_ba_farther_norm,
        t_ba_hit=t_ba_hit,
        t_ba_phong=t_ba_phong,
        view=np.asarray(args.view),
        hull_res=np.asarray(args.hull_res),
        hull_bound=np.asarray(args.bound),
        hull_render_down=np.asarray(args.render_down),
        model=np.asarray(args.model),
        depth_polarity=np.asarray(
            "da_inverse_depth_raw: larger=nearer; "
            "da_depth_farther_norm and t_ba_depth_farther_norm: larger=farther"
        ),
    )
    fig_path = args.out_dir / f"scan118_view{args.view:02d}_depth_comparison.png"
    _save_figure(
        fig_path,
        rgb,
        mask,
        t_ba_phong,
        t_ba_farther_norm,
        da_farther_norm,
        args.view,
    )
    print(f"[save] arrays: {npz_path}", flush=True)
    print(f"[save] figure: {fig_path}", flush=True)
    print(
        "[polarity] da_inverse_depth_raw larger=nearer; normalized displays larger=farther",
        flush=True,
    )


if __name__ == "__main__":
    main()
