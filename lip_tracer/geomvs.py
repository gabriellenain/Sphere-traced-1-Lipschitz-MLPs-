"""Dense depth priors from monocular depth maps aligned to SFM scale.

Loads the mono_depth_path .npy files from the DTU/SDFStudio dataset,
aligns each to metric z-depth using sparse SFM points (least-squares
scale+shift), and provides the aligned maps for supervision in method.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SCENE = Path(
    "/Data/gabriel.lenain/nerfstudio/sdfstudio/data/sdfstudio/sdfstudio-demo-data/dtu-scan65"
)
VIS_OUT = Path("/Data/gabriel.lenain/nerfstudio/outputs/geomvs_dtu65.png")


def load_aligned_depths(scene: Path = SCENE) -> dict:
    """Load mono depth maps and align each to metric z-depth via SFM points.

    Returns dict with:
        depths: list of (H, W) float32 tensors — aligned metric z-depth per view
        valid:  list of (H, W) bool tensors — pixels with valid aligned depth
        c2w, K, images: camera/image data
    """
    meta = json.loads((scene / "meta_data.json").read_text())
    frames = meta["frames"]

    aligned_depths = []
    valid_masks = []
    c2ws, Ks = [], []

    for vi, fr in enumerate(frames):
        c2w = np.asarray(fr["camtoworld"], dtype=np.float32)
        K = np.asarray(fr["intrinsics"], dtype=np.float32)[:3, :3]
        w2c = np.linalg.inv(np.eye(4, dtype=np.float32))
        w2c[:4, :4] = np.linalg.inv(
            np.vstack([c2w, [0, 0, 0, 1]]) if c2w.shape[0] == 3
            else c2w if c2w.shape == (4, 4)
            else np.vstack([c2w[:3], [0, 0, 0, 1]])
        )
        c2ws.append(c2w)
        Ks.append(K)

        mono = np.load(scene / fr["mono_depth_path"]).astype(np.float32)
        H, W = mono.shape

        # load SFM points visible in this view
        sfm_file = scene / fr["sfm_sparse_points_view"]
        pts = np.loadtxt(sfm_file, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None]

        # project SFM points to get their z-depth in this camera
        R = w2c[:3, :3]
        t_vec = w2c[:3, 3]
        pts_cam = (R @ pts.T).T + t_vec
        z_true = pts_cam[:, 2]
        u = K[0, 0] * pts_cam[:, 0] / z_true + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z_true + K[1, 2]

        # filter to in-frame points with positive z
        ok = (z_true > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if ok.sum() < 3:
            # not enough SFM points — mark all invalid
            aligned_depths.append(torch.zeros(H, W, dtype=torch.float32))
            valid_masks.append(torch.zeros(H, W, dtype=torch.bool))
            continue

        ui = np.clip(u[ok].astype(int), 0, W - 1)
        vi_arr = np.clip(v[ok].astype(int), 0, H - 1)
        z_gt = z_true[ok]
        mono_at_sfm = mono[vi_arr, ui]

        # least-squares: z_gt ≈ scale * mono + shift
        A = np.stack([mono_at_sfm, np.ones_like(mono_at_sfm)], axis=1)
        result = np.linalg.lstsq(A, z_gt, rcond=None)
        scale, shift = result[0]

        aligned = scale * mono + shift
        valid = aligned > 1e-3  # only positive depths

        aligned_depths.append(torch.from_numpy(aligned))
        valid_masks.append(torch.from_numpy(valid))

    return {
        "depths": aligned_depths,       # list of (H, W) tensors
        "valid": valid_masks,            # list of (H, W) bool tensors
        "c2w": np.stack(c2ws),           # (V, 3/4, 4)
        "K": np.stack(Ks),              # (V, 3, 3)
        "H": aligned_depths[0].shape[0],
        "W": aligned_depths[0].shape[1],
    }


def backproject_depth(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray,
                      valid: np.ndarray | None = None) -> np.ndarray:
    """Back-project a z-depth map to world-space 3D points. Returns (N, 3)."""
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dirs_cam = np.stack([
        (xs - K[0, 2]) / K[0, 0],
        (ys - K[1, 2]) / K[1, 1],
        np.ones_like(xs),
    ], axis=-1)
    pts_cam = dirs_cam * depth[..., None]
    c2w4 = np.eye(4, dtype=np.float32)
    c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
    pts_world = (pts_cam @ c2w4[:3, :3].T) + c2w4[:3, 3]
    if valid is not None:
        pts_world = pts_world[valid]
    return pts_world.reshape(-1, 3)


def visualize(scene: Path = SCENE) -> None:
    """Visualize aligned depth maps: depth images, back-projected point cloud, residuals."""
    import matplotlib.pyplot as plt

    data = load_aligned_depths(scene)
    sfm_pts = np.loadtxt(scene / "sparse_sfm_points.txt", dtype=np.float32)
    meta = json.loads((scene / "meta_data.json").read_text())

    V = len(data["depths"])
    show_views = list(range(0, V, max(1, V // 6)))[:6]

    fig, axes = plt.subplots(3, len(show_views), figsize=(4 * len(show_views), 10))

    # row 0: aligned depth maps
    # row 1: mono depth (raw) for comparison
    # row 2: back-projected point cloud (top-down)
    all_pts = []
    for col, vi in enumerate(show_views):
        depth = data["depths"][vi].numpy()
        valid = data["valid"][vi].numpy()
        mono = np.load(scene / meta["frames"][vi]["mono_depth_path"]).astype(np.float32)

        ax0 = axes[0, col]
        im0 = ax0.imshow(np.where(valid, depth, np.nan), cmap="plasma")
        ax0.set_title(f"view {vi} aligned z-depth")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        ax1 = axes[1, col]
        ax1.imshow(mono, cmap="plasma")
        ax1.set_title(f"view {vi} mono (raw)")
        ax1.axis("off")

        # back-project
        pts = backproject_depth(depth, data["K"][vi], data["c2w"][vi], valid)
        all_pts.append(pts[::16])  # subsample for viz

    # row 2: top-down scatter of back-projected points + SFM
    all_pts = np.concatenate(all_pts)
    for col in range(len(show_views)):
        ax2 = axes[2, col]
        if col == 0:
            ax2.scatter(all_pts[:, 0], all_pts[:, 2], s=0.1, c="cyan", alpha=0.3, label="depth prior")
            ax2.scatter(sfm_pts[:, 0], sfm_pts[:, 2], s=3, c="red", alpha=0.8, label="SFM")
            # cameras
            for i in range(V):
                o = data["c2w"][i][:3, 3]
                ax2.plot(o[0], o[2], "b^", markersize=4)
            ax2.set_aspect("equal")
            ax2.legend(fontsize=7)
            ax2.set_title("top-down (x-z)")
        elif col == 1:
            ax2.scatter(all_pts[:, 0], all_pts[:, 1], s=0.1, c="cyan", alpha=0.3)
            ax2.scatter(sfm_pts[:, 0], sfm_pts[:, 1], s=3, c="red", alpha=0.8)
            for i in range(V):
                o = data["c2w"][i][:3, 3]
                ax2.plot(o[0], o[1], "b^", markersize=4)
            ax2.set_aspect("equal")
            ax2.set_title("front (x-y)")
        else:
            ax2.axis("off")

    fig.suptitle("Mono depth aligned to SFM scale", fontsize=14)
    fig.tight_layout()
    VIS_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIS_OUT, dpi=150)
    print(f"saved → {VIS_OUT}")

    # print alignment stats
    print(f"\n{'view':>4}  {'n_sfm':>5}  {'depth_min':>9}  {'depth_max':>9}  {'depth_mean':>10}  {'valid%':>6}")
    for vi in range(V):
        d = data["depths"][vi].numpy()
        v = data["valid"][vi].numpy()
        fr = meta["frames"][vi]
        pts = np.loadtxt(scene / fr["sfm_sparse_points_view"], dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None]
        pct = 100 * v.sum() / v.size
        print(f"{vi:4d}  {len(pts):5d}  {d[v].min():9.4f}  {d[v].max():9.4f}  {d[v].mean():10.4f}  {pct:5.1f}%")


if __name__ == "__main__":
    visualize()
