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


def load_aligned_depths(scene: Path = SCENE) -> dict | None:
    """Load mono depth maps and align each to metric z-depth via SFM points.

    Returns dict with:
        depths: list of (H, W) float32 tensors — aligned metric z-depth per view
        valid:  list of (H, W) bool tensors — pixels with valid aligned depth
        c2w, K, images: camera/image data

    Returns None for old-format DTU scans (cameras.npz layout) that lack
    meta_data.json and per-frame depth/normal npy files.
    """
    if not (scene / "meta_data.json").exists():
        print(f"  [geomvs] no meta_data.json in {scene.name} — skipping MVS/normal supervision")
        return None
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

    # Load GT camera-space normals (stored as (n+1)/2 in [0,1], shape (3,H,W))
    gt_normals = []
    for fr in frames:
        n_path = scene / fr.get("mono_normal_path", "")
        if n_path.exists():
            n_raw = np.load(str(n_path)).astype(np.float32)  # (3, H, W)
            if n_raw.shape[0] == 3:
                n_raw = n_raw.transpose(1, 2, 0)             # (H, W, 3)
            n_dec = 2.0 * n_raw - 1.0                        # decode from [0,1]
            gt_normals.append(torch.from_numpy(n_dec))
        else:
            H_n, W_n = aligned_depths[0].shape
            gt_normals.append(torch.zeros(H_n, W_n, 3))

    return {
        "depths":   aligned_depths,     # list of (H, W) float32 tensors
        "valid":    valid_masks,         # list of (H, W) bool tensors
        "normals":  gt_normals,          # list of (H, W, 3) camera-space normals
        "c2w": np.stack(c2ws),           # (V, 3/4, 4)
        "K": np.stack(Ks),              # (V, 3, 3)
        "H": aligned_depths[0].shape[0],
        "W": aligned_depths[0].shape[1],
    }


def load_mast3r_depths_idr(scene: Path, depth_dir: Path) -> dict | None:
    """Load MASt3R depth maps for an IDR-style DTU scan (cameras.npz layout).

    Aligns each z-depth map to the normalised training coordinate system via a
    per-view least-squares scale+shift fit on projected SFM points — the same
    strategy as load_aligned_depths for sdfstudio-format scenes.

    depth_dir must contain a manifest.json produced by precompute_mast3r_depths.py.
    Returns the same dict as load_aligned_depths, or None on error.
    """
    manifest_path = depth_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  [geomvs] no manifest.json in {depth_dir} — skipping MASt3R depth loading")
        return None

    manifest = json.loads(manifest_path.read_text())
    frames_man = manifest["frames"]

    sfm_path = scene / "sparse_sfm_points.txt"
    if not sfm_path.exists():
        print(f"  [geomvs] no sparse_sfm_points.txt in {scene} — cannot align MASt3R depths")
        return None
    sfm_pts = np.loadtxt(sfm_path, dtype=np.float64)
    if sfm_pts.ndim == 1:
        sfm_pts = sfm_pts[None]

    from scipy.linalg import rq
    cam_dict = np.load(scene / "cameras.npz")
    img_paths = sorted(p for p in (scene / "image").glob("*.png") if not p.name.startswith("._"))

    aligned_depths, valid_masks, c2ws, Ks = [], [], [], []

    for i, img_path in enumerate(img_paths):
        if i >= len(frames_man):
            break

        # ---- reconstruct normalised camera (same as _load_dtu_views) ----
        P = cam_dict[f"world_mat_{i}"][:3, :4].astype(np.float64)
        M = P[:, :3]
        K_cam, R_cam = rq(M)
        sign = np.sign(np.diag(K_cam)); sign[sign == 0] = 1.0
        T = np.diag(sign)
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
        K_f = K_cam.astype(np.float32)
        R_f = R_cam.astype(np.float32)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_f.T
        c2w[:3, 3] = cam_center_norm.astype(np.float32)
        c2ws.append(c2w); Ks.append(K_f)

        # ---- load MASt3R z-depth ----
        depth = np.load(depth_dir / frames_man[i]["depth_npy"]).astype(np.float32)
        H_d, W_d = depth.shape

        # ---- project SFM points into normalised camera space ----
        pts_cam = (R_f @ sfm_pts.T).T + t_norm
        z_norm = pts_cam[:, 2].astype(np.float32)
        u = K_f[0, 0] * pts_cam[:, 0] / (z_norm + 1e-6) + K_f[0, 2]
        v = K_f[1, 1] * pts_cam[:, 1] / (z_norm + 1e-6) + K_f[1, 2]
        ok = (z_norm > 1e-3) & (u >= 0) & (u < W_d) & (v >= 0) & (v < H_d)

        if ok.sum() < 3:
            print(f"  [geomvs] view {i}: only {ok.sum()} SFM points in frame — marking all invalid")
            aligned_depths.append(torch.zeros(H_d, W_d, dtype=torch.float32))
            valid_masks.append(torch.zeros(H_d, W_d, dtype=torch.bool))
            continue

        ui = np.clip(u[ok].astype(int), 0, W_d - 1)
        vi_arr = np.clip(v[ok].astype(int), 0, H_d - 1)
        mast3r_at_sfm = depth[vi_arr, ui].astype(np.float64)
        sfm_at_sfm = z_norm[ok].astype(np.float64)

        # least-squares: z_norm ≈ a * z_mast3r + b
        A = np.stack([mast3r_at_sfm, np.ones_like(mast3r_at_sfm)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, sfm_at_sfm, rcond=None)
        aligned = (a * depth + b).astype(np.float32)
        valid = aligned > 1e-3
        aligned_depths.append(torch.from_numpy(aligned))
        valid_masks.append(torch.from_numpy(valid))
        pct = valid.mean() * 100
        print(f"  [geomvs] view {i}: scale={a:.4f} shift={b:.4f}  "
              f"valid={pct:.1f}%  z=[{aligned[valid].min():.3f},{aligned[valid].max():.3f}]")

    if not aligned_depths:
        return None

    H_out, W_out = aligned_depths[0].shape
    gt_normals = [torch.zeros(H_out, W_out, 3) for _ in aligned_depths]
    print(f"  [geomvs] loaded {len(aligned_depths)} MASt3R depth maps from {depth_dir.name}")
    return {
        "depths":  aligned_depths,
        "valid":   valid_masks,
        "normals": gt_normals,
        "c2w": np.stack(c2ws),
        "K":   np.stack(Ks),
        "H": H_out, "W": W_out,
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
