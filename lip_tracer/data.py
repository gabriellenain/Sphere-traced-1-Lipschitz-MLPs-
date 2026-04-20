"""Data loading and ray-building utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .config import SCENE, BLENDER_SCENE


# ---------- SFM / DTU ----------

def load_colmap_points(scene: Path = SCENE) -> Tensor:
    return torch.from_numpy(np.loadtxt(scene / "sparse_sfm_points.txt", dtype=np.float32))


def load_camera_centers(scene: Path = SCENE) -> Tensor:
    meta = json.loads((scene / "meta_data.json").read_text())
    c = np.stack([np.asarray(fr["camtoworld"], dtype=np.float32)[:3, 3]
                  for fr in meta["frames"]])
    return torch.from_numpy(c)


def load_sfm_pairs(scene: Path = SCENE) -> tuple[Tensor, Tensor]:
    """(origins, points) of shape (P, 3) — one row per (camera, sfm_point) pair."""
    meta = json.loads((scene / "meta_data.json").read_text())
    origins_list, points_list = [], []
    for fr in meta["frames"]:
        pts = np.loadtxt(scene / fr["sfm_sparse_points_view"], dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None]
        o = np.asarray(fr["camtoworld"], dtype=np.float32)[:3, 3]
        origins_list.append(np.broadcast_to(o, pts.shape).copy())
        points_list.append(pts)
    return (torch.from_numpy(np.concatenate(origins_list)),
            torch.from_numpy(np.concatenate(points_list)))


def load_views(scene: Path = SCENE) -> dict:
    meta = json.loads((scene / "meta_data.json").read_text())
    import imageio.v2 as imageio
    imgs, c2ws, Ks, masks = [], [], [], []
    for fr in meta["frames"]:
        img = imageio.imread(scene / fr["rgb_path"])[..., :3].astype(np.float32) / 255.0
        msk = imageio.imread(scene / fr["foreground_mask"])[..., 0] > 127
        img[~msk] = 0.0   # zero background — photo loss sees constant 0 there
        imgs.append(img)
        c2ws.append(np.asarray(fr["camtoworld"], dtype=np.float32))
        Ks.append(np.asarray(fr["intrinsics"], dtype=np.float32)[:3, :3])
        masks.append(msk)
    return {
        "images": torch.from_numpy(np.stack(imgs)),   # (V, H, W, 3)
        "masks":  torch.from_numpy(np.stack(masks)),  # (V, H, W) bool
        "c2w":    torch.from_numpy(np.stack(c2ws)),   # (V, 4, 4)
        "K":      torch.from_numpy(np.stack(Ks)),     # (V, 3, 3)
        "H": imgs[0].shape[0], "W": imgs[0].shape[1],
    }


# ---------- Blender ----------

def load_blender_views(scene: Path = BLENDER_SCENE, split: str = "train",
                       down: int = 2) -> dict:
    """Load NeRF Blender synthetic dataset (RGBA PNGs, NeRF camera convention).

    Converts from NeRF (x-right, y-up, z-back) to OpenCV (+z-forward) convention.
    Alpha channel is used as foreground mask; background is zeroed out.
    """
    import imageio.v2 as imageio
    meta = json.loads((scene / f"transforms_{split}.json").read_text())
    fov_x = meta["camera_angle_x"]
    imgs, c2ws, masks = [], [], []
    for fr in meta["frames"]:
        rgba = imageio.imread(scene / (fr["file_path"] + ".png")).astype(np.float32) / 255.0
        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3] * alpha
        msk = alpha[..., 0] > 0.5
        if down > 1:
            from PIL import Image
            H0, W0 = rgb.shape[:2]
            H1, W1 = H0 // down, W0 // down
            rgb = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
                (W1, H1), Image.BILINEAR)).astype(np.float32) / 255.0
            msk = np.array(Image.fromarray(msk).resize((W1, H1), Image.NEAREST))
        imgs.append(rgb)
        masks.append(msk)
        c2w_nerf = np.asarray(fr["transform_matrix"], dtype=np.float32)
        c2ws.append(c2w_nerf @ np.diag([1, -1, -1, 1]).astype(np.float32))

    H, W = imgs[0].shape[:2]
    fx = 0.5 * W / np.tan(0.5 * fov_x)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], np.float32)
    print(f"  blender: {len(imgs)} views {H}×{W}  fg={np.stack(masks).mean():.1%}")
    return {
        "images": torch.from_numpy(np.stack(imgs)),
        "masks":  torch.from_numpy(np.stack(masks)),
        "c2w":    torch.from_numpy(np.stack(c2ws)),
        "K":      torch.from_numpy(np.stack([K] * len(imgs))),
        "H": H, "W": W,
    }


# ---------- ray utilities ----------

def make_deterministic_rays(views: dict, down: int, device: str) -> dict:
    """Pre-build a regular pixel grid for every camera.

    Returns dict with keys: o, d, vi, gt, fg, rays_per_view.
    """
    V, H, W = views["c2w"].shape[0], views["H"], views["W"]
    H_d, W_d = H // down, W // down
    all_o, all_d, all_vi, all_gt, all_fg = [], [], [], [], []
    images  = views["images"].to(device)
    fg_maps = views["masks"].to(device) if "masks" in views else None
    for v in range(V):
        K   = views["K"][v].numpy()
        c2w = views["c2w"][v].numpy()
        ys, xs = np.meshgrid(np.arange(H_d), np.arange(W_d), indexing="ij")
        xs_f = (xs + 0.5) * down - 0.5
        ys_f = (ys + 0.5) * down - 0.5
        d_cam = np.stack([(xs_f - K[0, 2]) / K[0, 0],
                          (ys_f - K[1, 2]) / K[1, 1],
                          np.ones_like(xs_f)], axis=-1)
        dirs = d_cam @ c2w[:3, :3].T
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        o_np = np.broadcast_to(c2w[:3, 3], dirs.shape)
        N    = H_d * W_d
        all_o.append(torch.from_numpy(o_np.reshape(-1, 3).copy()).float().to(device))
        all_d.append(torch.from_numpy(dirs.reshape(-1, 3)).float().to(device))
        all_vi.append(torch.full((N,), v, dtype=torch.long, device=device))
        yi = torch.from_numpy((ys_f + 0.5).astype(np.int64).ravel()).clamp(0, H - 1).to(device)
        xi = torch.from_numpy((xs_f + 0.5).astype(np.int64).ravel()).clamp(0, W - 1).to(device)
        all_gt.append(images[v, yi, xi])
        if fg_maps is not None:
            all_fg.append(fg_maps[v, yi, xi])
        else:
            all_fg.append(torch.ones(N, dtype=torch.bool, device=device))
    return {
        "o":             torch.cat(all_o),
        "d":             torch.cat(all_d),
        "vi":            torch.cat(all_vi),
        "gt":            torch.cat(all_gt),
        "fg":            torch.cat(all_fg),
        "rays_per_view": H_d * W_d,
    }


def precompute_alt_cameras(views: dict, n_alt: int) -> Tensor:
    """(V, n_alt) int tensor of nearest-neighbour camera indices per view."""
    origins = views["c2w"][:, :3, 3]
    dists   = torch.cdist(origins, origins)
    dists.fill_diagonal_(float("inf"))
    _, nn_idx = dists.topk(n_alt, largest=False, dim=1)
    return nn_idx
