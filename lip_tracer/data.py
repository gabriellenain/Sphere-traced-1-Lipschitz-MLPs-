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


def colmap_visibility_counts(pts: Tensor, views: dict, chunk: int = 4096) -> Tensor:
    """Count how many masked cameras see each COLMAP point."""
    H, W   = views["H"], views["W"]
    masks  = views["masks"]
    w2c    = torch.linalg.inv(views["c2w"])
    R, t   = w2c[:, :3, :3], w2c[:, :3, 3]
    K      = views["K"]
    counts = torch.zeros(len(pts), dtype=torch.long)
    for i in range(0, len(pts), chunk):
        p   = pts[i:i + chunk]
        xc  = torch.einsum("vij,nj->vni", R, p) + t[:, None, :]
        uvh = torch.einsum("vij,vnj->vni", K, xc)
        uv  = uvh[..., :2] / uvh[..., 2:3].clamp_min(1e-6)
        u_, v_ = uv[..., 0], uv[..., 1]
        in_b = (u_ >= 0) & (u_ < W) & (v_ >= 0) & (v_ < H) & (xc[..., 2] > 1e-4)
        ui   = u_.round().long().clamp(0, W - 1)
        vi   = v_.round().long().clamp(0, H - 1)
        vi_  = torch.arange(masks.shape[0])[:, None].expand_as(ui)
        counts[i:i + chunk] = (in_b & masks[vi_, vi, ui]).sum(dim=0)
    return counts


def colmap_visibility_matrix(pts: Tensor, views: dict, chunk: int = 4096) -> Tensor:
    """Boolean (V, P): does each masked camera see each COLMAP point.

    Per-view version of colmap_visibility_counts (counts == matrix.sum(0)).
    This is Geo-Neus's view_id.npy: row v lists the points visible in view v,
    used for per-view SDF supervision. Built by projection with the same camera
    arrays the training loop uses, so view rows align with K/w2c indexing.
    """
    H, W   = views["H"], views["W"]
    masks  = views["masks"]
    w2c    = torch.linalg.inv(views["c2w"])
    R, t   = w2c[:, :3, :3], w2c[:, :3, 3]
    K      = views["K"]
    V      = masks.shape[0]
    vis    = torch.zeros(V, len(pts), dtype=torch.bool)
    for i in range(0, len(pts), chunk):
        p   = pts[i:i + chunk]
        xc  = torch.einsum("vij,nj->vni", R, p) + t[:, None, :]
        uvh = torch.einsum("vij,vnj->vni", K, xc)
        uv  = uvh[..., :2] / uvh[..., 2:3].clamp_min(1e-6)
        u_, v_ = uv[..., 0], uv[..., 1]
        in_b = (u_ >= 0) & (u_ < W) & (v_ >= 0) & (v_ < H) & (xc[..., 2] > 1e-4)
        ui   = u_.round().long().clamp(0, W - 1)
        vi   = v_.round().long().clamp(0, H - 1)
        vrow = torch.arange(V)[:, None].expand_as(ui)
        vis[:, i:i + chunk] = in_b & masks[vrow, vi, ui]
    return vis


def load_camera_centers(scene: Path = SCENE) -> Tensor:
    meta = json.loads((scene / "meta_data.json").read_text())
    c = np.stack([np.asarray(fr["camtoworld"], dtype=np.float32)[:3, 3]
                  for fr in meta["frames"]])
    return torch.from_numpy(c)


def _point_keyset(pts: np.ndarray, atol: float) -> set[tuple[int, int, int]]:
    quant = np.round(pts / atol).astype(np.int64)
    return {tuple(row) for row in quant}


def _filter_points_by_keyset(pts: np.ndarray, keys: set[tuple[int, int, int]],
                             atol: float) -> np.ndarray:
    if not len(pts):
        return pts
    quant = np.round(pts / atol).astype(np.int64)
    keep = np.fromiter((tuple(row) in keys for row in quant),
                       dtype=bool, count=len(pts))
    return pts[keep]


def load_sfm_pairs(scene: Path = SCENE, allowed_points: Tensor | np.ndarray | None = None,
                   atol: float = 1e-5) -> tuple[Tensor, Tensor]:
    """(origins, points) of shape (P, 3) — one row per (camera, sfm_point) pair.

    If allowed_points is provided, keep only pairs whose target belongs to that
    cleaned COLMAP set. Matching is quantized to tolerate text round-tripping.
    """
    # prefer dedicated sfm_pairs.json; fall back to meta_data.json
    meta_path = scene / "sfm_pairs.json"
    if not meta_path.exists():
        meta_path = scene / "meta_data.json"
    meta = json.loads(meta_path.read_text())
    allowed_keys = None
    if allowed_points is not None:
        allowed_np = (allowed_points.detach().cpu().numpy()
                      if isinstance(allowed_points, torch.Tensor)
                      else np.asarray(allowed_points))
        allowed_keys = _point_keyset(allowed_np.astype(np.float32), atol)
    origins_list, points_list = [], []
    for fr in meta["frames"]:
        pts = np.loadtxt(scene / fr["sfm_sparse_points_view"], dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None]
        if allowed_keys is not None:
            pts = _filter_points_by_keyset(pts, allowed_keys, atol)
        if len(pts) == 0:
            continue
        o = np.asarray(fr["camtoworld"], dtype=np.float32)[:3, 3]
        origins_list.append(np.broadcast_to(o, pts.shape).copy())
        points_list.append(pts)
    if not points_list:
        z = torch.empty(0, 3, dtype=torch.float32)
        return z, z.clone()
    return (torch.from_numpy(np.concatenate(origins_list)),
            torch.from_numpy(np.concatenate(points_list)))


def _load_dtu_views(scene: Path, down: int = 1) -> dict:
    """Load DTU / NeuS-BlendedMVS data from cameras*.npz + image/ + mask/ layout.

    DTU ships ``cameras.npz``; the NeuS-preprocessed BlendedMVS scenes ship the
    same IDR-format archive under ``cameras_sphere.npz`` (world_mat_i / scale_mat_i,
    object normalised to the unit sphere). Both decode identically here.
    """
    import imageio.v2 as imageio
    from PIL import Image as _PIL
    from scipy.linalg import rq

    cam_path = scene / "cameras.npz"
    if not cam_path.exists():
        cam_path = scene / "cameras_sphere.npz"
    cam_dict = np.load(cam_path)
    img_paths = sorted(p for p in (scene / "image").glob("*.png") if not p.name.startswith("._"))
    # Prefer eval_mask/ when present (MVMannequin ships clean eval masks next to
    # the noisy training mask/ — ~50x fewer stray foreground specks). DTU has no
    # eval_mask dir, so it transparently falls back to mask/.
    mask_dir = scene / "eval_mask" if (scene / "eval_mask").is_dir() else scene / "mask"
    mask_paths = sorted(p for p in mask_dir.glob("*.png") if not p.name.startswith("._"))
    if not img_paths:
        raise FileNotFoundError(f"no images found under {scene / 'image'}")
    if len(mask_paths) < len(img_paths):
        raise FileNotFoundError(f"expected at least {len(img_paths)} masks under {mask_dir}")
    print(f"  [data] masks from {mask_dir.name}/  ({len(mask_paths)} files)")

    imgs, c2ws, Ks, masks = [], [], [], []
    for i, img_path in enumerate(img_paths):
        img = imageio.imread(img_path)[..., :3].astype(np.float32) / 255.0
        msk = imageio.imread(mask_paths[i])
        if msk.ndim == 3:
            msk = msk[..., 0]
        msk = msk > 127
        img[~msk] = 0.0
        if down > 1:
            H0, W0 = img.shape[:2]
            H1, W1 = H0 // down, W0 // down
            img = np.array(_PIL.fromarray((img * 255).astype(np.uint8)).resize(
                (W1, H1), _PIL.BILINEAR)).astype(np.float32) / 255.0
            msk = np.array(_PIL.fromarray(msk).resize((W1, H1), _PIL.NEAREST))

        P = cam_dict[f"world_mat_{i}"][:3, :4].astype(np.float64)
        M = P[:, :3]
        K, R = rq(M)
        sign = np.sign(np.diag(K))
        sign[sign == 0] = 1.0
        T = np.diag(sign)
        K = K @ T
        R = T @ R
        if np.linalg.det(R) < 0:
            K[:, 2] *= -1.0
            R[2, :] *= -1.0
        K = (K / K[2, 2]).astype(np.float32)
        K_pose = K.astype(np.float64)
        t = np.linalg.solve(K_pose, P[:, 3])
        cam_center = -R.T @ t
        cam_center_h = np.concatenate([cam_center, [1.0]], axis=0)
        key_inv = f"scale_mat_inv_{i}"
        scale_mat_inv = cam_dict[key_inv] if key_inv in cam_dict else np.linalg.inv(cam_dict[f"scale_mat_{i}"])
        cam_center = (scale_mat_inv @ cam_center_h)[:3].astype(np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T.astype(np.float32)
        c2w[:3, 3] = cam_center
        if down > 1:
            K = K.copy()
            K[0] /= down
            K[1] /= down

        imgs.append(img)
        masks.append(msk)
        c2ws.append(c2w)
        Ks.append(K)

    return {
        "images": torch.from_numpy(np.stack(imgs)),
        "masks":  torch.from_numpy(np.stack(masks)),
        "c2w":    torch.from_numpy(np.stack(c2ws)),
        "K":      torch.from_numpy(np.stack(Ks)),
        "H": imgs[0].shape[0], "W": imgs[0].shape[1],
    }


def _load_tnt_views(scene: Path, down: int = 1) -> dict:
    """Load NSVF-preprocessed Tanks & Temples scene.

    Layout: intrinsics.txt (4x4), bbox.txt (xmin..zmax voxel_size),
    rgb/<split>_<frame>.png and pose/<split>_<frame>.txt (split 0 = train).
    Cameras + bbox are renormalised so the bbox fits the unit cube, so the
    rest of the DTU-tuned pipeline (bound=1.5, sphere radius~0.5) just works.
    Returns all-True masks so existing mask-keyed code paths degenerate to
    no-ops when mask losses are zero-weighted.
    """
    import imageio.v2 as imageio
    from PIL import Image as _PIL

    K_raw = np.loadtxt(scene / "intrinsics.txt", dtype=np.float32)[:3, :3]
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float32)
    bb_min, bb_max = bbox[:3], bbox[3:6]
    center = 0.5 * (bb_min + bb_max)
    scale  = float(np.max(0.5 * (bb_max - bb_min)))  # unit cube fit

    pose_paths = sorted(p for p in (scene / "pose").glob("0_*.txt"))
    if not pose_paths:
        raise FileNotFoundError(f"no training poses (0_*.txt) under {scene/'pose'}")

    imgs, c2ws, Ks, masks = [], [], [], []
    for pp in pose_paths:
        ip = scene / "rgb" / (pp.stem + ".png")
        if not ip.exists():
            for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                alt = scene / "rgb" / (pp.stem + suffix)
                if alt.exists():
                    ip = alt
                    break
        if not ip.exists():
            continue
        img = imageio.imread(ip).astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        c2w = np.loadtxt(pp, dtype=np.float32).reshape(4, 4)
        c2w[:3, 3] = (c2w[:3, 3] - center) / scale

        K = K_raw.copy()
        if down > 1:
            H0, W0 = img.shape[:2]
            H1, W1 = H0 // down, W0 // down
            img = np.array(_PIL.fromarray((img * 255).astype(np.uint8)).resize(
                (W1, H1), _PIL.BILINEAR)).astype(np.float32) / 255.0
            K[0] /= down
            K[1] /= down
        H, W = img.shape[:2]
        imgs.append(img)
        c2ws.append(c2w)
        Ks.append(K)
        mp = scene / "mask" / (pp.stem + ".png")
        if mp.exists():
            mraw = imageio.imread(mp)
            if mraw.ndim == 3:
                mraw = mraw[..., 0]
            if down > 1:
                mraw = np.array(_PIL.fromarray(mraw).resize((W, H), _PIL.NEAREST))
            masks.append(mraw > 127)
        else:
            masks.append(np.ones((H, W), dtype=bool))

    print(f"  tnt[{scene.name}]: {len(imgs)} views {imgs[0].shape[0]}x{imgs[0].shape[1]}  "
          f"bbox_scale={scale:.3f}  (down={down})")
    return {
        "images": torch.from_numpy(np.stack(imgs)),
        "masks":  torch.from_numpy(np.stack(masks)),
        "c2w":    torch.from_numpy(np.stack(c2ws)),
        "K":      torch.from_numpy(np.stack(Ks)),
        "H": imgs[0].shape[0], "W": imgs[0].shape[1],
    }


def load_view_keep(path) -> list[int]:
    """Read a newline/whitespace-separated list of view indices to keep."""
    txt = Path(path).read_text()
    return [int(t) for t in txt.split()]


def load_views(scene: Path = SCENE, down: int = 1,
               view_keep=None) -> dict:
    """Load all views, optionally subsetting to `view_keep` (list of indices
    into the full, file-order view list). The subset is applied uniformly to
    every per-view tensor so downstream indexing (alt-view NN, masks, depths)
    stays consistent."""
    out = _load_views_dispatch(scene, down=down)
    if view_keep is not None:
        idx = [int(i) for i in view_keep]
        n_full = out["c2w"].shape[0]
        if max(idx) >= n_full or min(idx) < 0:
            raise ValueError(f"view_keep index out of range for {n_full} views")
        for k in ("images", "masks", "c2w", "K"):
            if k in out and torch.is_tensor(out[k]):
                out[k] = out[k][idx]
        print(f"  view_keep: using {len(idx)}/{n_full} views")
    return out


def _load_views_dispatch(scene: Path = SCENE, down: int = 1) -> dict:
    if (scene / "intrinsics.txt").exists() and (scene / "pose").is_dir():
        return _load_tnt_views(scene, down=down)
    if not (scene / "meta_data.json").exists() or (scene / "image").exists():
        return _load_dtu_views(scene, down=down)
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


def load_blender_gt_points(scene: Path = BLENDER_SCENE,
                           n_pts: int = 30000) -> torch.Tensor:
    """Unproject test-split depth maps to a GT 3D point cloud.

    The depth PNGs are RGBA 8-bit; R channel encodes scene-space depth
    (Blender Z-pass normalised to the camera's far clip, exported at far=6.0).
    A channel > 0.5 marks valid (foreground) pixels.

    Returns (N, 3) float32 tensor in the same OpenCV world frame as training.
    """
    import imageio.v2 as imageio

    meta  = json.loads((scene / "transforms_test.json").read_text())
    fov_x = meta["camera_angle_x"]
    all_pts: list[np.ndarray] = []
    fx_set = False; fx = 1.0
    found_depth = 0

    for fr in meta["frames"]:
        depth_candidates = [
            scene / (fr["file_path"] + "_depth_0001.png"),
            scene / (fr["file_path"] + "_depth_0029.png"),
        ]
        depth_path = next((p for p in depth_candidates if p.exists()), None)
        if depth_path is None:
            continue
        found_depth += 1
        rgba = imageio.imread(str(depth_path)).astype(np.float32) / 255.0  # (H,W,4)
        H, W = rgba.shape[:2]
        if not fx_set:
            fx = 0.5 * W / np.tan(0.5 * fov_x)
            fx_set = True
        depth = rgba[..., 0]          # R channel: normalised depth in [0,1]
        valid = rgba[..., 3] > 0.5    # A channel: foreground mask
        if not valid.any():
            continue

        # Map [0,1] → scene-space depth; far=6.0 matches typical NeRF-synthetic clip
        depth_m = depth * 6.0

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx, cy = W / 2.0, H / 2.0
        pts_cam = np.stack([
            (xs - cx) / fx * depth_m,
            (ys - cy) / fx * depth_m,
            depth_m,
        ], axis=-1)  # (H, W, 3)

        c2w = np.asarray(fr["transform_matrix"], dtype=np.float32)
        c2w = c2w @ np.diag([1, -1, -1, 1]).astype(np.float32)  # NeRF→OpenCV
        pts_world = pts_cam[valid] @ c2w[:3, :3].T + c2w[:3, 3]
        all_pts.append(pts_world)

    if found_depth == 0:
        raise FileNotFoundError(
            f"No Blender test depth maps found in {scene}. "
            "Expected files like '*_depth_0001.png' or '*_depth_0029.png'."
        )
    if not all_pts:
        raise ValueError(
            f"Found Blender depth maps in {scene}, but none contained valid foreground pixels."
        )
    pts = np.concatenate(all_pts, axis=0)
    if len(pts) > n_pts:
        idx = np.random.default_rng(0).choice(len(pts), n_pts, replace=False)
        pts = pts[idx]

    t = torch.from_numpy(pts).float()
    print(f"  [gt_pts] {len(t)} points  "
          f"x∈[{t[:,0].min():.2f},{t[:,0].max():.2f}]  "
          f"y∈[{t[:,1].min():.2f},{t[:,1].max():.2f}]  "
          f"z∈[{t[:,2].min():.2f},{t[:,2].max():.2f}]  "
          f"r_mean={t.norm(dim=-1).mean():.3f}")
    return t


# ---------- ray utilities ----------

def make_deterministic_rays(views: dict, down: int, device: str) -> dict:
    """Pre-build a regular pixel grid for every camera.

    Returns dict with keys: o, d, vi, gt, fg, rays_per_view.
    """
    V, H, W = views["c2w"].shape[0], views["H"], views["W"]
    H_d, W_d = H // down, W // down
    all_o, all_d, all_vi, all_gt, all_fg = [], [], [], [], []
    all_px, all_py = [], []
    images  = views["images"]   # keep on CPU — batches moved to device at sample time
    fg_maps = views["masks"] if "masks" in views else None
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
        all_o.append(torch.from_numpy(o_np.reshape(-1, 3).copy()).float())
        all_d.append(torch.from_numpy(dirs.reshape(-1, 3)).float())
        all_vi.append(torch.full((N,), v, dtype=torch.long))
        all_px.append(torch.from_numpy(xs_f.ravel().astype(np.float32).copy()))
        all_py.append(torch.from_numpy(ys_f.ravel().astype(np.float32).copy()))
        yi = torch.from_numpy((ys_f + 0.5).astype(np.int64).ravel()).clamp(0, H - 1)
        xi = torch.from_numpy((xs_f + 0.5).astype(np.int64).ravel()).clamp(0, W - 1)
        all_gt.append(images[v, yi, xi])
        if fg_maps is not None:
            all_fg.append(fg_maps[v, yi, xi])
        else:
            all_fg.append(torch.ones(N, dtype=torch.bool))
    return {
        "o":             torch.cat(all_o),
        "d":             torch.cat(all_d),
        "vi":            torch.cat(all_vi),
        "gt":            torch.cat(all_gt),
        "fg":            torch.cat(all_fg),
        "px":            torch.cat(all_px),
        "py":            torch.cat(all_py),
        "rays_per_view": H_d * W_d,
    }


def precompute_alt_cameras(views: dict, n_alt: int) -> Tensor:
    """(V, n_alt) int tensor of nearest-neighbour camera indices per view."""
    origins = views["c2w"][:, :3, 3]
    dists   = torch.cdist(origins, origins)
    dists.fill_diagonal_(float("inf"))
    _, nn_idx = dists.topk(n_alt, largest=False, dim=1)
    return nn_idx


def precompute_alt_cameras_arccos(views: dict, n_alt: int) -> Tensor:
    """(V, n_alt) int tensor sorted by arccos angular distance between viewing
    directions.

    For each camera k, compute d_k = normalise(scene_centre − cam_centre_k)
    where scene_centre is the mean camera centre. Sort all other cameras by
    arccos(dot(d_ref, d_src)) — smallest angle first (most similar viewing
    direction). This is the source-view score used in papers such as
    PatchmatchNet / MVSNet.
    """
    origins = views["c2w"][:, :3, 3]           # (V, 3)
    scene_centre = origins.mean(dim=0)          # (3,)
    dirs = scene_centre.unsqueeze(0) - origins  # (V, 3) — cam → scene centre
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # cos similarity matrix → angular distance (smaller = more similar direction)
    cos_sim = dirs @ dirs.T                     # (V, V)
    cos_sim.clamp_(-1.0, 1.0)
    ang_dist = torch.acos(cos_sim)              # (V, V) in [0, π]
    ang_dist.fill_diagonal_(float("inf"))

    _, nn_idx = ang_dist.topk(n_alt, largest=False, dim=1)
    return nn_idx


def load_pair_file(path: Path) -> dict[int, list[int]]:
    """Parse an MVSNet / NeuralWarp ``pair.txt`` into {ref_id: ranked src ids}.

    Canonical format::

        <N>                                   # number of reference views
        <ref_id>
        <num_src> s0 score0 s1 score1 ...     # sources ranked best-first
        ...                                   # repeated N times

    Pair scores are ignored — only the left-to-right ranking of source ids is
    kept. Robust to the score-less variant (``<num_src> s0 s1 ...``).
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty pair file: {path}")
    n_ref = int(float(lines[0]))
    ranking: dict[int, list[int]] = {}
    i = 1
    for _ in range(n_ref):
        if i + 1 >= len(lines):
            break
        ref_id = int(float(lines[i])); i += 1
        toks = lines[i].split(); i += 1
        if not toks:
            ranking[ref_id] = []
            continue
        num = int(float(toks[0]))
        rest = toks[1:]
        if num > 0 and len(rest) == 2 * num:          # interleaved (id, score)
            ids = rest[0::2]
        elif num > 0 and len(rest) == num:            # ids only, no scores
            ids = rest
        else:                                         # best-effort: assume interleaved
            ids = rest[0::2]
        ranking[ref_id] = [int(float(t)) for t in ids]
    return ranking


def alt_cameras_from_pairs(path: Path, n_views: int, n_alt: int) -> Tensor:
    """(V, n_alt) source-camera ids from a pair.txt, ranking preserved.

    For every reference view, take its ranked source list and keep the first
    ``n_alt`` valid ids — excluding out-of-range ids, the reference itself, and
    duplicates. Errors if a reference view has fewer than ``n_alt`` valid
    sources (so a too-thin pair file fails loudly instead of silently padding).
    """
    ranking = load_pair_file(path)
    alt = torch.empty(n_views, n_alt, dtype=torch.long)
    for r in range(n_views):
        ranked = ranking.get(r)
        if ranked is None:
            raise ValueError(f"pair file {path}: no entry for reference view {r}")
        seen: set[int] = set()
        valid: list[int] = []
        for s in ranked:
            if s == r or s < 0 or s >= n_views or s in seen:
                continue
            seen.add(s)
            valid.append(s)
        if len(valid) < n_alt:
            raise ValueError(
                f"pair file {path}: reference view {r} has only {len(valid)} valid "
                f"source views after excluding self / out-of-range / duplicates, "
                f"need n_alt={n_alt}")
        alt[r] = torch.tensor(valid[:n_alt], dtype=torch.long)
    return alt


def selected_pair_triangulation_angles(views: dict, alt_nn: Tensor,
                                        scene: Path | None = None) -> np.ndarray:
    """(V, n_alt) parallax angle (deg) at the object centre for each selected pair.

    Object centre = centroid of ``sparse_sfm_points.txt`` when present, else the
    world origin. This is a cheap, dataset-agnostic triangulation-angle proxy
    that tracks the per-point SfM median within ~1° on DTU. Logging-only.
    """
    centres = views["c2w"][:, :3, 3].cpu().numpy().astype(np.float64)
    obj = np.zeros(3, dtype=np.float64)
    if scene is not None and (Path(scene) / "sparse_sfm_points.txt").exists():
        pts = np.loadtxt(Path(scene) / "sparse_sfm_points.txt", dtype=np.float64)
        if pts.size:
            obj = pts.reshape(-1, 3).mean(axis=0)
    vec = centres - obj
    vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
    alt = alt_nn.cpu().numpy()
    ang = np.empty(alt.shape, dtype=np.float64)
    for r in range(alt.shape[0]):
        cos = np.clip(vec[alt[r]] @ vec[r], -1.0, 1.0)
        ang[r] = np.degrees(np.arccos(cos))
    return ang
