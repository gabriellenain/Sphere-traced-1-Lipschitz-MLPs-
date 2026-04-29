"""Minimal visual hull: voxel carving from masks → fit FTheta → PNG."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from .config import BLENDER_SCENE, OUT_DIR, ModelConfig, SCENE
from .data import load_blender_gt_points, load_blender_views, load_views
from .model import FTheta, make_model
from .visualize import _trace_view



# ------------------------------------------------------------------ carving --

def _load_masked_views(scene: Path) -> dict:
    """Load views with masks for either Blender or DTU scenes."""
    if (scene / "meta_data.json").exists() or (scene / "cameras.npz").exists():
        return load_views(scene)
    return load_blender_views(scene, split="train", down=1)


def _points_inside_masks(
    pts: np.ndarray,
    masks: np.ndarray,
    c2ws: np.ndarray,
    Ks: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Return whether each 3D point is consistent with every silhouette mask."""
    inside = np.ones(len(pts), dtype=bool)
    for mask, c2w, K in zip(masks, c2ws, Ks):
        R, t = c2w[:3, :3], c2w[:3, 3]
        cam  = (pts - t[None]) @ R
        valid = cam[:, 2] > 0
        z = np.where(valid, cam[:, 2], 1.0)   # avoid div-by-zero for behind-camera pts
        px = (cam[:, 0] / z) * K[0, 0] + K[0, 2]
        py = (cam[:, 1] / z) * K[1, 1] + K[1, 2]
        x0 = np.floor(px).astype(np.int32)
        y0 = np.floor(py).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        in_bounds = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H) & valid
        visible_bg = np.zeros(len(pts), dtype=bool)
        if np.any(in_bounds):
            x = px[in_bounds]
            y = py[in_bounds]
            x0b, x1b = x0[in_bounds], x1[in_bounds]
            y0b, y1b = y0[in_bounds], y1[in_bounds]
            wx = x - x0b
            wy = y - y0b
            m00 = mask[y0b, x0b]
            m10 = mask[y0b, x1b]
            m01 = mask[y1b, x0b]
            m11 = mask[y1b, x1b]
            mask_val = ((1.0 - wx) * (1.0 - wy) * m00 +
                        wx * (1.0 - wy) * m10 +
                        (1.0 - wx) * wy * m01 +
                        wx * wy * m11)
            visible_bg[in_bounds] = mask_val < 0.5
        inside &= ~visible_bg
        if not inside.any():
            break
    return inside


def carve(scene: Path = BLENDER_SCENE, res: int = 128, bound: float = 1.5) -> np.ndarray:
    """Returns (res, res, res) bool occupancy grid."""
    views = _load_masked_views(scene)
    masks = views["masks"].numpy().astype(np.float32)   # (V, H, W) in [0, 1]
    c2ws  = views["c2w"].numpy()     # (V, 4, 4)
    Ks    = views["K"].numpy()       # (V, 3, 3)
    H, W  = views["H"], views["W"]

    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (N, 3)
    inside = _points_inside_masks(pts, masks, c2ws, Ks, H, W)
    occ = inside.reshape(res, res, res)
    is_dtu = (scene / "cameras.npz").exists() or (scene / "meta_data.json").exists()
    return keep_central_component(occ) if is_dtu else occ


def carve_octree(
    scene: Path = BLENDER_SCENE,
    res: int = 128,
    bound: float = 1.5,
    min_depth: int = 5,
    sample_grid: int = 3,
) -> np.ndarray:
    """Adaptive octree-style carving, rasterized back to a dense grid.

    The tree only subdivides ambiguous cells. Fully-inside cells are accepted
    once `min_depth` is reached, which keeps the carve cheap while preserving
    more detail near silhouette boundaries than a coarse uniform grid.
    """
    if res <= 0 or res & (res - 1):
        raise ValueError(f"octree carve expects power-of-two res, got {res}")
    views = _load_masked_views(scene)
    masks = views["masks"].numpy().astype(np.float32)
    c2ws  = views["c2w"].numpy()
    Ks    = views["K"].numpy()
    H, W  = views["H"], views["W"]

    occ = np.zeros((res, res, res), dtype=bool)
    cell = 2.0 * bound / res
    max_depth = int(np.log2(res))
    min_depth = max(0, min(min_depth, max_depth))
    sample_axis = np.linspace(0.0, 1.0, sample_grid, dtype=np.float32)

    def classify(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> tuple[bool, bool]:
        z_lo, z_hi = -bound + z0 * cell, -bound + z1 * cell
        y_lo, y_hi = -bound + y0 * cell, -bound + y1 * cell
        x_lo, x_hi = -bound + x0 * cell, -bound + x1 * cell
        zs = z_lo + (z_hi - z_lo) * sample_axis
        ys = y_lo + (y_hi - y_lo) * sample_axis
        xs = x_lo + (x_hi - x_lo) * sample_axis
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        inside = _points_inside_masks(pts, masks, c2ws, Ks, H, W)
        return bool(inside.any()), bool(inside.all())

    def recurse(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int, depth: int) -> None:
        any_inside, all_inside = classify(z0, z1, y0, y1, x0, x1)
        if not any_inside:
            return
        if depth == max_depth or (all_inside and depth >= min_depth):
            occ[z0:z1, y0:y1, x0:x1] = True
            return

        z_mid = (z0 + z1) // 2
        y_mid = (y0 + y1) // 2
        x_mid = (x0 + x1) // 2
        for za, zb in ((z0, z_mid), (z_mid, z1)):
            for ya, yb in ((y0, y_mid), (y_mid, y1)):
                for xa, xb in ((x0, x_mid), (x_mid, x1)):
                    if za < zb and ya < yb and xa < xb:
                        recurse(za, zb, ya, yb, xa, xb, depth + 1)

    recurse(0, res, 0, res, 0, res, 0)
    is_dtu = (scene / "cameras.npz").exists() or (scene / "meta_data.json").exists()
    return keep_central_component(occ) if is_dtu else occ


def save_views(occ: np.ndarray, out: Path) -> None:
    """Save three axis-aligned max-projections as a single PNG strip."""
    def proj(ax):
        s = occ.max(axis=ax).astype(np.uint8) * 255
        return np.stack([s, s, s], axis=-1)

    strip = np.concatenate([proj(0), proj(1), proj(2)], axis=1)
    Image.fromarray(strip).save(out)
    print(f"saved → {out}")


# ------------------------------------------------------------------ fitting --

def keep_central_component(occ: np.ndarray) -> np.ndarray:
    """Keep only the connected component closest to the grid center (DTU objects are centered)."""
    from scipy.ndimage import label, center_of_mass
    labeled, n = label(occ)
    if n <= 1:
        return occ
    center = np.array(occ.shape) / 2.0
    best, best_dist = 1, float("inf")
    for i in range(1, n + 1):
        com = np.array(center_of_mass(labeled == i))
        dist = np.linalg.norm(com - center)
        if dist < best_dist:
            best_dist = dist
            best = i
    return labeled == best


def occ_to_sdf(occ: np.ndarray, bound: float) -> tuple[np.ndarray, np.ndarray]:
    """Approximate SDF from occupancy via distance transform.

    Returns (pts, sdf) arrays of shape (N, 3) and (N,).
    """
    from scipy.ndimage import distance_transform_edt
    res = occ.shape[0]
    voxel_size = 2 * bound / max(res - 1, 1)

    dist_in  = distance_transform_edt(occ)        # dist to surface from inside
    dist_out = distance_transform_edt(~occ)        # dist to surface from outside
    sdf_grid = (dist_out - dist_in) * voxel_size  # in world units, negative inside

    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    sdf = sdf_grid.reshape(-1).astype(np.float32)
    return pts, sdf


def fit_to_hull(
    occ: np.ndarray,
    bound: float = 1.5,
    steps: int = 2000,
    batch: int = 8192,
    lr: float = 2e-3,
    cfg: ModelConfig | None = None,
    device: str | None = None,
    depth_points: torch.Tensor | None = None,
    w_depth_surface: float = 0.0,
    cam_origins: np.ndarray | None = None,
    w_cam_free: float = 1.0,
) -> FTheta:
    """Fit an FTheta SDF to the visual hull occupancy grid."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = cfg or ModelConfig()

    pts, sdf = occ_to_sdf(occ, bound)
    pts_t = torch.from_numpy(pts).to(device)
    sdf_t = torch.from_numpy(sdf).to(device)
    voxel_size = 2 * bound / max(occ.shape[0] - 1, 1)
    band_width = 4.0 * voxel_size

    f   = make_model(hidden=cfg.hidden, depth=cfg.depth,
                     group_size=cfg.group_size, activation=cfg.activation,
                     input_encoding=cfg.input_encoding, multires=cfg.multires,
                     architecture=getattr(cfg, "architecture", "cpl")).to(device)
    opt = torch.optim.Adam(f.parameters(), lr=lr)

    N = len(pts_t)
    narrow_idx = torch.nonzero(sdf_t.abs() <= band_width, as_tuple=False).squeeze(-1)
    use_neus_init_balance = cfg.input_encoding == "neus"
    n_narrow = min((3 * batch // 4) if use_neus_init_balance else (batch // 2),
                   int(narrow_idx.numel()))
    depth_pts_t: torch.Tensor | None = None
    n_depth = 0
    if depth_points is not None and w_depth_surface > 0:
        depth_pts_t = depth_points.to(device=device, dtype=torch.float32)
        n_depth = min(batch // 4, int(depth_pts_t.shape[0]))
        print(f"  depth surface init: n_pts={depth_pts_t.shape[0]}  w={w_depth_surface}  batch={n_depth}")
    cam_t: torch.Tensor | None = None
    if cam_origins is not None and w_cam_free > 0:
        cam_t = torch.from_numpy(cam_origins).to(device=device, dtype=torch.float32)
        print(f"  cam-free hull constraint: {len(cam_t)} cameras  w={w_cam_free}")

    for step in range(steps):
        idx_uniform = torch.randint(N, (batch - n_narrow,), device=device)
        if n_narrow > 0:
            idx_narrow = narrow_idx[torch.randint(narrow_idx.numel(), (n_narrow,), device=device)]
            idx = torch.cat([idx_uniform, idx_narrow], dim=0)
        else:
            idx = idx_uniform
        target_full = sdf_t[idx]
        target = target_full
        pred_raw = f(pts_t[idx])
        pred = pred_raw  # fit f.forward directly; 1-Lip guarantee held by f.sdf = f.forward * lip_scale at trace time
        hull_loss = F.mse_loss(pred, target)
        depth_loss = torch.zeros(1, device=device).squeeze()
        if depth_pts_t is not None and n_depth > 0:
            didx = torch.randint(depth_pts_t.shape[0], (n_depth,), device=device)
            depth_loss = f(depth_pts_t[didx]).square().mean()
        cam_loss = F.relu(-f(cam_t)).mean() if cam_t is not None else torch.zeros(1, device=device).squeeze()
        loss = hull_loss + w_depth_surface * depth_loss + w_cam_free * cam_loss
        opt.zero_grad(); loss.backward()
        if step == 0:
            gnorm = sum(p.grad.norm().item()**2 for p in f.parameters() if p.grad is not None) ** 0.5
            hw = getattr(f, "head_weight", None)
            head_gnorm = hw.grad.norm().item() if (hw is not None and hw.grad is not None) else 0.0
            from .model import ConvexPotentialLayer
            first_cpl = next((m for m in f.net if isinstance(m, ConvexPotentialLayer)), None)
            first_gnorm = first_cpl.weight.grad.norm().item() if (first_cpl is not None and first_cpl.weight.grad is not None) else 0.0
            print(f"  [grad@0] total={gnorm:.3e}  head={head_gnorm:.3e}  first_cpl={first_gnorm:.3e}")
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                err = (pred - target).abs()
                sign_acc = ((pred >= 0) == (target >= 0)).float().mean()
                narrow_mask = target_full.abs() <= band_width
                if narrow_mask.any():
                    narrow_err = err[narrow_mask].mean()
                    narrow_sign = ((pred[narrow_mask] >= 0) == (target[narrow_mask] >= 0)).float().mean()
                else:
                    narrow_err = err.new_tensor(0.0)
                    narrow_sign = err.new_tensor(0.0)
                raw_target = target / f.lip_scale
                raw_err = (pred_raw - raw_target).abs().mean()

                n_uniform = batch - n_narrow

                def _split_stats(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    if not mask.any():
                        z = err.new_tensor(0.0)
                        return z, z, z, z
                    e = err[mask].mean()
                    s = ((pred[mask] >= 0) == (target[mask] >= 0)).float().mean()
                    tm = target[mask].mean()
                    pf = (pred[mask] >= 0).float().mean()
                    return e, s, tm, pf

                uniform_pos = torch.zeros_like(target, dtype=torch.bool)
                uniform_pos[:n_uniform] = True
                narrow_pos = ~uniform_pos
                inside = target_full < 0
                outside = ~inside
                uniform_err, uniform_sign, uniform_t, uniform_pred_pos = _split_stats(uniform_pos)
                narrow_s_err, narrow_s_sign, narrow_t, narrow_pred_pos = _split_stats(narrow_pos)
                inside_err, inside_sign, inside_t, inside_pred_pos = _split_stats(inside)
                outside_err, outside_sign, outside_t, outside_pred_pos = _split_stats(outside)
                inside_frac = inside.float().mean()
                pts_batch = pts_t[idx]
                omega_box_diam = torch.linalg.vector_norm(
                    pts_batch.amax(dim=0) - pts_batch.amin(dim=0)
                )
                if f.encoder is not None:
                    enc = f.encoder(pts_batch)
                    enc_box_diam = torch.linalg.vector_norm(enc.amax(dim=0) - enc.amin(dim=0))
                else:
                    enc_box_diam = omega_box_diam
                rs = target.max() - target.min()
                rf = pred.max() - pred.min()
                rf_raw = pred_raw.max() - pred_raw.min()
                rf_bound_enc = enc_box_diam * f.lip_scale
                rf_bound_lip = omega_box_diam

            grad_n = None
            if step % 1000 == 0 or step == steps - 1:
                gidx = idx[:min(256, idx.numel())]
                xg = pts_t[gidx].detach().requires_grad_(True)
                with torch.enable_grad():
                    gg = torch.autograd.grad(f(xg).sum(), xg, create_graph=False)[0]
                grad_n = gg.norm(dim=-1).detach()

            grad_str = ""
            if grad_n is not None:
                grad_str = f"  |grad f|={grad_n.mean():.2f}/{grad_n.quantile(.9):.2f}"

            print(
                f"  [{step:4d}/{steps}] loss={loss.item():.5f} "
                f"(hull={hull_loss.item():.5f} depth={depth_loss.item():.5f} "
                f"cam={cam_loss.item():.5f}) "
                f"K={1.0 / f.lip_scale:.2f} band={band_width:.4f} narrow={n_narrow}/{batch} "
                f"|sdf-t|={err.mean():.4f}/{err.quantile(.9):.4f} "
                f"narrow={narrow_err:.4f} sign={sign_acc:.1%}/{narrow_sign:.1%} "
                f"sdf=[{pred.mean():+.3f},{pred.std():.3f}] "
                f"t=[{target.mean():+.3f},{target.std():.3f}] "
                f"raw|f-Kt|={raw_err:.3f}{grad_str}"
            )
            print(
                f"      split: uniform err/sign/t/p+={uniform_err:.3f}/{uniform_sign:.1%}/{uniform_t:+.3f}/{uniform_pred_pos:.1%} "
                f"narrow err/sign/t/p+={narrow_s_err:.3f}/{narrow_s_sign:.1%}/{narrow_t:+.3f}/{narrow_pred_pos:.1%} "
                f"inside frac={inside_frac:.1%} err/sign/t/p+={inside_err:.3f}/{inside_sign:.1%}/{inside_t:+.3f}/{inside_pred_pos:.1%} "
                f"outside err/sign/t/p+={outside_err:.3f}/{outside_sign:.1%}/{outside_t:+.3f}/{outside_pred_pos:.1%}"
            )
            print(
                f"      range: Rs={rs:.3f} RF_sdf={rf:.3f} RF_raw={rf_raw:.3f} "
                f"bound_sdf<=diam(gamma(batch))/K={rf_bound_enc:.3f} "
                f"lip_bound<=diam(omega_batch)={rf_bound_lip:.3f} "
                f"diam_gamma_box={enc_box_diam:.3f} diam_omega_box={omega_box_diam:.3f}"
            )

    del opt, pts_t, sdf_t   # free optimizer state + large tensors before returning
    return f


# ------------------------------------------------------------------ render ---

LIGHT = np.array([0.5, 0.8, 1.0], dtype=np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def _shade(n_img: np.ndarray, hit_img: np.ndarray) -> np.ndarray:
    """Lambertian grey shading: white background, grey surface."""
    diffuse = np.clip((n_img * 2 - 1) @ LIGHT, 0, 1)          # dot(n, light)
    grey    = (0.25 + 0.75 * diffuse)[..., None].repeat(3, -1) # ambient + diffuse
    bg      = np.ones_like(grey)
    return np.where(hit_img[..., None], grey, bg)


def render_views(f: FTheta, scene: Path = BLENDER_SCENE,
                 n_views: int = 8, down: int = 4) -> np.ndarray:
    """Render grey geometry from n_views evenly-spaced training cameras → (H, W*n, 3)."""
    device = str(next(f.parameters()).device)
    views  = _load_masked_views(scene)
    V      = views["c2w"].shape[0]
    idx    = np.linspace(0, V - 1, n_views, dtype=int)
    H, W   = views["H"] // down, views["W"] // down

    strips = []
    for i, vi in enumerate(idx):
        c2w = views["c2w"][vi].numpy()
        K   = views["K"][vi].numpy()
        print(f"  tracing view {i+1}/{n_views} …")
        n_img, hit_img = _trace_view(f, c2w, K, H, W, down, device)
        img = (_shade(n_img, hit_img) * 255).astype(np.uint8)
        strips.append(img)

    return np.concatenate(strips, axis=1)  # horizontal strip


# -------------------------------------------------------------------- main ---

if __name__ == "__main__":
    RES   = 128
    BOUND = 1.5

    print("=== carving ===")
    occ = carve(scene=SCENE, res=RES, bound=BOUND)
    print(f"  occupied: {occ.sum()} / {occ.size}")
    save_views(occ, OUT_DIR / "visual_hull_occ.png")

    print("=== fitting FTheta ===")
    depth_pts = None
    try:
        depth_pts = load_blender_gt_points(scene=SCENE)
    except (FileNotFoundError, ValueError):
        pass
    f = fit_to_hull(occ, bound=BOUND, depth_points=depth_pts, w_depth_surface=0.0)

    print("=== rendering views ===")
    img = render_views(f, n_views=8, down=4)
    out = OUT_DIR / "visual_hull_fit.png"
    Image.fromarray(img).save(out)
    print(f"saved → {out}")
