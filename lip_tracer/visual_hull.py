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

def _load_masked_views(scene: Path, view_keep=None) -> dict:
    """Load views with masks for Blender, DTU, or NSVF-T&T scenes.
    For T&T, masks are read from scene/mask/<stem>.png (e.g. SegFormer sky masks)."""
    if (scene / "intrinsics.txt").exists() and (scene / "pose").is_dir():
        return load_views(scene, view_keep=view_keep)   # NSVF-T&T (reads scene/mask/* if present)
    if ((scene / "meta_data.json").exists() or (scene / "cameras.npz").exists()
            or (scene / "cameras_sphere.npz").exists()):  # bmvs ships cameras_sphere.npz
        return load_views(scene, view_keep=view_keep)
    if view_keep is not None:
        raise NotImplementedError("view_keep not supported for Blender scenes")
    return load_blender_views(scene, split="train", down=1)


def _percentile_inside_masks(
    pts: np.ndarray,
    masks: np.ndarray,
    c2ws: np.ndarray,
    Ks: np.ndarray,
    H: int,
    W: int,
    percentile: float = 0.99,
    min_views: int = 8,
) -> np.ndarray:
    """Visibility-aware probabilistic visual hull.

    A voxel V is inside iff:
      n_view(V) >= min_views,                              # not unseen
      n_fg(V) / n_view(V) >= percentile                    # tolerated outliers

    Robust to a handful of noisy masks per voxel — required for many-view
    (e.g. 336) noisy seg-derived silhouettes where strict AND collapses.
    """
    N = len(pts)
    n_view = np.zeros(N, dtype=np.int32)
    n_fg   = np.zeros(N, dtype=np.int32)
    for mask, c2w, K in zip(masks, c2ws, Ks):
        # A fully-empty mask is a segmentation *failure* (detector found no
        # object), not evidence of background. Counting it would inflate n_view
        # without ever adding to n_fg, diluting the foreground ratio and eroding
        # voxels at the silhouette. Treat it as "no observation" → skip entirely.
        if not mask.any():
            continue
        R, t = c2w[:3, :3], c2w[:3, 3]
        cam = (pts - t[None]) @ R
        z   = cam[:, 2]
        valid = z > 1e-3
        zz = np.where(valid, z, 1.0)
        px = (cam[:, 0] / zz) * K[0, 0] + K[0, 2]
        py = (cam[:, 1] / zz) * K[1, 1] + K[1, 2]
        xi = np.floor(np.clip(px, -1, W)).astype(np.int32)
        yi = np.floor(np.clip(py, -1, H)).astype(np.int32)
        in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & valid
        n_view += in_bounds.astype(np.int32)
        if in_bounds.any():
            is_fg = np.zeros(N, dtype=bool)
            is_fg[in_bounds] = mask[yi[in_bounds], xi[in_bounds]]
            n_fg += is_fg.astype(np.int32)
    return (n_view >= min_views) & (n_fg >= percentile * np.maximum(n_view, 1))


def _points_inside_masks(
    pts: np.ndarray,
    masks: np.ndarray,
    c2ws: np.ndarray,
    Ks: np.ndarray,
    H: int,
    W: int,
    min_views: int = 0,
    border_aware: bool = False,
    edge: int = 2,
    return_support: bool = False,
):
    """Return whether each 3D point is consistent with observed silhouette masks.

    border_aware (Fix 1): a voxel that projects *off-frame* in front of a camera
    is treated as background — and therefore carved — only through image edges the
    view's silhouette does NOT touch. Off-frame through a touched edge stays
    ambiguous (the object may continue out of view), so it is kept. This trims the
    classic visual-hull bloat where a large object's silhouette spills past the
    frame and its cone is otherwise unbounded there.

    return_support additionally returns, per voxel, the number of views in which it
    projected in-frame in front of the camera (silhouette support count) — used by
    the SFM gate to find low-support voxels kept only by the border rule.
    """
    inside = np.ones(len(pts), dtype=bool)
    n_view = np.zeros(len(pts), dtype=np.int32)   # in-frame, front-of-camera count
    for mask, c2w, K in zip(masks, c2ws, Ks):
        R, t = c2w[:3, :3], c2w[:3, 3]
        cam  = (pts - t[None]) @ R
        valid = cam[:, 2] > 0
        z = np.where(valid, cam[:, 2], 1.0)   # avoid div-by-zero for behind-camera pts
        px = (cam[:, 0] / z) * K[0, 0] + K[0, 2]
        py = (cam[:, 1] / z) * K[1, 1] + K[1, 2]
        x0 = np.floor(np.clip(px, -1, W)).astype(np.int32)
        y0 = np.floor(np.clip(py, -1, H)).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        in_bounds = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H) & valid
        n_view += in_bounds.astype(np.int32)
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
        carve_here = visible_bg
        if border_aware:
            mb = mask > 0.5
            touch_l = bool(mb[:, :edge].any());  touch_r = bool(mb[:, -edge:].any())
            touch_t = bool(mb[:edge, :].any());  touch_b = bool(mb[-edge:, :].any())
            off_bg = valid & ~in_bounds & (
                ((px < 0)  & (not touch_l)) | ((px >= W) & (not touch_r)) |
                ((py < 0)  & (not touch_t)) | ((py >= H) & (not touch_b)))
            carve_here = carve_here | off_bg
        inside &= ~carve_here
        if not inside.any():
            break
    if min_views > 0:
        inside &= n_view >= min_views
    if return_support:
        return inside, n_view
    return inside


def _sfm_gate(
    inside: np.ndarray,
    pts: np.ndarray,
    support: np.ndarray,
    sfm_pts: np.ndarray,
    conf_views: int = 3,
    margin: float = 0.06,
) -> np.ndarray:
    """Disambiguate low-support hull voxels using COLMAP sparse points.

    Voxels with silhouette support >= conf_views are kept untouched (the
    silhouettes have real evidence there). The remaining *ambiguous* voxels — kept
    only by the border rule, with little in-frame confirmation — are carved unless
    they lie within `margin` (world units) of a COLMAP point, which sits on the
    true surface. Restricting the distance test to the ambiguous set keeps it safe:
    sparse points miss textureless surface, but there the alternative is full bloat.
    """
    from scipy.spatial import cKDTree
    ambiguous = inside & (support < conf_views)
    n_amb = int(ambiguous.sum())
    if n_amb == 0:
        return inside, (0, 0)
    tree = cKDTree(sfm_pts)
    d, _ = tree.query(pts[ambiguous])
    keep = d <= margin
    out = inside.copy()
    amb_idx = np.nonzero(ambiguous)[0]
    out[amb_idx[~keep]] = False
    return out, (n_amb, int((~keep).sum()))


def _carve_sightlines(res: int, bound: float, origins: np.ndarray, points: np.ndarray,
                      eps: float, chunk: int = 8000) -> np.ndarray:
    """Free-space carving: mark every voxel on a camera→(point−eps) sight-line.

    A COLMAP point is triangulated as visible from its cameras, so the segment
    camera→point is unoccluded; the −eps stops short of the surface voxel itself.
    Returns a (res,res,res) bool grid (axes z,y,x) of voxels to carve to free.
    """
    vox = 2 * bound / max(res - 1, 1)
    free = np.zeros((res, res, res), dtype=bool)   # axes (z, y, x), matches occ
    O = np.asarray(origins, np.float64); P = np.asarray(points, np.float64)
    dirs = P - O
    L = np.linalg.norm(dirs, axis=1)
    keep = L > eps
    O, dirs, L = O[keep], dirs[keep], L[keep]
    if len(O) == 0:
        return free
    u = dirs / L[:, None]
    far = L - eps
    m = int(np.ceil(far.max() / vox)) + 2          # ≤1 voxel spacing on the longest ray
    ts = np.linspace(0.0, 1.0, m)[None, :]
    for s in range(0, len(O), chunk):
        o, uu, fr = O[s:s+chunk], u[s:s+chunk], far[s:s+chunk]
        pw = o[:, None, :] + (ts * fr[:, None])[:, :, None] * uu[:, None, :]   # (c,m,3) xyz
        gi = np.round((pw + bound) / vox).astype(np.int64)
        ix, iy, iz = gi[..., 0], gi[..., 1], gi[..., 2]
        ok = ((ix >= 0) & (ix < res) & (iy >= 0) & (iy < res) & (iz >= 0) & (iz < res))
        free[iz[ok], iy[ok], ix[ok]] = True         # occ axes are (z, y, x)
    return free


def carve(scene: Path = BLENDER_SCENE, res: int = 128, bound: float = 1.5,
          roi_bounds: tuple[np.ndarray, np.ndarray] | None = None,
          min_views: int = 0, border_aware: bool = False,
          sfm_gate: bool = False, sfm_pts: np.ndarray | None = None,
          sfm_conf_views: int = 3, sfm_margin_voxels: float = 3.0,
          sfm_sightlines: tuple[np.ndarray, np.ndarray] | None = None,
          sfm_free_eps: float = 0.02,
          vh_percentile: float = 0.99,
          vh_min_views: int = 8,
          view_keep=None) -> np.ndarray:
    """Returns (res, res, res) bool occupancy grid.

    For NSVF-T&T scenes (intrinsics.txt + pose/ + scene/mask/), masks are
    noisy — uses visibility-aware percentile carving instead of strict AND.
    """
    views = _load_masked_views(scene, view_keep=view_keep)
    masks = views["masks"].numpy().astype(np.float32)   # (V, H, W) in [0, 1]
    c2ws  = views["c2w"].numpy()     # (V, 4, 4)
    Ks    = views["K"].numpy()       # (V, 3, 3)
    H, W  = views["H"], views["W"]

    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (N, 3)
    is_tnt = (scene / "intrinsics.txt").exists() and (scene / "pose").is_dir()
    if is_tnt:
        print(f"  [carve] T&T scene -> visibility-aware percentile "
              f"p>={vh_percentile:.3f}, min_views={vh_min_views}")
        inside = _percentile_inside_masks(pts, masks, c2ws, Ks, H, W,
                                          percentile=vh_percentile,
                                          min_views=vh_min_views)
        occ = inside.reshape(res, res, res)
        # Open-air scenes have multiple legitimate components (barn + ground +
        # trees + ...); skip keep_central_component which assumes one central
        # object and would collapse the hull to a few voxels.
        return occ
    else:
        out = _points_inside_masks(pts, masks, c2ws, Ks, H, W,
                                   min_views=min_views,
                                   border_aware=border_aware,
                                   return_support=sfm_gate)
        inside, support = out if sfm_gate else (out, None)
        if border_aware:
            print(f"  [carve] border-aware: off-frame through clear edges → carved")
        if min_views > 0:
            print(f"  [carve] requiring visibility in >= {min_views} views")
        if sfm_sightlines is not None:
            free = _carve_sightlines(res, bound, sfm_sightlines[0], sfm_sightlines[1],
                                     eps=sfm_free_eps).reshape(-1)
            n_before = int(inside.sum())
            inside &= ~free
            print(f"  [carve] SFM sight-line free-carve: {n_before - int(inside.sum())} "
                  f"voxels carved ({int(free.sum())} on rays, eps={sfm_free_eps})")
        if sfm_gate and sfm_pts is not None:
            voxel_size = 2 * bound / max(res - 1, 1)
            margin = sfm_margin_voxels * voxel_size
            inside, (n_amb, n_carved) = _sfm_gate(
                inside, pts, support, sfm_pts,
                conf_views=sfm_conf_views, margin=margin)
            print(f"  [carve] SFM gate: {n_carved}/{n_amb} ambiguous voxels carved "
                  f"(support<{sfm_conf_views}, margin={margin:.3f})")
        if roi_bounds is not None:
            lo, hi = roi_bounds
            inside &= np.all((pts >= lo[None]) & (pts <= hi[None]), axis=1)
            print(f"  [carve] SFM ROI x=[{lo[0]:+.3f},{hi[0]:+.3f}] "
                  f"y=[{lo[1]:+.3f},{hi[1]:+.3f}] "
                  f"z=[{lo[2]:+.3f},{hi[2]:+.3f}]")
        occ = inside.reshape(res, res, res)
        # Border-aware carving removes the off-frame bloat at the source, so the
        # object is the dominant blob → robust largest-component cleanup. Without
        # it the bloat is the largest component, so fall back to the legacy
        # closest-to-grid-center heuristic that historically isolated the object.
        return keep_largest_component(occ) if border_aware else keep_central_component(occ)



def save_views(occ: np.ndarray, out: Path) -> None:
    """Save three axis-aligned max-projections as a single PNG strip."""
    def proj(ax):
        s = occ.max(axis=ax).astype(np.uint8) * 255
        return np.stack([s, s, s], axis=-1)

    strip = np.concatenate([proj(0), proj(1), proj(2)], axis=1)
    Image.fromarray(strip).save(out)
    print(f"saved → {out}")


# ------------------------------------------------------------------ fitting --

def keep_largest_component(occ: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (26-connectivity).

    The robust visual-hull cleanup once silhouette bloat has been removed (e.g. by
    border-aware carving): the object is the dominant blob, stray noise components
    are small. Do NOT use on un-border-aware carvings — there the off-frame bloat
    is itself the largest component.
    """
    from scipy.ndimage import label
    struct = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity
    labeled, n = label(occ, structure=struct)
    if n <= 1:
        return occ
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return labeled == int(sizes.argmax())


def keep_central_component(occ: np.ndarray) -> np.ndarray:
    """Keep the connected component whose center-of-mass is closest to grid center.

    Legacy cleanup for un-border-aware carvings: a bloated hull is roughly centered
    so this isolates the object even when the off-frame bloat is the largest blob.
    Uses 26-connectivity so diagonally-adjacent voxels are connected, preventing
    noisy grids from fragmenting into thousands of 1-voxel components.
    """
    from scipy.ndimage import label, center_of_mass
    struct = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity
    labeled, n = label(occ, structure=struct)
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
    sfm_pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
    w_sfm_free: float = 0.0,
    sfm_free_eps: float = 0.02,
    n_sfm_free: int = 4096,
) -> FTheta:
    """Fit an FTheta SDF to the visual hull occupancy grid.

    sfm_pairs (origins, points): per (camera, COLMAP-point) sight-line. With
    w_sfm_free>0, samples free-space points along each camera→point ray up to
    (point − sfm_free_eps) and penalises f<0 there. A triangulated point is
    unoccluded from its observing camera, so that whole segment is empty; enforcing
    f>0 carves the hull wherever a sight-line passes through it, pulling the surface
    back toward the real points (including into silhouette-invisible concavities).
    """
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
    n_narrow = min(batch // 2,
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
    sfm_o_t: torch.Tensor | None = None
    sfm_p_t: torch.Tensor | None = None
    if sfm_pairs is not None and w_sfm_free > 0:
        sfm_o_t = sfm_pairs[0].to(device=device, dtype=torch.float32)
        sfm_p_t = sfm_pairs[1].to(device=device, dtype=torch.float32)
        n_sfm_free = min(n_sfm_free, int(sfm_o_t.shape[0]))
        print(f"  sfm ray free-space: {sfm_o_t.shape[0]} pairs  "
              f"w={w_sfm_free}  eps={sfm_free_eps}  batch={n_sfm_free}")

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
        pred = pred_raw
        hull_loss = F.mse_loss(pred, target)
        depth_loss = torch.zeros(1, device=device).squeeze()
        if depth_pts_t is not None and n_depth > 0:
            didx = torch.randint(depth_pts_t.shape[0], (n_depth,), device=device)
            depth_loss = f(depth_pts_t[didx]).square().mean()
        cam_loss = F.relu(-f(cam_t)).mean() if cam_t is not None else torch.zeros(1, device=device).squeeze()
        sfm_free_loss = torch.zeros(1, device=device).squeeze()
        if sfm_o_t is not None:
            j = torch.randint(sfm_o_t.shape[0], (n_sfm_free,), device=device)
            o, p = sfm_o_t[j], sfm_p_t[j]
            dirv = p - o
            L = dirv.norm(dim=1, keepdim=True).clamp_min(1e-6)
            # sample distance in [0, L - eps] → free-space points up to point−eps
            far = (L - sfm_free_eps).clamp_min(0.0)
            dist = torch.rand(n_sfm_free, 1, device=device) * far
            s = o + dist * (dirv / L)
            sfm_free_loss = F.relu(-f(s)).mean()
        loss = (hull_loss + w_depth_surface * depth_loss
                + w_cam_free * cam_loss + w_sfm_free * sfm_free_loss)
        opt.zero_grad(); loss.backward()
        if step == 0:
            gnorm = sum(p.grad.norm().item()**2 for p in f.parameters() if p.grad is not None) ** 0.5
            hw = getattr(f, "head_weight", None)
            head_gnorm = hw.grad.norm().item() if (hw is not None and hw.grad is not None) else 0.0
            from .model import ConvexPotentialLayer
            first_cpl = next((m for m in getattr(f, "net", []) if isinstance(m, ConvexPotentialLayer)), None)
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
                f"cam={cam_loss.item():.5f} sfmfree={sfm_free_loss.item():.5f}) "
                f"band={band_width:.4f} narrow={n_narrow}/{batch} "
                f"|sdf-t|={err.mean():.4f}/{err.quantile(.9):.4f} "
                f"narrow={narrow_err:.4f} sign={sign_acc:.1%}/{narrow_sign:.1%} "
                f"sdf=[{pred.mean():+.3f},{pred.std():.3f}] "
                f"t=[{target.mean():+.3f},{target.std():.3f}] "
                f"{grad_str}"
            )
            print(
                f"      split: uniform err/sign/t/p+={uniform_err:.3f}/{uniform_sign:.1%}/{uniform_t:+.3f}/{uniform_pred_pos:.1%} "
                f"narrow err/sign/t/p+={narrow_s_err:.3f}/{narrow_s_sign:.1%}/{narrow_t:+.3f}/{narrow_pred_pos:.1%} "
                f"inside frac={inside_frac:.1%} err/sign/t/p+={inside_err:.3f}/{inside_sign:.1%}/{inside_t:+.3f}/{inside_pred_pos:.1%} "
                f"outside err/sign/t/p+={outside_err:.3f}/{outside_sign:.1%}/{outside_t:+.3f}/{outside_pred_pos:.1%}"
            )
            print(
                f"      range: Rs={rs:.3f} RF_sdf={rf:.3f} RF_raw={rf_raw:.3f} "
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
