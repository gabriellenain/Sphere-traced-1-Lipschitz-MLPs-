#!/usr/bin/env python3
"""Fit the 1-Lipschitz FTheta to a COLMAP sparse cloud — closed surface guaranteed.

"Can the sparse SfM points alone seed a usable surface?" — yes, but only if the
SIGN of f is injected somewhere: unsigned point samples cannot decide inside vs
outside (a closed surface IS a sign change), and the textbook IGR/SAL objective

    L = mean_p |f(p)|^2 + lam_off * mean_q exp(-alpha |f(q)|)        (--legacy-igr)

fragments on sparse clouds: the push-away term punches through the zero set in
the gaps between points, and random init settles into specks around clusters.

Default method instead builds the sign by construction, from the points alone
(no normals, no Poisson, no eikonal):

  1. unsigned distance d(x, cloud) on a regular grid over the MC cube;
  2. solid = {d <= eps}, eps ~ 2x median point spacing — the union of eps-balls;
     its boundary is watertight by construction;
  3. flood-fill from the cube border: voxels unreachable from outside are
     interior -> filled (this closes the object and decides the sign);
  4. signed grid = EDT(outside) - EDT(inside): an EXACT SDF of a closed solid —
     and an exact SDF is 1-Lipschitz, i.e. the natural target for FTheta;
  5. regress f onto the grid (MSE, trilinear targets, surface-biased sampling).

The zero set sits eps OUTSIDE the points (inflated like a hull init; training
carves it back). --shift deflates it toward the cloud at the risk of re-opening
thin-coverage gaps. target_mesh.ply is the MC of the grid itself — what the net
is asked to fit. Sibling of fit_gt_sdf.py; reuses its MC / render helpers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lip_tracer.config import ModelConfig
from lip_tracer.model import make_model
from fit_gt_sdf import save_mc_mesh, save_render_png, save_hq_renders, save_loss_plot


def load_points(path: Path) -> np.ndarray:
    """Load an (N,3) cloud. .ply (e.g. the dense MVSFormer++ fused cloud) is read
    via trimesh; anything else is parsed as a whitespace text file (COLMAP
    sparse_sfm_points.txt). Both are assumed to already live in the normalized
    training frame — the MVSFormer cam files are written there too (see
    precompute_mvsformer_depths._decompose_idr_camera), so the fused .ply and the
    sparse .txt share one frame."""
    if path.suffix.lower() == ".ply":
        import trimesh
        m = trimesh.load(str(path), process=False)
        pts = np.asarray(m.vertices, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"no vertices in {path} (got {pts.shape})")
        return pts[:, :3]
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"expected (N,>=3) point file, got {pts.shape} from {path}")
    return pts[:, :3]


def clip_to_roi(pts: np.ndarray, roi_path: Path, pad: float) -> np.ndarray:
    """Keep only points inside the padded AABB of a reference cloud (the sparse
    SfM points). The dense MVSFormer cloud spans the whole scene — table, walls,
    floor — which the eps-ball solid would otherwise swallow; the SfM points
    cluster on the object, so their AABB is the cheap object ROI the carve uses
    via --sfm-clip."""
    ref = load_points(roi_path)
    lo, hi = sfm_roi(ref, pad)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    print(f"  ROI clip to {roi_path.name} AABB (pad={pad:g}): "
          f"[{lo.round(3)} .. {hi.round(3)}]  kept {keep.sum():,}/{len(pts):,}", flush=True)
    return pts[keep]


def remove_dominant_plane(pts: np.ndarray, thr: float, iters: int = 400,
                          min_frac: float = 0.12, seed: int = 0) -> np.ndarray:
    """RANSAC the single largest plane and drop its inliers. The dense MVS cloud
    is dominated by the support table / back walls the object rests on; those are
    the biggest planar structure in the ROI, so the dominant plane is (almost
    always) the table. Only removed if its inliers exceed min_frac of the cloud,
    a guard against gutting a genuinely planar OBJECT face (e.g. scan24 house)."""
    rng = np.random.default_rng(seed)
    N = len(pts)
    score = pts if N <= 200_000 else pts[rng.choice(N, 200_000, replace=False)]
    best_inl = best_n = best_d = None
    best_cnt = -1
    for _ in range(iters):
        s = pts[rng.choice(N, 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(n @ s[0])
        cnt = int((np.abs(score @ n + d) < thr).sum())
        if cnt > best_cnt:
            best_cnt, best_n, best_d = cnt, n, d
    inl = np.abs(pts @ best_n + best_d) < thr
    frac = float(inl.mean())
    if frac < min_frac:
        print(f"  dominant plane: inliers {frac:.3f} < min_frac {min_frac:g}; "
              f"NOT removed (no table-like plane)", flush=True)
        return pts
    print(f"  dominant plane removed: normal={best_n.round(3)} inliers {inl.sum():,} "
          f"({frac:.3f}); kept {(~inl).sum():,}/{N:,}", flush=True)
    return pts[~inl]


def keep_largest_cluster(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Keep the largest 26-connected component on a coarse occupancy grid — after
    plane removal this isolates the object from floaters and severed table scraps."""
    from scipy.ndimage import label
    keys = np.floor((pts - pts.min(0)) / voxel).astype(np.int64)
    dims = tuple(keys.max(0) + 3)
    occ = np.zeros(dims, dtype=bool)
    occ[keys[:, 0] + 1, keys[:, 1] + 1, keys[:, 2] + 1] = True
    lab, _ = label(occ, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = sizes.argmax()
    vl = lab[keys[:, 0] + 1, keys[:, 1] + 1, keys[:, 2] + 1]
    keep = vl == big
    print(f"  largest cluster (voxel={voxel:g}): kept {keep.sum():,}/{len(pts):,} points", flush=True)
    return pts[keep]


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """One representative point per occupied voxel — flattens the per-view density
    bias of a dense MVS cloud to a uniform sheet, so the eps-ball connectivity
    search sees true surface gaps, not overlap, and the KDTree stays tractable."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    print(f"  voxel downsample (voxel={voxel:g}): {len(idx):,}/{len(pts):,} points", flush=True)
    return pts[np.sort(idx)]


def _read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        color = header == b"PF"
        dim = f.readline().decode("ascii").strip()
        while dim.startswith("#"):
            dim = f.readline().decode("ascii").strip()
        w, h = map(int, dim.split())
        scale = float(f.readline().decode("ascii").strip())
        data = np.frombuffer(f.read(), dtype="<f" if scale < 0 else ">f")
    data = data.reshape((h, w, 3) if color else (h, w))
    return np.flipud(data).copy()


def _read_mvs_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """MVSNet cam.txt -> (world-to-camera extrinsic, intrinsic)."""
    lines = [l.strip() for l in path.read_text().splitlines()]
    i = lines.index("extrinsic")
    ext = np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(4)],
                   dtype=np.float32)
    j = lines.index("intrinsic")
    K = np.array([[float(x) for x in lines[j + 1 + r].split()] for r in range(3)],
                 dtype=np.float32)
    return ext, K


def resolve_mvs_depth_root(points_path: Path, root: Path | None) -> Path | None:
    """Auto-resolve the sibling MVSFormer depth directory for a fused scanXX.ply."""
    if root is not None:
        return root
    cand = points_path.parent / points_path.stem
    if (cand / "depth_est").is_dir() and (cand / "cams").is_dir():
        return cand
    return None


def load_mvs_ray_samples(root: Path, conf_thr: float, max_rays: int, pix_stride: int,
                         roi: tuple[np.ndarray, np.ndarray] | None,
                         seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample MVSFormer depth rays as (camera centre, unit world ray, ray distance).

    The depth map stores camera-space z, while the sign constraints are expressed
    as c + (d +/- tau) u with ||u||=1, so d is the Euclidean distance to the
    backprojected depth point, not the z value.
    """
    pfms = sorted((root / "depth_est").glob("*.pfm"))
    if not pfms:
        raise FileNotFoundError(f"no depth_est/*.pfm under {root}")
    rng = np.random.default_rng(seed)
    per_view = max(1, int(np.ceil(max_rays / max(1, len(pfms)))))
    all_o, all_u, all_d = [], [], []
    lo = hi = None
    if roi is not None:
        lo, hi = roi

    for pfm in pfms:
        stem = pfm.stem
        depth = np.asarray(_read_pfm(pfm), dtype=np.float32)
        conf_path = root / "confidence" / f"{stem}.npy"
        conf = np.load(conf_path).astype(np.float32) if conf_path.exists() else np.ones_like(depth)
        if conf.max() > 1.5:
            conf /= 255.0
        ext, K = _read_mvs_cam(root / "cams" / f"{stem}_cam.txt")

        H, W = depth.shape
        ys = np.arange(0, H, pix_stride)
        xs = np.arange(0, W, pix_stride)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        z = depth[gy, gx]
        ok = (z > 1e-3) & (conf[gy, gx] >= conf_thr)
        if not ok.any():
            continue
        gx, gy, z = gx[ok], gy[ok], z[ok]

        x_cam = (gx.astype(np.float32) - K[0, 2]) / K[0, 0] * z
        y_cam = (gy.astype(np.float32) - K[1, 2]) / K[1, 1] * z
        X_cam = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
        ray_d = np.linalg.norm(X_cam, axis=1).astype(np.float32)
        ray_u_cam = X_cam / np.maximum(ray_d[:, None], 1e-8)

        R, t = ext[:3, :3], ext[:3, 3]
        center = (-R.T @ t).astype(np.float32)
        ray_u = (ray_u_cam @ R).astype(np.float32)

        if lo is not None and hi is not None:
            X_world = center[None, :] + ray_d[:, None] * ray_u
            in_roi = np.all((X_world >= lo) & (X_world <= hi), axis=1)
            if not in_roi.any():
                continue
            ray_u, ray_d = ray_u[in_roi], ray_d[in_roi]

        if len(ray_d) > per_view:
            sel = rng.choice(len(ray_d), per_view, replace=False)
            ray_u, ray_d = ray_u[sel], ray_d[sel]
        all_o.append(np.broadcast_to(center, ray_u.shape).copy())
        all_u.append(ray_u)
        all_d.append(ray_d)

    if not all_d:
        raise RuntimeError(f"no MVS ray samples survived confidence/ROI filters under {root}")
    origins = np.concatenate(all_o).astype(np.float32)
    dirs = np.concatenate(all_u).astype(np.float32)
    dists = np.concatenate(all_d).astype(np.float32)
    print(f"  ray sign samples: {len(dists):,} from {len(pfms)} views  "
          f"conf>={conf_thr:g}  pix_stride={pix_stride}  root={root}", flush=True)
    return origins, dirs, dists


def load_mvs_depth_bank(root: Path, conf_thr: float, pix_stride: int, device: str):
    """Load every MVSFormer depth map + camera as a GPU bank for the MVSDF L_D loss.

    Returns a dict of stacked per-view tensors (V = #views, h,w = strided map size):
      depth   (V,h,w)   camera-space z of the MVS surface (0 where invalid)
      valid   (V,h,w)   bool: positive depth AND confidence >= conf_thr
      normal  (V,h,w,3) WORLD surface normal n_d, oriented toward the camera
      R       (V,3,3)   world->camera rotation;  c (V,3) camera centre
      fx,fy,cx,cy (V,)  intrinsics in the strided pixel grid
      hw      (h,w)

    L_D needs, for an arbitrary sample x, the surface point on x's projected ray
    and the local surface normal; this bank keeps the full maps (strided) so any x
    can be projected into any view at train time. Normals are estimated by cross
    product of the world backprojection's pixel-neighbour differences (the same
    n_d the paper reads off the depth map)."""
    pfms = sorted((root / "depth_est").glob("*.pfm"))
    if not pfms:
        raise FileNotFoundError(f"no depth_est/*.pfm under {root}")
    depths, valids, normals, Rs, cs, fxs, fys, cxs, cys = ([] for _ in range(9))
    h = w = None
    for pfm in pfms:
        stem = pfm.stem
        depth = np.asarray(_read_pfm(pfm), dtype=np.float32)[::pix_stride, ::pix_stride]
        conf_path = root / "confidence" / f"{stem}.npy"
        conf = (np.load(conf_path).astype(np.float32) if conf_path.exists()
                else np.full_like(depth, 255.0))[::pix_stride, ::pix_stride]
        if conf.max() > 1.5:
            conf = conf / 255.0
        ext, K = _read_mvs_cam(root / "cams" / f"{stem}_cam.txt")
        R, t = ext[:3, :3].astype(np.float32), ext[:3, 3].astype(np.float32)
        c = (-R.T @ t).astype(np.float32)
        fx, fy = K[0, 0] / pix_stride, K[1, 1] / pix_stride
        cx, cy = K[0, 2] / pix_stride, K[1, 2] / pix_stride

        H, W = depth.shape
        h = h or H; w = w or W
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        # backproject to world to estimate the surface normal n_d
        x_cam = (xs.astype(np.float32) - cx) / fx * depth
        y_cam = (ys.astype(np.float32) - cy) / fy * depth
        P_cam = np.stack([x_cam, y_cam, depth], axis=-1)               # (H,W,3)
        P_world = P_cam @ R + c                                        # (H,W,3)
        dv = np.zeros_like(P_world); du = np.zeros_like(P_world)
        dv[:-1] = P_world[1:] - P_world[:-1]
        du[:, :-1] = P_world[:, 1:] - P_world[:, :-1]
        n = np.cross(du, dv)
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.maximum(nn, 1e-12)
        toward = (c[None, None, :] - P_world)
        flip = np.sum(n * toward, axis=-1, keepdims=True) < 0          # orient toward cam
        n = np.where(flip, -n, n).astype(np.float32)

        valid = (depth > 1e-3) & (conf >= conf_thr)
        # a normal needs both forward neighbours valid and a non-degenerate cross
        nb = np.zeros((H, W), bool); nb[:-1, :-1] = True
        valid = valid & nb & (nn[..., 0] > 1e-9)

        depths.append(depth); valids.append(valid); normals.append(n)
        Rs.append(R); cs.append(c)
        fxs.append(fx); fys.append(fy); cxs.append(cx); cys.append(cy)

    t_ = lambda a, dt=torch.float32: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    bank = dict(
        depth=t_(depths), valid=t_(valids, torch.bool), normal=t_(normals),
        R=t_(Rs), c=t_(cs),
        fx=t_(fxs), fy=t_(fys), cx=t_(cxs), cy=t_(cys), hw=(h, w))
    nval = int(bank["valid"].sum().item())
    print(f"  L_D depth bank: {len(pfms)} views @ {h}x{w} (stride {pix_stride})  "
          f"conf>={conf_thr:g}  valid px {nval:,} "
          f"({nval / (len(pfms) * h * w):.2%})  root={root}", flush=True)
    return bank


def ld_target(x: torch.Tensor, bank: dict, t_out: int):
    """MVSDF L_D target l(x) for sample points x (B,3), fused across all views.

    Per view the signed distance is  l_v = sgn[(x_D - x)·v] (-n_d·v) ||x_D - x||
    with x_D the depth surface point on x's projected ray and v the unit view ray
    (eq. in Zhang et al. 2021). Views are fused by majority vote: a point is
    OUTSIDE iff >= t_out views see it as outside; the target is then the minimum
    absolute distance over the sign-consistent views. Returns (target, keep_mask).
    """
    B = x.shape[0]
    V = bank["depth"].shape[0]
    h, w = bank["hw"]
    l = torch.zeros(B, V, device=x.device)
    valid = torch.zeros(B, V, dtype=torch.bool, device=x.device)
    for v in range(V):
        R, c = bank["R"][v], bank["c"][v]
        fx, fy, cx, cy = bank["fx"][v], bank["fy"][v], bank["cx"][v], bank["cy"][v]
        Xc = (x - c) @ R.T                                  # world->camera
        z = Xc[:, 2]
        front = z > 1e-4
        zc = z.clamp_min(1e-4)
        u = fx * Xc[:, 0] / zc + cx
        vv = fy * Xc[:, 1] / zc + cy
        inb = front & (u >= 0) & (u <= w - 1) & (vv >= 0) & (vv <= h - 1)
        # bilinear sample depth / normal / validity at the projected pixel
        gx = (u / (w - 1) * 2 - 1).clamp(-1, 1)
        gy = (vv / (h - 1) * 2 - 1).clamp(-1, 1)
        grid = torch.stack([gx, gy], dim=-1).view(1, B, 1, 2)
        samp = lambda vol, mode="bilinear": torch.nn.functional.grid_sample(
            vol, grid, mode=mode, padding_mode="border", align_corners=True)
        d_p = samp(bank["depth"][v][None, None])[0, :, :, 0].reshape(B)
        n_d = samp(bank["normal"][v].permute(2, 0, 1)[None])[0, :, :, 0].permute(1, 0)
        vmask = samp(bank["valid"][v][None, None].float(), mode="nearest")[0, 0, :, 0].reshape(B)
        ok = inb & (vmask > 0.5) & (d_p > 1e-3)
        # surface point on x's ray: backproject pixel (u,vv) at depth d_p
        Xd_cam = torch.stack([(u - cx) / fx * d_p, (vv - cy) / fy * d_p, d_p], dim=-1)
        x_D = Xd_cam @ R + c                                # camera->world
        ray = x_D - c
        ray = ray / ray.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        diff = x_D - x
        dist = diff.norm(dim=-1)
        sgn = torch.sign((diff * ray).sum(-1))              # + when x in front (outside)
        fore = (-(n_d * ray).sum(-1)).clamp(0.05, 1.0)      # foreshortening cos
        l[:, v] = sgn * fore * dist
        valid[:, v] = ok

    pos = valid & (l > 0)
    neg = valid & (l < 0)
    n_pos = pos.sum(1)
    outside = n_pos >= t_out
    BIG = torch.full_like(l, 1e9)
    # outside: min positive distance; inside: negative closest to 0 (= -min|neg|)
    min_pos = torch.where(pos, l, BIG).min(1).values
    max_neg = torch.where(neg, l, -BIG).max(1).values
    target = torch.where(outside, min_pos, max_neg)
    keep = torch.where(outside, n_pos >= t_out, neg.sum(1) >= 1)
    return target, keep


def sfm_roi(pts: np.ndarray, pad: float) -> tuple[np.ndarray, np.ndarray]:
    """Padded axis-aligned bbox of the SfM cloud. pad is a fraction of extent."""
    lo, hi = pts.min(0), pts.max(0)
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def remove_outliers(pts: np.ndarray, k: int = 8, factor: float = 3.0) -> np.ndarray:
    """Statistical outlier removal: drop points whose k-th NN distance exceeds
    factor x the median. COLMAP sparse clouds always carry stray triangulations;
    each one would otherwise mint its own floating eps-ball speck."""
    from scipy.spatial import cKDTree
    dk = cKDTree(pts).query(pts, k=k + 1, workers=-1)[0][:, -1]
    keep = dk <= factor * np.median(dk)
    print(f"  outlier filter (k={k}, x{factor:g} median): kept {keep.sum():,}/{len(pts):,} "
          f"(dropped {(~keep).sum():,})", flush=True)
    return pts[keep]


def distance_grid(pts: np.ndarray, bound: float, grid_res: int) -> tuple[np.ndarray, float]:
    """Unsigned distance to the cloud on a regular grid over [-bound,bound]^3."""
    from scipy.spatial import cKDTree
    G = grid_res
    xs = np.linspace(-bound, bound, G, dtype=np.float32)
    voxel = float(xs[1] - xs[0])
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    grid_pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    d = cKDTree(pts).query(grid_pts, workers=-1)[0].reshape(G, G, G)
    return d.astype(np.float32), voxel


def _fill_and_count(d: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Flood-fill the eps-ball solid from the cube border; return it + #components."""
    from scipy.ndimage import label
    solid = d <= eps
    lab, _ = label(~solid)
    border = np.unique(np.concatenate([
        lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
        lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    border = border[border != 0]
    solid_filled = ~np.isin(lab, border)
    _, n_comp = label(solid_filled)
    return solid_filled, n_comp


def auto_eps_connect(d: np.ndarray, eps_min: float, voxel: float,
                     max_components: int = 1) -> float:
    """Smallest eps (to ~1 voxel) whose filled solid has <= max_components.

    Median NN spacing says nothing about the LARGEST gaps in a sparse cloud, so
    a fixed eps either fragments or over-inflates. Re-thresholding the same
    distance grid is cheap, so search for the connectivity transition instead.
    """
    hi = eps_min
    for _ in range(20):
        _, n = _fill_and_count(d, hi)
        print(f"    eps={hi:.4g} -> {n} components", flush=True)
        if n <= max_components:
            break
        hi *= 1.4
    else:
        print(f"  [warn] no eps <= {hi:.4g} reaches {max_components} components; "
              f"using it anyway", flush=True)
        return hi
    lo = hi / 1.4 if hi > eps_min else eps_min
    while hi - lo > voxel:
        mid = 0.5 * (lo + hi)
        _, n = _fill_and_count(d, mid)
        if n <= max_components:
            hi = mid
        else:
            lo = mid
    return hi


def signed_grid(d: np.ndarray, eps: float, voxel: float,
                largest_component: bool = False) -> np.ndarray:
    """Exact SDF grid (positive outside) of the flood-filled union of eps-balls.
    Its zero level set is watertight by construction (boundary of a union of
    closed balls, interior cavities filled by the border flood-fill)."""
    from scipy.ndimage import distance_transform_edt, label
    solid_filled, n_comp = _fill_and_count(d, eps)
    if largest_component and n_comp > 1:
        # A small eps on a dense MVS cloud leaves real fragments + floaters as
        # separate components; keep only the biggest so the init is a single
        # clean object (mirrors the carve's --largest-component).
        lab, n_before = label(solid_filled)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        solid_filled = lab == sizes.argmax()
        print(f"  largest-component: kept {solid_filled.sum():,} voxels "
              f"(dropped {n_before - 1} fragments)", flush=True)
        n_comp = 1
    fill_frac = solid_filled.mean()
    print(f"  eps-ball solid: filled={fill_frac:.4f} of cube  components={n_comp}  "
          f"(eps={eps:.4g}, voxel={voxel:.4g})", flush=True)
    if fill_frac > 0.9:
        print("  [warn] solid fills >90% of the cube — eps likely too large or "
              "flood-fill leaked; check bound/eps", flush=True)
    inside_d = distance_transform_edt(solid_filled, sampling=voxel)
    outside_d = distance_transform_edt(~solid_filled, sampling=voxel)
    return (outside_d - inside_d).astype(np.float32)


class GridSDF:
    """Trilinear sampler of the target grid on the training device."""

    def __init__(self, sdf_grid: np.ndarray, bound: float, device: str):
        self.vol = torch.from_numpy(sdf_grid)[None, None].to(device)  # (1,1,Dx,Hy,Wz)
        self.bound = bound

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        u = (x / self.bound).clamp(-1.0, 1.0)
        # grid_sample coord order is (w,h,d) = our (z,y,x) — flip the last dim.
        g = u.flip(-1).view(1, -1, 1, 1, 3)
        out = torch.nn.functional.grid_sample(
            self.vol, g, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(-1)

    def self_test(self, sdf_grid: np.ndarray, n: int = 1000) -> float:
        """Max |grid_sample - scipy trilinear| on random points (axis-order check)."""
        from scipy.ndimage import map_coordinates
        G = sdf_grid.shape[0]
        rs = np.random.default_rng(0)
        x = (rs.random((n, 3), dtype=np.float32) * 2 - 1) * self.bound
        idx = (x + self.bound) / (2 * self.bound) * (G - 1)
        ref = map_coordinates(sdf_grid, idx.T, order=1)
        got = self(torch.from_numpy(x).to(self.vol.device)).cpu().numpy()
        return float(np.abs(got - ref).max())


def main() -> None:
    mc = ModelConfig()
    ap = argparse.ArgumentParser(description="Fit FTheta to a COLMAP sparse cloud (closed by construction).")
    ap.add_argument("--points", type=Path, required=True,
                    help="point cloud in the normalized training frame: COLMAP "
                         "sparse_sfm_points.txt, or a dense .ply (MVSFormer++ fused cloud)")
    ap.add_argument("--roi-points", type=Path, default=None,
                    help="clip --points to the padded AABB of this reference cloud "
                         "(typically sparse_sfm_points.txt) before fitting; drops the "
                         "table/floor/wall context a dense MVS cloud carries")
    ap.add_argument("--roi-clip-pad", type=float, default=0.05,
                    help="fractional padding of the --roi-points AABB used for clipping")
    ap.add_argument("--voxel-downsample", type=float, default=0.0,
                    help="keep one point per voxel of this size before fitting "
                         "(uniform density for a dense MVS cloud; 0 = off)")
    ap.add_argument("--remove-plane", action="store_true",
                    help="RANSAC out the dominant plane (the support table/back wall) "
                         "so the init is the object, not object-on-slab")
    ap.add_argument("--plane-thr", type=float, default=0.01,
                    help="RANSAC plane inlier distance threshold (normalized units)")
    ap.add_argument("--plane-min-frac", type=float, default=0.12,
                    help="only remove the dominant plane if its inliers exceed this "
                         "fraction of the cloud (guards a planar object face)")
    ap.add_argument("--keep-largest-cluster", type=float, default=0.0,
                    help="after preprocessing, keep the largest connected component on "
                         "an occupancy grid of this voxel size (isolates the object; 0=off)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/points_sdf_fit"))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eps", type=float, default=0.0,
                    help="ball radius for the solid; 0 = auto (smallest eps whose "
                         "filled solid has <= max-components components)")
    ap.add_argument("--max-components", type=int, default=1,
                    help="auto-eps target: allowed connected components of the solid")
    ap.add_argument("--largest-component", action="store_true",
                    help="keep only the largest connected component of the eps-ball "
                         "solid (drops dense-cloud fragments/floaters; eps-ball method only)")
    ap.add_argument("--sor-k", type=int, default=8,
                    help="outlier filter: k-th nearest neighbour used for the test")
    ap.add_argument("--sor-factor", type=float, default=3.0,
                    help="outlier filter: drop points with kNN dist > factor x median")
    ap.add_argument("--no-sor", action="store_true",
                    help="disable the statistical outlier filter on the cloud")
    ap.add_argument("--clean-only", action="store_true",
                    help="run only the cloud preprocessing (ROI clip, voxel downsample, "
                         "plane removal, largest cluster, outlier filter), write the cleaned "
                         "cloud + stats to --out-dir, and exit before fitting any SDF. For "
                         "eyeballing whether a dense MVS cloud is clean enough to seed init.")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="deflate the fitted level set toward the cloud by this distance "
                         "(target+shift). 0 keeps the guaranteed-closed eps-offset surface")
    ap.add_argument("--grid-res", type=int, default=256,
                    help="target SDF grid resolution over the MC cube")
    ap.add_argument("--ld", action="store_true",
                    help="MVSDF L_D mode (Zhang et al. 2021): fit f with ONLY the direct "
                         "depth-supervision loss L1(f(x) - l(x)), l from MVSFormer depth maps. "
                         "No eps-ball / no IGR. Needs --mvs-depth-root or sibling depth dir.")
    ap.add_argument("--ld-conf-thr", type=float, default=0.5,
                    help="(L_D) confidence threshold for valid depth pixels")
    ap.add_argument("--ld-pix-stride", type=int, default=4,
                    help="(L_D) stride for downsampling depth/normal maps in the bank")
    ap.add_argument("--ld-t-out", type=int, default=2,
                    help="(L_D) views voting 'outside' required to call a sample outside")
    ap.add_argument("--ld-surface-frac", type=float, default=0.5,
                    help="(L_D) fraction of each batch jittered around the MVS surface cloud "
                         "(rest uniform in the cube). The paper samples uniform-in-space PLUS "
                         "points jittered off the MVS depth surface to recover thin topology.")
    ap.add_argument("--ld-surface-sigma", type=float, default=0.02,
                    help="(L_D) Gaussian jitter std for the surface-sampled half")
    ap.add_argument("--ld-eikonal-weight", type=float, default=0.1,
                    help="(L_D) eikonal weight w_E (paper uses 0.1); 0 disables. "
                         "Regularizes |grad f|->1 so the field stays a smooth SDF.")
    ap.add_argument("--legacy-igr", action="store_true",
                    help="old objective: |f(p)|^2 + lam_off exp(-alpha|f(q)|) (fragments on sparse clouds)")
    ap.add_argument("--alpha", type=float, default=100.0,
                    help="(legacy) SAL off-surface sharpness: exp(-alpha|f(q)|)")
    ap.add_argument("--lam-off", type=float, default=0.1,
                    help="(legacy) weight on the off-surface term")
    ap.add_argument("--roi-pad", type=float, default=0.1,
                    help="(legacy) off-surface samples drawn inside the SfM AABB padded by this "
                         "fraction of its extent; set <0 to sample the full [-bound,bound] cube")
    ap.add_argument("--mvs-depth-root", type=Path, default=None,
                    help="(legacy ray sign) MVSFormer root containing depth_est/, cams/, "
                         "confidence/. If omitted, auto-resolves sibling dir next to scanXX.ply.")
    ap.add_argument("--ray-sign-weight", type=float, default=0.0,
                    help="(legacy) weight for ray sign losses: "
                         "f(c+(d-tau)u)>0 and f(c+(d+tau)u)<0; 0 disables")
    ap.add_argument("--ray-sign-tau", type=float, default=0.016,
                    help="(legacy) offset distance tau along each MVS depth ray")
    ap.add_argument("--ray-sign-margin", type=float, default=0.0,
                    help="(legacy) optional hinge margin for ray sign losses")
    ap.add_argument("--ray-sign-conf-thr", type=float, default=0.5,
                    help="(legacy) confidence threshold for MVS depth ray samples")
    ap.add_argument("--ray-sign-max-rays", type=int, default=500_000,
                    help="(legacy) maximum cached MVS depth rays for sign supervision")
    ap.add_argument("--ray-sign-pix-stride", type=int, default=4,
                    help="(legacy) pixel stride when collecting MVS depth ray samples")
    ap.add_argument("--ray-sign-batch", type=int, default=4096,
                    help="(legacy) ray sign samples per optimization step")
    ap.add_argument("--bound", type=float, default=1.0, help="marching-cubes half-extent")
    ap.add_argument("--mc-res", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=mc.hidden)
    ap.add_argument("--depth", type=int, default=mc.depth)
    ap.add_argument("--group-size", type=int, default=mc.group_size)
    ap.add_argument("--activation",
                    choices=["groupsort", "nact", "softplus", "centered_softplus", "softplus_cpl"],
                    default=mc.activation)
    ap.add_argument("--input-encoding", choices=["identity", "pe"], default=mc.input_encoding)
    ap.add_argument("--multires", type=int, default=mc.multires)
    ap.add_argument("--architecture", choices=["cpl", "neus"], default=mc.architecture)
    ap.add_argument("--lipschitz-mode", choices=["none", "uniform", "per_band"],
                    default=mc.lipschitz_mode)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="skip the slow HQ render")
    ap.add_argument("--save-grid", type=Path, default=None,
                    help="also dump the eps-ball signed-distance grid as .npy, transposed to "
                         "the [z,y,x] axis order the training init_sdf_grid loader expects "
                         "(matches sphere_occ / save_sdf_grid). Drop-in for --init-sdf-grid. "
                         "eps-ball method only (ignored with --legacy-igr).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts_np = load_points(args.points)
    print(f"points: {args.points}")
    print(f"  N={len(pts_np):,}  r_mean={np.linalg.norm(pts_np, axis=1).mean():.3f}  "
          f"aabb=[{pts_np.min(0).round(3)} .. {pts_np.max(0).round(3)}]")
    if args.roi_points is not None:
        pts_np = clip_to_roi(pts_np, args.roi_points, args.roi_clip_pad)
    if args.voxel_downsample > 0:
        pts_np = voxel_downsample(pts_np, args.voxel_downsample)
    if args.remove_plane:
        pts_np = remove_dominant_plane(pts_np, args.plane_thr,
                                       min_frac=args.plane_min_frac, seed=args.seed)
    if args.keep_largest_cluster > 0:
        pts_np = keep_largest_cluster(pts_np, args.keep_largest_cluster)

    if args.clean_only:
        if not args.no_sor:
            pts_np = remove_outliers(pts_np, k=args.sor_k, factor=args.sor_factor)
        import trimesh
        out_ply = args.out_dir / "cleaned.ply"
        trimesh.PointCloud(pts_np).export(out_ply)
        lo, hi = pts_np.min(0), pts_np.max(0)
        stats = {
            "n_points": int(len(pts_np)),
            "aabb_min": lo.round(4).tolist(),
            "aabb_max": hi.round(4).tolist(),
            "extent": (hi - lo).round(4).tolist(),
            "r_mean": float(np.linalg.norm(pts_np, axis=1).mean()),
            "centroid": pts_np.mean(0).round(4).tolist(),
        }
        (args.out_dir / "clean_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"  clean-only: wrote {len(pts_np):,} points -> {out_ply}")
        print(f"  stats: {json.dumps(stats)}", flush=True)
        return

    pts = torch.from_numpy(pts_np).to(device)
    n_pts = pts.shape[0]

    f = make_model(hidden=args.hidden, depth=args.depth, group_size=args.group_size,
                   activation=args.activation, input_encoding=args.input_encoding,
                   multires=args.multires, architecture=args.architecture,
                   lipschitz_mode=args.lipschitz_mode).to(device)
    opt = torch.optim.Adam(f.parameters(), lr=args.lr)

    history: list[tuple[int, float, float, float]] = []

    if args.ld:
        mvs_root = resolve_mvs_depth_root(args.points, args.mvs_depth_root)
        if mvs_root is None:
            raise ValueError("--ld needs --mvs-depth-root, or --points must be a fused "
                             "scanXX.ply with a sibling scanXX/ depth dir")
        bank = load_mvs_depth_bank(mvs_root, args.ld_conf_thr, args.ld_pix_stride, device)
        w_eik = args.ld_eikonal_weight
        sigma = args.ld_surface_sigma
        for s in range(args.steps + 1):
            # paper sampling: uniform-in-space + points jittered off the MVS surface
            n_srf = int(args.batch * args.ld_surface_frac)
            x_uni = (torch.rand(args.batch - n_srf, 3, device=device) * 2 - 1) * args.bound
            if n_srf > 0:
                idx = torch.randint(0, n_pts, (n_srf,), device=device)
                x_srf = (pts[idx] + sigma * torch.randn(n_srf, 3, device=device)
                         ).clamp(-args.bound, args.bound)
                x = torch.cat([x_uni, x_srf], dim=0)
            else:
                x = x_uni
            with torch.no_grad():
                target, keep = ld_target(x, bank, args.ld_t_out)
            if keep.sum() < 2:
                continue
            x.requires_grad_(True)
            fx = f.sdf(x)
            loss_ld = (fx[keep] - target[keep]).abs().mean()      # L1, exactly L_D
            loss_eik = torch.zeros((), device=device)
            if w_eik > 0:
                g = torch.autograd.grad(fx.sum(), x, create_graph=True)[0]
                loss_eik = (g.norm(dim=-1) - 1.0).square().mean()  # L_E, |grad f|->1
            loss = loss_ld + w_eik * loss_eik
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s % 200 == 0 or s == args.steps:
                history.append((s, loss.item(), loss_ld.item(),
                                w_eik * loss_eik.item()))
                print(f"  step {s:5d}  loss={loss.item():.6f}  L_D={loss_ld.item():.6f}  "
                      f"eik={loss_eik.item():.4f}  kept={keep.float().mean().item():.2%}  "
                      f"|f|={fx.detach().abs().mean().item():.4f}", flush=True)
        eps = float("nan")
    elif args.legacy_igr:
        if args.roi_pad >= 0:
            lo, hi = sfm_roi(pts_np, args.roi_pad)
            print(f"  SFM ROI (pad={args.roi_pad}): [{lo.round(3)} .. {hi.round(3)}]")
        else:
            lo = np.full(3, -args.bound, np.float32)
            hi = np.full(3,  args.bound, np.float32)
            print(f"  off-surface sampling: full cube ±{args.bound}")
        lo_t = torch.from_numpy(lo.astype(np.float32)).to(device)
        hi_t = torch.from_numpy(hi.astype(np.float32)).to(device)

        ray_o = ray_u = ray_d = None
        n_rays = 0
        if args.ray_sign_weight > 0:
            mvs_root = resolve_mvs_depth_root(args.points, args.mvs_depth_root)
            if mvs_root is None:
                raise ValueError("--ray-sign-weight > 0 needs --mvs-depth-root, "
                                 "or --points must be a fused scanXX.ply with a sibling scanXX/ depth dir")
            ray_np = load_mvs_ray_samples(
                mvs_root, conf_thr=args.ray_sign_conf_thr,
                max_rays=args.ray_sign_max_rays,
                pix_stride=args.ray_sign_pix_stride,
                roi=(lo, hi), seed=args.seed)
            ray_o = torch.from_numpy(ray_np[0]).to(device)
            ray_u = torch.from_numpy(ray_np[1]).to(device)
            ray_d = torch.from_numpy(ray_np[2]).to(device)
            n_rays = int(ray_d.shape[0])

        for s in range(args.steps + 1):
            idx = torch.randint(0, n_pts, (min(args.batch, n_pts),), device=device)
            p = pts[idx]
            q = lo_t + (hi_t - lo_t) * torch.rand(args.batch, 3, device=device)
            fp = f.sdf(p)
            loss_surface = (fp ** 2).mean()
            loss_off = torch.exp(-args.alpha * f.sdf(q).abs()).mean()
            loss = loss_surface + args.lam_off * loss_off
            loss_ray = torch.zeros((), device=device)
            front_ok = back_ok = float("nan")
            if n_rays > 0:
                nb = min(args.ray_sign_batch, n_rays)
                ridx = torch.randint(0, n_rays, (nb,), device=device)
                o = ray_o[ridx]
                u = ray_u[ridx]
                d = ray_d[ridx]
                front = o + (d - args.ray_sign_tau).clamp_min(1e-4).unsqueeze(1) * u
                back = o + (d + args.ray_sign_tau).unsqueeze(1) * u
                f_front = f.sdf(front)
                f_back = f.sdf(back)
                margin = args.ray_sign_margin
                loss_front = torch.relu(margin - f_front).square().mean()
                loss_back = torch.relu(f_back + margin).square().mean()
                loss_ray = loss_front + loss_back
                loss = loss + args.ray_sign_weight * loss_ray
                with torch.no_grad():
                    front_ok = (f_front > 0).float().mean().item()
                    back_ok = (f_back < 0).float().mean().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s % 200 == 0 or s == args.steps:
                history.append((s, loss.item(), loss_surface.item(),
                                args.lam_off * loss_off.item()))
                print(f"  step {s:5d}  loss={loss.item():.5f}  surf={loss_surface.item():.6f}  "
                      f"off={loss_off.item():.5f}  ray={loss_ray.item():.5f}  "
                      f"front+={front_ok:.3f} back-={back_ok:.3f}  "
                      f"|f(p)|={fp.detach().abs().mean().item():.4f}",
                      flush=True)
        eps = float("nan")
    else:
        if not args.no_sor:
            pts_np = remove_outliers(pts_np, k=args.sor_k, factor=args.sor_factor)
            pts = torch.from_numpy(pts_np).to(device)
            n_pts = pts.shape[0]

        d_grid, voxel = distance_grid(pts_np, args.bound, args.grid_res)
        eps_min = 1.5 * voxel                          # must be resolvable on the grid
        if args.eps > 0:
            eps = max(args.eps, eps_min)
        else:
            # balls must overlap across the LARGEST surface gaps, not the median
            # spacing — search the connectivity transition of the filled solid.
            print(f"  auto-eps: smallest radius with <= {args.max_components} "
                  f"component(s)", flush=True)
            eps = auto_eps_connect(d_grid, eps_min, voxel, args.max_components)
        print(f"  eps={eps:.4g}  shift={args.shift:g}  grid={args.grid_res}^3", flush=True)

        sdf_grid = signed_grid(d_grid, eps, voxel, largest_component=args.largest_component)
        if args.save_grid is not None:
            # distance_grid is [x,y,z]; the training loader (sphere_occ / fit_to_hull)
            # is [z,y,x] — transpose so this is a drop-in --init-sdf-grid.
            args.save_grid.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_grid, np.transpose(sdf_grid, (2, 1, 0)).astype(np.float32))
            print(f"  saved init grid -> {args.save_grid}  (shape {sdf_grid.shape}, [z,y,x])",
                  flush=True)
        target = GridSDF(sdf_grid, args.bound, device)
        err = target.self_test(sdf_grid)
        assert err < 1e-3 * args.bound, f"grid_sample axis-order self-test failed: {err}"

        # MC of the target itself — the guaranteed-closed surface the net must fit.
        try:
            from skimage import measure
            import trimesh
            v, fc, _, _ = measure.marching_cubes(sdf_grid, level=-args.shift, spacing=(voxel,) * 3)
            v += -args.bound
            tm = trimesh.Trimesh(v, fc, process=False)
            tm.export(args.out_dir / "target_mesh.ply")
            print(f"  target mesh: {len(v):,} verts  {len(fc):,} faces  "
                  f"components={tm.body_count}  watertight={tm.is_watertight}", flush=True)
        except Exception as e:                         # diagnostic artifact only
            print(f"  [warn] target mesh export failed: {e}", flush=True)

        sigma = 3.0 * eps                               # surface-biased half-batch
        for s in range(args.steps + 1):
            nb = args.batch // 2
            q_uni = (torch.rand(nb, 3, device=device) * 2 - 1) * args.bound
            idx = torch.randint(0, n_pts, (args.batch - nb,), device=device)
            q_srf = (pts[idx] + sigma * torch.randn(args.batch - nb, 3, device=device)
                     ).clamp(-args.bound, args.bound)
            q = torch.cat([q_uni, q_srf], dim=0)
            t = target(q) + args.shift
            fq = f.sdf(q)
            loss = ((fq - t) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s % 200 == 0 or s == args.steps:
                with torch.no_grad():
                    # at the cloud points the target is ~ -eps+shift (inside the shell)
                    fp = f.sdf(pts[:min(n_pts, 16384)])
                    drift = (fp - (args.shift - eps)).abs().mean().item()
                history.append((s, loss.item(), loss.item(), drift))
                print(f"  step {s:5d}  mse={loss.item():.6f}  "
                      f"|f(p)-({args.shift - eps:+.3g})|={drift:.4f}", flush=True)

    torch.save({
        "f": f.state_dict(),
        "architecture": f.architecture, "group_size": f.group_size, "depth": f.depth,
        "activation": f.activation, "input_encoding": f.input_encoding,
        "multires": f.multires, "lipschitz_mode": f.lipschitz_mode,
        "points": str(args.points), "bound": args.bound, "eps": eps,
        "shift": args.shift, "legacy_igr": args.legacy_igr,
        "mvs_depth_root": None if args.mvs_depth_root is None else str(args.mvs_depth_root),
        "ray_sign_weight": args.ray_sign_weight,
        "ray_sign_tau": args.ray_sign_tau,
        "ray_sign_margin": args.ray_sign_margin,
        "ray_sign_conf_thr": args.ray_sign_conf_thr,
    }, args.out_dir / "checkpoint_points_sdf.pt")
    save_loss_plot(history, args.out_dir / "loss.png")

    pred_mesh = args.out_dir / "pred_mesh.ply"
    n_v, n_f = save_mc_mesh(f, pred_mesh, args.bound, args.mc_res, device)

    # How faithfully does the extracted surface pass by the cloud? (the default
    # method sits eps-shift OUTSIDE the points by design.) Plus closedness stats.
    if n_f > 0 and n_v > 0:
        import trimesh
        from scipy.spatial import cKDTree
        m = trimesh.load(str(pred_mesh), force="mesh")
        surf, _ = m.sample(min(200_000, max(1, 50 * len(m.faces))), return_index=True)
        d, _ = cKDTree(surf).query(pts_np, workers=-1)   # point -> nearest surface
        fit = {"point_to_surface_mean": float(d.mean()),
               "point_to_surface_p90": float(np.percentile(d, 90)),
               "point_to_surface_max": float(d.max()),
               "expected_offset_eps_minus_shift": (None if (args.legacy_igr or args.ld)
                                                   else float(eps - args.shift)),
               "n_verts": int(n_v), "n_faces": int(n_f),
               "n_components": int(m.body_count),
               "watertight": bool(m.is_watertight)}
        (args.out_dir / "fit.json").write_text(json.dumps(fit, indent=2))
        print(f"fit: point->surface mean={fit['point_to_surface_mean']:.5f}  "
              f"p90={fit['point_to_surface_p90']:.5f}  max={fit['point_to_surface_max']:.5f}  "
              f"components={fit['n_components']}  watertight={fit['watertight']}")

    if not args.sweep and n_f > 0:
        save_render_png(pred_mesh, args.out_dir / "pred_mesh_render.png")
        save_hq_renders(pred_mesh, args.out_dir / "pred_mesh_hq.png")
        print("saved renders -> pred_mesh_render.png, pred_mesh_hq.png")


if __name__ == "__main__":
    main()
