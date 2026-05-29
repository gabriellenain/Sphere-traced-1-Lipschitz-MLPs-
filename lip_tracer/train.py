"""Training loop and sphere-SDF initialisation for the 1-Lip tracer."""
from __future__ import annotations

import datetime
import dataclasses
import json
import math
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import SCENE, BLENDER_SCENE, OUT_DIR, Config, ModelConfig, InitConfig, TrainConfig, TraceConfig, EvalConfig, MvsdfScheduleConfig, BundleAdjustConfig
from .data import (load_colmap_points, colmap_visibility_counts, load_camera_centers, load_sfm_pairs,
                   load_views, load_blender_views, load_blender_gt_points,
                   make_deterministic_rays, precompute_alt_cameras)
from .loss import photo_loss, eikonal_loss, cam_free_loss
from .model import FTheta, ConvexPotentialLayer, NeuSMLP, make_model
from .profile import StepProfiler, MemorySnapshot, dump_static_accounting
from .sphere_tracing import trace_unrolled, trace_idr, trace_nograd, get_last_trace_stats


def _image_grad_ray_weights(det: dict, H_d: int, W_d: int) -> torch.Tensor:
    """Per-ray weight ∝ image-gradient magnitude (central differences on luminance).

    det rays are view-major, each view a row-major H_d×W_d grid (see
    make_deterministic_rays), so det["gt"] reshapes to (Vp, H_d, W_d, 3).
    Returns a (N,) CPU tensor of non-negative weights.
    """
    gt  = det["gt"]                                   # (N, 3) cpu float in [0, 1]
    N   = gt.shape[0]
    rpv = H_d * W_d
    Vp  = N // rpv
    lum = (gt.reshape(Vp, H_d, W_d, 3)
           * gt.new_tensor([0.299, 0.587, 0.114])).sum(-1)      # (Vp, H_d, W_d)
    gx = torch.zeros_like(lum); gy = torch.zeros_like(lum)
    gx[:, :, 1:-1] = 0.5 * (lum[:, :, 2:] - lum[:, :, :-2])
    gy[:, 1:-1, :] = 0.5 * (lum[:, 2:, :] - lum[:, :-2, :])
    return torch.sqrt(gx * gx + gy * gy + 1e-12).reshape(-1)     # (N,)


def _dataclass_from_dict(cls, data: dict):
    """Build config dataclasses from saved JSON, ignoring keys unknown to this checkout."""
    known = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in known})


def _trace_stat_float(stats: dict, key: str, default: float = 0.0) -> float:
    val = stats.get(key, default)
    if torch.is_tensor(val):
        return float(val.detach().cpu())
    return float(val)


def _format_neus_trace_stats(stats: dict) -> str:
    if not stats.get("neus_bracket_enabled", False):
        return "  neus_xing off"
    count = int(_trace_stat_float(stats, "neus_bracket_count"))
    frac = _trace_stat_float(stats, "neus_bracket_frac")
    width_mean = _trace_stat_float(stats, "neus_bracket_width_mean")
    width_max = _trace_stat_float(stats, "neus_bracket_width_max")
    trace_name = stats.get("trace", "?")
    return f"  neus[{trace_name}] xing {count}({frac:.2f})  Δt {width_mean:.4g}/{width_max:.4g}"


def load_config_json(path: Path) -> Config:
    """Load a training Config from a saved run config.json."""
    data = json.loads(path.read_text())
    train_data = dict(data.get("train", {}))
    train_data["mvsdf_schedule"] = _dataclass_from_dict(
        MvsdfScheduleConfig,
        train_data.get("mvsdf_schedule", {}),
    )
    train_data["bundle"] = _dataclass_from_dict(
        BundleAdjustConfig,
        train_data.get("bundle", {}),
    )
    for key in ("feature_maps", "mvs_depth_dir"):
        if train_data.get(key) is not None:
            train_data[key] = Path(train_data[key])

    eval_data = dict(data.get("eval", {}))
    if eval_data.get("dtu_eval_dir") is not None:
        eval_data["dtu_eval_dir"] = Path(eval_data["dtu_eval_dir"])

    return Config(
        model=_dataclass_from_dict(ModelConfig, data.get("model", {})),
        trace=_dataclass_from_dict(TraceConfig, data.get("trace", {})),
        init=_dataclass_from_dict(InitConfig, data.get("init", {})),
        train=_dataclass_from_dict(TrainConfig, train_data),
        eval=_dataclass_from_dict(EvalConfig, eval_data),
        scene=Path(data.get("scene", SCENE)),
        out_dir=Path(data.get("out_dir", OUT_DIR)),
    )


# ---------- initialisations ----------

def fit_sphere_init(
    model_cfg: ModelConfig = None,
    init_cfg:  InitConfig  = None,
    scene:     Path        = SCENE,
) -> tuple[FTheta, float]:
    """Train f_θ ≈ ‖x‖ − r as a warm start.

    If init_cfg.radius is None, uses 1.1 × p80 of COLMAP point distances so the
    sphere encloses the scene while cameras remain outside.
    """
    model_cfg = model_cfg or ModelConfig()
    init_cfg  = init_cfg  or InitConfig()
    radius    = init_cfg.radius

    if radius is None:
        pts    = load_colmap_points(scene)
        r      = pts.norm(dim=-1)
        radius = 1.1 * r.quantile(0.60).item()
        print(f"  sphere init  auto radius={radius:.3f}  "
              f"(p60={r.quantile(0.60):.3f} p99={r.quantile(0.99):.3f} max={r.max():.3f})")
    bound  = radius * 1.5   # tighter box → more interior samples → better f(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f      = make_model(hidden=model_cfg.hidden, depth=model_cfg.depth,
                        group_size=model_cfg.group_size,
                        activation=model_cfg.activation,
                        input_encoding=model_cfg.input_encoding,
                        multires=model_cfg.multires,
                        architecture=getattr(model_cfg, "architecture", "cpl")).to(device)
    opt    = torch.optim.Adam(f.parameters(), lr=init_cfg.lr)
    for step in range(init_cfg.steps):
        x      = (2 * torch.rand(init_cfg.batch, 3, device=device) - 1) * bound
        target = x.norm(dim=-1) - radius
        loss   = F.mse_loss(f(x), target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 200 == 0:
            f0 = f.sdf(torch.zeros(1, 3, device=device)).item()
            print(f"  sphere init  step {step:5d}  mse {loss.item():.6f}  f(0)={f0:.4f} (target={-radius:.4f})")
    with torch.no_grad():
        f_origin = f.sdf(torch.zeros(1, 3, device=device)).item()
        if (scene / "meta_data.json").exists():
            frac_out = (f(load_camera_centers(scene).to(device)) > 0).float().mean().item()
            cams_str = f"  cams_outside={frac_out:.0%}"
        else:
            cams_str = "  cams_outside=n/a (blender)"
    print(f"  sphere init done  r={radius:.3f}  f(0)={f_origin:.4f}{cams_str}")
    return f, radius


def fit_hull_init(
    model_cfg: ModelConfig = None,
    init_cfg:  InitConfig  = None,
    scene:     Path        = SCENE,
    bound:     float       = 1.5,
    return_info: bool = False,
) -> FTheta | tuple[FTheta, np.ndarray]:
    """Warm-start f_θ by fitting to the visual hull SDF."""
    from .visual_hull import carve, fit_to_hull
    model_cfg = model_cfg or ModelConfig()
    init_cfg  = init_cfg  or InitConfig()
    print(f"  hull init: carving at res={init_cfg.hull_res} …")
    occ = carve(scene=scene, res=init_cfg.hull_res, bound=bound)
    print(f"  occupied voxels: {occ.sum()} / {occ.size}")
    depth_pts = None
    w_depth_surface = init_cfg.w_depth_surface
    if (scene / "transforms_train.json").exists() and w_depth_surface > 0:
        try:
            depth_pts = load_blender_gt_points(scene=scene)
        except (FileNotFoundError, ValueError) as e:
            print(f"  hull init: blender depth points unavailable ({e})")
    from .data import load_views, load_blender_views
    if (scene / "transforms_train.json").exists():
        _views = load_blender_views(scene=scene, split="train", down=1)
    else:
        _views = load_views(scene)
    cam_origins_np = _views["c2w"][:, :3, 3].numpy()

    is_blender = (scene / "transforms_train.json").exists()
    # MLP has no Lipschitz bound → large initial gradients with high-freq PE → need lower lr
    hull_lr = init_cfg.lr if getattr(model_cfg, "architecture", "cpl") == "cpl" else min(init_cfg.lr, 5e-4)
    f = fit_to_hull(occ, bound=bound, steps=init_cfg.steps, batch=init_cfg.batch,
                    lr=hull_lr, cfg=model_cfg,
                    depth_points=depth_pts, w_depth_surface=w_depth_surface,
                    cam_origins=cam_origins_np, w_cam_free=0.0 if is_blender else 1.0)
    print("  hull init done")
    if return_info:
        return f, occ
    return f


def fit_colmap_init(
    model_cfg: ModelConfig = None,
    init_cfg:  InitConfig  = None,
    scene:     Path        = SCENE,
    bound:     float       = 1.5,
) -> tuple[FTheta, np.ndarray]:
    """Warm-start f_theta from a COLMAP dense mesh (poisson.ply, fused.ply).

    Carves an occupancy grid by querying the mesh's signed distance at each
    voxel centre (negative = inside), then reuses fit_to_hull just like the
    silhouette path. Leaves fit_hull_init / carve / load_views untouched.

    Mesh is expected in NSVF-COLMAP (un-normalised) frame; we re-normalise it
    into the unit cube the model lives in via scene/bbox.txt.
    """
    from .visual_hull import fit_to_hull
    from .data import load_views
    import open3d as o3d

    model_cfg = model_cfg or ModelConfig()
    init_cfg  = init_cfg  or InitConfig()
    if init_cfg.init_mesh is None:
        raise ValueError("--init colmap requires --init-mesh /path/to/poisson.ply")
    mesh_path = Path(init_cfg.init_mesh)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"init mesh not found: {mesh_path}")

    print(f"  colmap init: loading {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_path} has no triangles — pass a mesh, not a point cloud")

    # un-normalised -> NSVF unit-cube frame (matches data loader's c2w transform)
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float32)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    bb_scale = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    verts_n = (np.asarray(mesh.vertices, dtype=np.float64) - center) / bb_scale
    mesh.vertices = o3d.utility.Vector3dVector(verts_n)
    print(f"  colmap init: bbox center={center}  scale={bb_scale:.4f}  "
          f"verts={len(verts_n):,}")

    # voxel-grid occupancy via signed distance to the closed Poisson mesh
    res = init_cfg.hull_res
    grid = np.linspace(-bound, bound, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    scene_rc = o3d.t.geometry.RaycastingScene()
    scene_rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    sd = scene_rc.compute_signed_distance(o3d.core.Tensor(pts)).numpy()
    occ = (sd < 0).reshape(res, res, res)
    print(f"  colmap init: mesh-only occupied {occ.sum()} / {occ.size}  "
          f"({100*occ.mean():.2f}%)")

    # If a sibling fused.ply exists, union its point-cloud occupancy in: this
    # recovers trees / vehicles / thin objects that Poisson trimmed away. Voxel
    # is "occupied" when any fused point lies within ~1 voxel of its centre.
    fused_path = mesh_path.parent / "fused.ply"
    if fused_path.exists():
        from scipy.spatial import cKDTree
        pcd = o3d.io.read_point_cloud(str(fused_path))
        pts_n = (np.asarray(pcd.points, dtype=np.float64) - center) / bb_scale
        voxel_size = 2.0 * bound / max(res - 1, 1)
        tree = cKDTree(pts_n)
        d, _ = tree.query(pts.astype(np.float64), k=1,
                          distance_upper_bound=voxel_size * 1.5)
        occ_pts = (d < voxel_size * 1.0).reshape(res, res, res)
        occ_union = occ | occ_pts
        added = int(occ_union.sum() - occ.sum())
        occ = occ_union
        print(f"  colmap init: +fused.ply ({len(pts_n):,} pts) -> {occ.sum()} / "
              f"{occ.size} ({100*occ.mean():.2f}%, +{added} voxels)")
    else:
        print(f"  colmap init: no sibling fused.ply at {fused_path} (mesh-only)")

    # camera origins (for fit_to_hull's free-space term — same as hull path)
    is_blender = (scene / "transforms_train.json").exists()
    _views = load_views(scene)
    cam_origins_np = _views["c2w"][:, :3, 3].numpy()
    hull_lr = init_cfg.lr if getattr(model_cfg, "architecture", "cpl") == "cpl" else min(init_cfg.lr, 5e-4)
    f = fit_to_hull(occ, bound=bound, steps=init_cfg.steps, batch=init_cfg.batch,
                    lr=hull_lr, cfg=model_cfg,
                    depth_points=None, w_depth_surface=0.0,
                    cam_origins=cam_origins_np,
                    w_cam_free=0.0 if is_blender else 1.0)
    print("  colmap init done")
    return f, occ


def _visual_hull_sample_bounds(occ: np.ndarray, bound: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a padded xyz AABB around occupied visual-hull voxels."""
    idx = np.argwhere(occ)
    if idx.size == 0:
        return None

    res = occ.shape[0]
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    lo_zyx = idx.min(axis=0)
    hi_zyx = idx.max(axis=0)

    lo = np.array([lin[lo_zyx[2]], lin[lo_zyx[1]], lin[lo_zyx[0]]], dtype=np.float32)
    hi = np.array([lin[hi_zyx[2]], lin[hi_zyx[1]], lin[hi_zyx[0]]], dtype=np.float32)

    voxel = 2 * bound / max(res - 1, 1)
    pad = max(4.0 * voxel, 0.05 * float((hi - lo).max()))
    lo = np.maximum(lo - pad, -bound).astype(np.float32)
    hi = np.minimum(hi + pad, bound).astype(np.float32)
    return lo, hi


# ---------- residual map (single-view overfit diagnostic) ----------

def _render_residual_map(
    f, sv_det: dict, view_idx: int,
    images, K_all, w2c_all, origins_all, alt_nn,
    H: int, W: int, H_d: int, W_d: int,
    n_alt: int, cos_thresh: float,
    trace_cfg, run_dir: Path, step: int,
) -> None:
    """Full-image residual map for single-view overfit diagnostic.

    For every pixel in the training view:
      - sphere-trace the ray
      - reproject the hit point to each alt camera, sample GT colour
      - compare with self-view colour at the projected pixel
      - save (H_d, W_d) images of mean |residual| and std across alt cameras
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sphere_tracing import trace_nograd
    from .loss import bilinear_sample

    device = next(f.parameters()).device
    N = sv_det["o"].shape[0]  # H_d * W_d

    # full-image trace
    chunk = 4096
    x_hits, ts, hits = [], [], []
    with torch.no_grad():
        for i in range(0, N, chunk):
            xh, t, hit = trace_nograd(f, sv_det["o"][i:i+chunk].to(device),
                                       sv_det["d"][i:i+chunk].to(device), trace_cfg)
            x_hits.append(xh); ts.append(t); hits.append(hit)
    x_theta = torch.cat(x_hits)
    hit     = torch.cat(hits)
    vi      = sv_det["vi"].to(device)   # all == view_idx

    # self-view colour at the reprojected pixel
    w2c_self = w2c_all[view_idx]
    xc_self  = x_theta @ w2c_self[:3, :3].T + w2c_self[:3, 3]
    uv_h     = xc_self @ K_all[view_idx].T
    uv_self  = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
    c_self   = bilinear_sample(images, vi, uv_self, H, W).float()   # (N, 3)

    # residuals across alt cameras
    alts = alt_nn[view_idx]   # (n_alt,)
    per_alt_res = []           # each (N, 3), NaN where invalid

    with torch.no_grad():
        for k in range(n_alt):
            ak   = alts[k].item()
            op   = origins_all[ak:ak+1].expand(N, -1)
            diff = x_theta - op
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dp   = diff / dist
            _, tp, hitp = trace_nograd(f, op, dp, trace_cfg)
            depth_ok = dist.squeeze(-1) <= tp + 0.1
            not_occl = hitp & depth_ok

            w2ca = w2c_all[ak]
            xca  = x_theta @ w2ca[:3, :3].T + w2ca[:3, 3]
            uvh  = xca @ K_all[ak].T
            uva  = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)
            in_frame = (xca[:, 2] > 0) & (uva[:, 0] >= 0) & (uva[:, 0] < W) \
                       & (uva[:, 1] >= 0) & (uva[:, 1] < H)
            valid = hit & not_occl & in_frame

            res = torch.full((N, 3), float("nan"), device=device)
            if valid.any():
                ak_t  = torch.full((N,), ak, dtype=torch.long, device=device)
                c_alt = bilinear_sample(images, ak_t, uva, H, W).float()
                res[valid] = (c_self - c_alt).abs()[valid]
            per_alt_res.append(res)

    stack = torch.stack(per_alt_res, dim=0)   # (n_alt, N, 3)
    # nanmean / nanstd over alt cameras
    res_mean = stack.nanmean(dim=0).mean(dim=-1).cpu()  # (N,)
    count    = (~stack[:, :, 0].isnan()).sum(dim=0).float().cpu()
    res_std  = torch.zeros(N)
    valid_px = count >= 2
    if valid_px.any():
        res_std[valid_px] = stack[:, valid_px, :].nanmean(dim=-1).std(dim=0).cpu()[valid_px]

    # reshape to image grids
    mean_img = res_mean.reshape(H_d, W_d).numpy()
    std_img  = res_std.reshape(H_d, W_d).numpy()
    hit_img  = hit.cpu().reshape(H_d, W_d).numpy()
    gt_img   = sv_det["gt"].float().cpu().reshape(H_d, W_d, 3).numpy()

    # mask misses as grey
    mean_img[~hit_img] = float("nan")
    std_img[~hit_img]  = float("nan")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(gt_img.clip(0, 1)); axes[0].axis("off")
    axes[0].set_title("GT (training view)")
    im1 = axes[1].imshow(mean_img, cmap="hot", vmin=0, vmax=0.3)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].axis("off"); axes[1].set_title("mean |residual| across alt cams")
    im2 = axes[2].imshow(std_img, cmap="plasma", vmin=0, vmax=0.2)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].axis("off"); axes[2].set_title("std |residual| across alt cams")
    im3 = axes[3].imshow(count.reshape(H_d, W_d).numpy(), cmap="viridis")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    axes[3].axis("off"); axes[3].set_title("n valid alt cams per pixel")
    fig.suptitle(f"step {step}  view {view_idx}", fontsize=11)
    fig.tight_layout()
    out = run_dir / "diag" / f"residual_map_{step:05d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [residual_map] → {out.name}  "
          f"mean={res_mean[hit.cpu()].mean():.4f}  "
          f"std={res_std[valid_px].mean():.4f}")


# ---------- periodic render ----------

def _render_poses(f, views, step: int, run_dir: Path, device: str,
                  res: int = 400, trace_cfg: TraceConfig | None = None) -> None:
    """Sphere-trace 4 training views with Phong shading → PNG strip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sphere_tracing import trace_nograd

    V      = views["c2w"].shape[0]
    H_full = views["H"]; W_full = views["W"]
    ids    = [int(round(i * (V - 1) / 2)) for i in range(3)]
    # 4th view: camera most opposite to view 0 (back of the object)
    cam_positions = views["c2w"][:, :3, 3].numpy()  # (V, 3)
    dir0 = cam_positions[ids[0]] / (np.linalg.norm(cam_positions[ids[0]]) + 1e-6)
    dots = (cam_positions / (np.linalg.norm(cam_positions, axis=-1, keepdims=True) + 1e-6)) @ dir0
    back_id = int(np.argmin(dots))
    ids.append(back_id)
    down   = max(1, H_full // res)
    H, W   = H_full // down, W_full // down

    light = np.array([0.577, 0.577, 0.577], dtype=np.float32)
    base  = np.array([0.72, 0.72, 0.85],    dtype=np.float32)

    imgs_phong, imgs_color, imgs_hit = [], [], []
    for vi in ids:
        K   = views["K"][vi].numpy()
        c2w = views["c2w"][vi].numpy()
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xs_f = (xs + 0.5) * down - 0.5; ys_f = (ys + 0.5) * down - 0.5
        d_cam = np.stack([(xs_f - K[0,2]) / K[0,0],
                          (ys_f - K[1,2]) / K[1,1],
                          np.ones_like(xs_f)], axis=-1)
        d_w = d_cam @ c2w[:3, :3].T
        d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
        o_t = torch.from_numpy(np.broadcast_to(c2w[:3, 3], d_w.shape).copy().reshape(-1, 3)).float().to(device)
        d_t = torch.from_numpy(d_w.reshape(-1, 3)).float().to(device)

        # Chunk to bound peak memory on large images (TnT etc.); no-grad and
        # per-ray independent, so chunking is bit-identical to one big call.
        x_hit_parts, hit_parts = [], []
        for i in range(0, o_t.shape[0], 65536):
            xh, _, hh = trace_nograd(f, o_t[i:i + 65536], d_t[i:i + 65536],
                                     *((trace_cfg,) if trace_cfg is not None else ()))
            x_hit_parts.append(xh); hit_parts.append(hh)
        x_hit = torch.cat(x_hit_parts, dim=0)
        hit   = torch.cat(hit_parts,   dim=0)
        del x_hit_parts, hit_parts
        torch.cuda.empty_cache()
        # Chunk normal computation to avoid holding the full graph for H*W rays
        xr_all = x_hit.detach()
        grads = []
        for i in range(0, xr_all.shape[0], 4096):
            xr_chunk = xr_all[i:i + 4096].requires_grad_(True)
            with torch.enable_grad():
                grads.append(torch.autograd.grad(f(xr_chunk).sum(), xr_chunk)[0].detach())
        n = torch.cat(grads, dim=0)
        n = (n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)).cpu().numpy()

        diffuse = np.clip((n * light).sum(-1, keepdims=True), 0, 1)
        shaded  = (0.35 + 0.65 * diffuse) * base
        hit_np  = hit.cpu().numpy().reshape(H, W, 1)
        imgs_phong.append(np.where(hit_np, shaded.reshape(H, W, 3), 1.0))

        # colour render: GT image downsampled to render resolution, masked by hit
        gt = views["images"][vi].numpy()   # (H_full, W_full, 3) float32 [0,1]
        if down > 1:
            from PIL import Image as _PIL
            gt = np.array(_PIL.fromarray((gt * 255).astype(np.uint8)).resize(
                (W, H), _PIL.BILINEAR)).astype(np.float32) / 255.0
        imgs_color.append(np.where(hit_np, gt, 1.0))

        # hit-map: green=hit, red=miss where GT is foreground (hole), white=bg
        gt_mask = views["masks"][vi].numpy() if "masks" in views else (gt.sum(-1) < 2.95)
        if down > 1:
            from PIL import Image as _PIL
            gt_mask = np.array(_PIL.fromarray(gt_mask.astype(np.uint8) * 255).resize(
                (W, H), _PIL.NEAREST)) > 127
        fg = gt_mask.reshape(H, W)
        hit2 = hit_np[:, :, 0]
        hmap = np.ones((H, W, 3), dtype=np.float32)          # white = bg
        hmap[fg &  hit2.astype(bool)] = [0.4, 0.85, 0.4]     # green = correct hit
        hmap[fg & ~hit2.astype(bool)] = [0.9, 0.2,  0.2]     # red   = missed fg (hole)
        imgs_hit.append(hmap)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    labels = ["view A", "view B", "view C", f"back (v{back_id})"]
    for ax, img, lbl in zip(axes[0], imgs_phong, labels):
        ax.imshow(img.clip(0, 1)); ax.axis("off"); ax.set_title(lbl, fontsize=9)
    for ax, img in zip(axes[1], imgs_color):
        ax.imshow(img.clip(0, 1)); ax.axis("off")
    for ax, img in zip(axes[2], imgs_hit):
        ax.imshow(img.clip(0, 1)); ax.axis("off")
    axes[0][1].set_title("Phong", fontsize=10)
    axes[1][1].set_title("Colour (GT × hit)", fontsize=10)
    axes[2][1].set_title("Hit map (green=hit  red=hole)", fontsize=10)
    fig.suptitle(f"step {step}", fontsize=11)
    fig.tight_layout()
    out = run_dir / "render" / f"render_{step:05d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [render] → {out.name}")


def _save_view_diag_png(panels: list, vi: int, step: int,
                        run_dir: Path) -> None:
    """All per-pixel diagnostics for one view on a SINGLE png.

    panels : list of dicts {arr, title, cmap, vmin, vmax, label}.
    Written as diag_view{vi}_step_{step}.png. Never raises.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncol = len(panels)
        fig, axes = plt.subplots(1, ncol, figsize=(7 * ncol, 7),
                                 squeeze=False)
        for ax, p in zip(axes[0], panels):
            arr = p["arr"]
            finite = arr[np.isfinite(arr)]
            mean_v = float(finite.mean()) if finite.size else float("nan")
            med_v = float(np.median(finite)) if finite.size else float("nan")
            zero_frac = (float((finite == 0).mean())
                         if finite.size else float("nan"))
            im = ax.imshow(arr, cmap=p["cmap"], vmin=p["vmin"], vmax=p["vmax"])
            ax.axis("off")
            ax.set_title(f"{p['title']}\nmean={mean_v:.3g} "
                         f"med={med_v:.3g} =0={zero_frac:.1%}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=.046, shrink=.85,
                         label=p["label"])
        fig.suptitle(f"view {vi}  per-pixel diagnostics  step {step}",
                     fontsize=12)
        fig.tight_layout()
        out = run_dir / "diag" / f"diag_view{vi}_step_{step:06d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [diag] → {out.name}", flush=True)
    except Exception as exc:
        print(f"  [diag] save failed (non-fatal): {exc}", flush=True)


def _dump_mc_normal_maps(f, views, view_ids: list[int], step: int,
                         run_dir: Path, device: str,
                         mc_res: int = 128, bound: float = 1.0,
                         down: int = 1, trace_cfg=None,
                         train_cfg=None, alt_nn=None,
                         coverage_res: int = 500) -> None:
    """Marching-cubes-mesh normal maps (à la render_paper_marching) for the
    given camera views, all extracted from the *same* checkpoint. When training
    config is supplied, also adds K_valid heatmaps for the NCC visibility gate.

    Uses a low mc_res grid and downsampled cameras so it is cheap enough to
    run at the 10k-step checkpoint cadence. Mesh vertex normals (the
    render_paper_marching default), not analytic ∇f. Never raises — a dump
    failure must not kill training.
    """
    try:
        import sys as _sys
        _root = str(Path(__file__).resolve().parent.parent)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        import mcubes  # noqa: F401  (presence check)
        from render_paper_marching import (_mesh_from_volume, _make_intersector,
                                           render_normals_only)

        V   = views["c2w"].shape[0]
        ids = [vi for vi in view_ids if 0 <= vi < V]
        if not ids:
            print(f"  [normals] no valid view in {view_ids} (V={V}); skipped", flush=True)
            return

        # --- SDF grid → mesh (low res for speed) ---
        vox  = torch.linspace(-bound, bound, mc_res)
        grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"),
                           dim=-1).reshape(-1, 3)
        vals = []
        with torch.no_grad():
            for i in range(0, grid.shape[0], 65536):
                vals.append(f(grid[i:i + 65536].to(device)).detach().cpu())
        vol = torch.cat(vals).reshape(mc_res, mc_res, mc_res).numpy()
        if vol.min() > 0 or vol.max() < 0:
            print(f"  [normals] no zero crossing "
                  f"(range=[{vol.min():.4f},{vol.max():.4f}]); skipped", flush=True)
            return
        mesh        = _mesh_from_volume(vol, bound, mc_res, 0.0)
        intersector = _make_intersector(mesh)

        # --- downsampled camera intrinsics for speed ---
        H_full, W_full = views["H"], views["W"]
        d        = max(1, down)
        H_d, W_d = H_full // d, W_full // d

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        K_all = views["K"].float().to(device)
        c2w_all = views["c2w"].float().to(device)
        w2c_all = torch.linalg.inv(c2w_all)
        imgs_t = views["images"].float().to(device)
        masks_t = (views["masks"].bool().to(device) if "masks" in views
                   else (imgs_t.sum(-1) < 2.95))
        cams = views["c2w"][:, :3, 3].numpy()
        origins = c2w_all[:, :3, 3]

        show_coverage = train_cfg is not None and trace_cfg is not None
        n_rows = 2 if show_coverage else 1
        fig, axes = plt.subplots(n_rows, len(ids), figsize=(5 * len(ids), 5 * n_rows),
                                 squeeze=False)
        for col, vi in enumerate(ids):
            ax = axes[0, col]
            K = views["K"][vi].numpy().copy()
            K[0, 0] /= d; K[1, 1] /= d
            K[0, 2] = (K[0, 2] + 0.5) / d - 0.5
            K[1, 2] = (K[1, 2] + 0.5) / d - 0.5
            img = render_normals_only(mesh, intersector,
                                      views["c2w"][vi].numpy(), K,
                                      H_d, W_d, 1)
            ax.imshow(np.clip(img, 0, 1)); ax.axis("off")
            ax.set_title(f"view {vi} normals", fontsize=10)

            if not show_coverage:
                continue

            # K_valid(p) = #{alt views: in-frame, not occluded, cos-ok, fg masks,
            # patch valid/textured}. This mirrors the training NCC gates but
            # intentionally stops before the ZNCC > ncc_min keep threshold.
            try:
                cov_down = max(1, H_full // max(coverage_res, 1))
                Hc, Wc = H_full // cov_down, W_full // cov_down
                K0 = views["K"][vi].numpy()
                ys, xs = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing="ij")
                xf = (xs + .5) * cov_down - .5
                yf = (ys + .5) * cov_down - .5
                d_cam = np.stack([(xf - K0[0, 2]) / K0[0, 0],
                                  (yf - K0[1, 2]) / K0[1, 1],
                                  np.ones_like(xf)], -1)
                c2w = views["c2w"][vi].numpy()
                d_w = d_cam @ c2w[:3, :3].T
                d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
                o_t = torch.from_numpy(np.broadcast_to(c2w[:3, 3], d_w.shape)
                                       .copy().reshape(-1, 3)).float().to(device)
                d_t = torch.from_numpy(d_w.reshape(-1, 3)).float().to(device)

                with torch.no_grad():
                    x_hit, _, hit, f_pre = trace_nograd(
                        f, o_t, d_t, trace_cfg, return_diag=True)
                grads = []
                for i in range(0, x_hit.shape[0], 4096):
                    xc = x_hit[i:i + 4096].detach().requires_grad_(True)
                    with torch.enable_grad():
                        grads.append(torch.autograd.grad(f(xc).sum(), xc)[0].detach())
                nrm = torch.cat(grads)
                nrm = F.normalize(nrm, dim=-1)

                if alt_nn is not None:
                    alts = [int(a) for a in alt_nn[vi].tolist()]
                else:
                    others = [j for j in range(V) if j != vi]
                    order = np.argsort(np.linalg.norm(cams[others] - cams[vi], axis=1))
                    alts = [others[k] for k in order[:getattr(train_cfg, "n_alt", 6)]]

                fx = np.clip(np.rint(xf).astype(np.int64), 0, W_full - 1)
                fy = np.clip(np.rint(yf).astype(np.int64), 0, H_full - 1)
                fg_self = masks_t[vi, torch.from_numpy(fy.reshape(-1)).to(device),
                                  torch.from_numpy(fx.reshape(-1)).to(device)]
                valid_count = torch.zeros(x_hit.shape[0], device=device)
                valid_count_thr = torch.zeros(x_hit.shape[0], device=device)
                KZ_TAU = 0.3   # stricter K_valid: also require ZNCC > τ
                cos_thr = getattr(train_cfg, "cos_thresh", 0.1)
                occ_md = getattr(train_cfg, "occ_mode", "pinhole")
                ncc_P = getattr(train_cfg, "ncc_patch", 5)
                ncc_hp = getattr(train_cfg, "ncc_half_pix", 2.0)
                ncc_clr = getattr(train_cfg, "ncc_color", "gray")
                ncc_ga = getattr(train_cfg, "ncc_grad_alpha", 0.0)

                for s in range(0, x_hit.shape[0], 8192):
                    sl = slice(s, s + 8192)
                    xs_t, ns_t = x_hit[sl], nrm[sl]
                    h_t, fss = hit[sl], fg_self[sl]
                    for ak in alts:
                        op = origins[ak]
                        dirv = op.unsqueeze(0) - xs_t
                        dist = dirv.norm(dim=-1).clamp(min=1e-6)
                        dp = dirv / dist.unsqueeze(-1)
                        Rb = w2c_all[ak, :3, :3]; tb = w2c_all[ak, :3, 3]
                        xc = xs_t @ Rb.T + tb
                        ph = xc @ K_all[ak].T
                        uv = ph[:, :2] / ph[:, 2:3].clamp(min=1e-6)
                        in_fr = ((xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W_full)
                                 & (uv[:, 1] >= 0) & (uv[:, 1] < H_full))
                        cos_ok = (ns_t * dp).sum(-1).abs() > cos_thr
                        uc = uv.long()
                        fg_alt = masks_t[ak,
                                         uc[:, 1].clamp(0, H_full - 1),
                                         uc[:, 0].clamp(0, W_full - 1)]
                        if occ_md == "from_hit":
                            _, tp, hp = trace_nograd(f, xs_t + 1e-2 * dp, dp, trace_cfg)
                            not_occl = (~hp) | (tp > dist - 0.1)
                        else:
                            _, tp, hp = trace_nograd(f, op.unsqueeze(0).expand_as(xs_t),
                                                     -dp, trace_cfg)
                            not_occl = hp & (dist <= tp + 0.1)
                        gate = h_t & in_fr & not_occl & cos_ok & fg_alt & fss
                        if not gate.any():
                            continue
                        z = _zncc_per_ray(imgs_t, xs_t, ns_t, vi, ak, K_all, w2c_all,
                                          H_full, W_full, ncc_P, ncc_hp,
                                          ncc_clr, ncc_ga)
                        zfin = gate & torch.isfinite(z)
                        valid_count[sl] += zfin.float()
                        valid_count_thr[sl] += (zfin & (z > KZ_TAU)).float()

                cov = valid_count.cpu().numpy().reshape(Hc, Wc)
                fg_np = fg_self.cpu().numpy().reshape(Hc, Wc).astype(bool)
                hit_np = hit.cpu().numpy().reshape(Hc, Wc).astype(bool)
                cov = np.where(fg_np & hit_np, cov, np.nan)

                # stricter coverage: also require ZNCC > KZ_TAU
                cov_thr = valid_count_thr.cpu().numpy().reshape(Hc, Wc)
                cov_thr = np.where(fg_np & hit_np, cov_thr, np.nan)

                # trace-residual / Newton convergence map: |f(x_hit)| / eps
                eps_v = float(getattr(trace_cfg, "eps", 1e-3))
                fr = []
                with torch.no_grad():
                    for i in range(0, x_hit.shape[0], 65536):
                        fr.append(f(x_hit[i:i + 65536]).abs().detach().cpu())
                fres = torch.cat(fr).numpy().reshape(Hc, Wc) / max(eps_v, 1e-9)
                fres = np.where(fg_np & hit_np, fres, np.nan)

                # pre-Newton residual: |f| at the hit *before* the Newton loop.
                # If Newton is the cause, fres_pre ≈ 0–1 (clean) while the
                # post-Newton fres ≈ 2 (noisy).
                fres_pre = (f_pre.cpu().numpy().reshape(Hc, Wc)
                            / max(eps_v, 1e-9))
                fres_pre = np.where(fg_np & hit_np, fres_pre, np.nan)
                _fin_pre, _fin_post = (np.isfinite(fres_pre).any(),
                                       np.isfinite(fres).any())
                print(f"[diag v{vi} step {step}] |f|/eps  "
                      f"pre-Newton mean={np.nanmean(fres_pre) if _fin_pre else float('nan'):.2f} "
                      f"med={np.nanmedian(fres_pre) if _fin_pre else float('nan'):.2f}  |  "
                      f"post-Newton mean={np.nanmean(fres) if _fin_post else float('nan'):.2f} "
                      f"med={np.nanmedian(fres) if _fin_post else float('nan'):.2f}",
                      flush=True)

                # all per-pixel diagnostics for this view on ONE png
                nA = max(len(alts), 1)
                _panels = [
                    dict(arr=cov, title=f"K_valid / {nA}",
                         cmap="viridis", vmin=0, vmax=nA,
                         label="valid alt-view count"),
                    dict(arr=cov_thr, title=f"K_valid / {nA} (ZNCC>{KZ_TAU})",
                         cmap="viridis", vmin=0, vmax=nA,
                         label="valid alt-view count"),
                    dict(arr=fres_pre, title="|f|/eps pre-Newton",
                         cmap="magma", vmin=0, vmax=5.0,
                         label="|f(x_hit)| / eps  (before Newton)"),
                    dict(arr=fres, title="|f|/eps post-Newton (trace residual)",
                         cmap="magma", vmin=0, vmax=5.0,
                         label="|f(x_hit)| / eps  (after Newton)"),
                ]
                # image-gradient map (only if the grad-ZNCC term is active)
                if ncc_ga > 0:
                    gimg = imgs_t[vi].cpu().numpy()             # (Hf,Wf,3)
                    if ncc_clr == "gray":
                        gimg = (gimg * np.array([0.299, 0.587, 0.114],
                                                np.float32)).sum(-1)
                    else:
                        gimg = gimg.mean(-1)
                    gy_i, gx_i = np.gradient(gimg)
                    gmag = np.sqrt(gx_i ** 2 + gy_i ** 2)       # (Hf,Wf)
                    gm = gmag[fy, fx].reshape(Hc, Wc)
                    gm = np.where(fg_np & hit_np, gm, np.nan)
                    gvmax = float(np.nanpercentile(gm, 98)) if np.isfinite(
                        gm).any() else 1.0
                    _panels.append(dict(
                        arr=gm, title=f"|∇I| ({ncc_clr})  α={ncc_ga:.2f}",
                        cmap="inferno", vmin=0, vmax=max(gvmax, 1e-6),
                        label="image gradient magnitude"))
                _save_view_diag_png(_panels, vi, step, run_dir)

                ax_cov = axes[1, col]
                im_cov = ax_cov.imshow(cov, cmap="viridis", vmin=0, vmax=max(len(alts), 1))
                ax_cov.axis("off")
                ax_cov.set_title(f"view {vi} K_valid / {len(alts)}", fontsize=10)
                fig.colorbar(im_cov, ax=ax_cov, fraction=.046, shrink=.8)
            except Exception as cov_exc:
                ax_cov = axes[1, col]
                ax_cov.text(0.5, 0.5, f"K_valid failed\n{cov_exc}",
                            ha="center", va="center", fontsize=8)
                ax_cov.axis("off")

        fig.suptitle(f"MC normals + NCC valid-view coverage — step {step}  (mc_res={mc_res})",
                     fontsize=11)
        fig.tight_layout()
        out = run_dir / "render" / f"normals_step_{step:06d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  [normals] → {out.name}", flush=True)
    except Exception as exc:
        print(f"  [normals] dump failed (non-fatal): {exc}", flush=True)


@torch.no_grad()
def _zncc_per_ray(images, x3d, normals, vi_a: int, vi_b: int,
                  K_all, w2c_all, H: int, W: int,
                  patch: int = 5, half_pix: float = 2.0,
                  ncc_color: str = "gray", ncc_grad_alpha: float = 0.0):
    """Per-ray ZNCC vs one alt view, aligned (NaN where invalid/untextured).

    Mirrors pmvs_ncc_loss's oriented-tangent-patch math but keeps the (B,)
    layout so it maps back to pixels — the actual training NCC signal.
    """
    import torch.nn.functional as F
    B = x3d.shape[0]
    P = patch
    n = F.normalize(normals, dim=-1)
    up = n.new_zeros(B, 3); up[:, 1] = 1.0
    sw = n[:, 1].abs() > 0.9
    up[sw, 1] = 0.0; up[sw, 0] = 1.0
    t1 = F.normalize(torch.cross(n, up, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    Ra = w2c_all[vi_a, :3, :3]; ta = w2c_all[vi_a, :3, 3]
    xc_a = x3d @ Ra.T + ta
    z_ref = xc_a[:, 2].clamp(min=1e-3)
    step = (2.0 * half_pix / max(P - 1, 1)) * z_ref / K_all[vi_a, 0, 0]
    offs = torch.linspace(-(P - 1) / 2, (P - 1) / 2, P, device=x3d.device)
    oi, oj = torch.meshgrid(offs, offs, indexing="ij")
    oi = oi.reshape(-1); oj = oj.reshape(-1)
    pts = (x3d.unsqueeze(1) + step[:, None, None]
           * (oi[None, :, None] * t1.unsqueeze(1)
              + oj[None, :, None] * t2.unsqueeze(1)))            # (B,P²,3)

    def proj(vi):
        R = w2c_all[vi, :3, :3]; tv = w2c_all[vi, :3, 3]
        xc = pts @ R.T + tv
        ph = xc @ K_all[vi].T
        uv = ph[:, :, :2] / ph[:, :, 2:3].clamp(min=1e-6)
        return uv, xc[:, :, 2]

    uva, za = proj(vi_a); uvb, zb = proj(vi_b)
    okv = (lambda uv, z: (z > 0).all(1)
           & (uv[:, :, 0] >= 0).all(1) & (uv[:, :, 0] < W).all(1)
           & (uv[:, :, 1] >= 0).all(1) & (uv[:, :, 1] < H).all(1))
    valid = okv(uva, za) & okv(uvb, zb)

    def samp(uv, vi):
        u = uv[:, :, 0].clamp(0, W - 1); v = uv[:, :, 1].clamp(0, H - 1)
        u0 = u.long(); u1 = (u0 + 1).clamp(max=W - 1)
        v0 = v.long(); v1 = (v0 + 1).clamp(max=H - 1)
        wu = (u - u0.float()).unsqueeze(-1); wv = (v - v0.float()).unsqueeze(-1)
        c00 = images[vi, v0, u0].float(); c10 = images[vi, v1, u0].float()
        c01 = images[vi, v0, u1].float(); c11 = images[vi, v1, u1].float()
        return (c00 * (1 - wu) * (1 - wv) + c01 * wu * (1 - wv)
                + c10 * (1 - wu) * wv + c11 * wu * wv)
    pa = samp(uva, vi_a); pb = samp(uvb, vi_b)
    if ncc_color == "gray":
        _lw = pa.new_tensor([0.299, 0.587, 0.114])
        pa = (pa * _lw).sum(-1, keepdim=True)
        pb = (pb * _lw).sum(-1, keepdim=True)
    raw_a, raw_b = pa, pb
    pa = pa - pa.mean(1, keepdim=True); pb = pb - pb.mean(1, keepdim=True)
    sa = pa.norm(dim=1); sb = pb.norm(dim=1)
    tex = (sa > 1e-4).all(1) & (sb > 1e-4).all(1)
    pa = pa / sa.unsqueeze(1).clamp(min=1e-6)
    pb = pb / sb.unsqueeze(1).clamp(min=1e-6)
    zncc = (pa * pb).sum(1).mean(1).clamp(-1, 1)
    if ncc_grad_alpha > 0.0:
        def _gm(p):
            p2 = p.reshape(p.shape[0], P, P, -1)
            gx = torch.zeros_like(p2); gy = torch.zeros_like(p2)
            gx[:, :, 1:-1, :] = 0.5 * (p2[:, :, 2:, :] - p2[:, :, :-2, :])
            gy[:, 1:-1, :, :] = 0.5 * (p2[:, 2:, :, :] - p2[:, :-2, :, :])
            return torch.sqrt(gx * gx + gy * gy + 1e-12).reshape(
                p.shape[0], P * P, -1)
        ga = _gm(raw_a); gb = _gm(raw_b)
        ga = ga - ga.mean(1, keepdim=True); gb = gb - gb.mean(1, keepdim=True)
        ga = ga / ga.norm(dim=1, keepdim=True).clamp(min=1e-6)
        gb = gb / gb.norm(dim=1, keepdim=True).clamp(min=1e-6)
        zncc_g = (ga * gb).sum(1).mean(1).clamp(-1, 1)
        zncc = ((1.0 - ncc_grad_alpha) * zncc
                + ncc_grad_alpha * zncc_g).clamp(-1, 1)
    out = torch.full((B,), float("nan"), device=x3d.device)
    good = valid & tex
    out[good] = zncc[good]
    return out                                                   # (B,) NaN=invalid


@torch.no_grad()
def _trace_diag(f, o, d, cfg):
    """Instrumented sphere-trace (mirrors trace_nograd) — per-ray numerics:
    iters_used, min|f| reached, final|f|, final t, t_far, escaped, stalled,
    sign-flips. Lets you read *why* a hole pixel didn't converge.
    """
    from .sphere_tracing import ray_sphere_exit
    B = o.shape[0]
    t_far = (ray_sphere_exit(o, d, cfg.bsphere_radius) if cfg.bsphere_radius > 0
             else torch.full((B,), cfg.t_far, device=o.device))
    t = torch.zeros(B, device=o.device)
    conv = torch.zeros(B, dtype=torch.bool, device=o.device)
    iters_used = torch.full((B,), cfg.iters, dtype=torch.float32, device=o.device)
    min_abs = torch.full((B,), float("inf"), device=o.device)
    nflip = torch.zeros(B, device=o.device)
    prev_sign = torch.zeros(B, device=o.device)
    for i in range(cfg.iters):
        escaped = t >= t_far
        sdf = f(o + t.unsqueeze(-1) * d)
        a = sdf.abs()
        min_abs = torch.minimum(min_abs, a)
        sg = torch.sign(sdf)
        flip = (prev_sign != 0) & (sg != prev_sign) & ~conv & ~escaped
        nflip += flip.float()
        prev_sign = sg
        just = (~conv) & (a < cfg.eps)
        iters_used = torch.where(just, torch.full_like(iters_used, i), iters_used)
        conv = conv | (a < cfg.eps)
        t = t + torch.where(conv | escaped, torch.zeros_like(t), sdf)
    sdf = f(o + t.unsqueeze(-1) * d)
    conv = conv | (sdf.abs() < cfg.eps)
    escaped = (~conv) & (t >= t_far)
    stalled = (~conv) & (~escaped)            # ran out of iters, still inside
    return dict(iters=iters_used, min_abs=min_abs, final_abs=sdf.abs(),
                t=t, t_far=t_far, conv=conv, escaped=escaped,
                stalled=stalled, nflip=nflip)


def _debug_views(f, views, view_ids: list[int], step: int, run_dir: Path,
                 device: str, trace_cfg, res: int = 500,
                 zoom_json: str = "", train_cfg=None, alt_nn=None) -> None:
    """Thorough per-view diagnostics for *specific* views (opt-in).

    Per view, 8 panels — GT×hit | hit-map | min_t f (tracer-independent) |
    |f(x_end)| | normals | photo-L1 vs nearest cam | cause(blue=tracer/red=
    absent) — plus a scalar row appended to debug/viewXX.csv.

    zoom_json: optional path to {"<view>": {"nose":[x0,y0,x1,y1], ...}} in
    full-res pixel coords → extra debug/zoom_<step>.png cropped on each region.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv, json
    from .sphere_tracing import trace_nograd

    zoom = {}
    if zoom_json:
        try:
            zoom = json.loads(Path(zoom_json).read_text())
        except Exception as e:
            print(f"  [debug] zoom json ignored ({e})")
    zoom_rows = []  # (view, region, [(crop, title, cmap), ...])

    dbg = run_dir / "debug"; dbg.mkdir(exist_ok=True)
    V = views["c2w"].shape[0]
    H_full, W_full = views["H"], views["W"]
    down = max(1, H_full // res)
    H, W = H_full // down, W_full // down
    cams = views["c2w"][:, :3, 3].numpy()
    light = np.array([0.577, 0.577, 0.577], np.float32)

    # full-res tensors for the actual ZNCC (training NCC signal)
    K_all = views["K"].float().to(device)
    c2w_all = views["c2w"].float().to(device)
    w2c_all = torch.linalg.inv(c2w_all)
    imgs_t = views["images"].float().to(device)            # (V,Hf,Wf,3)
    masks_t = (views["masks"].bool().to(device) if "masks" in views
               else (imgs_t.sum(-1) < 2.95))               # (V,Hf,Wf)
    n_alt = getattr(train_cfg, "n_alt", 6) if train_cfg else 6
    ncc_P = getattr(train_cfg, "ncc_patch", 5) if train_cfg else 5
    ncc_hp = getattr(train_cfg, "ncc_half_pix", 2.0) if train_cfg else 2.0
    ncc_min = getattr(train_cfg, "ncc_min", 0.0) if train_cfg else 0.0
    ncc_clr = getattr(train_cfg, "ncc_color", "gray") if train_cfg else "gray"
    ncc_ga = getattr(train_cfg, "ncc_grad_alpha", 0.0) if train_cfg else 0.0
    # Normal-branch patch (sentinel <0 → share the position branch's).
    _nnp = getattr(train_cfg, "ncc_normal_patch", -1) if train_cfg else -1
    _nnh = getattr(train_cfg, "ncc_normal_half_pix", -1.0) if train_cfg else -1.0
    ncc_nP = _nnp if _nnp > 0 else ncc_P
    ncc_nhp = _nnh if _nnh > 0 else ncc_hp
    ncc_n_split = (ncc_nP != ncc_P) or (ncc_nhp != ncc_hp)

    n_panel = 10 if ncc_n_split else 9
    fig, axes = plt.subplots(len(view_ids), n_panel,
                             figsize=(4 * n_panel, 4 * len(view_ids)))
    if len(view_ids) == 1:
        axes = axes[None, :]

    for row, vi in enumerate(view_ids):
        if vi < 0 or vi >= V:
            continue
        K = views["K"][vi].numpy(); c2w = views["c2w"][vi].numpy()
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xf = (xs + .5) * down - .5; yf = (ys + .5) * down - .5
        d_cam = np.stack([(xf - K[0, 2]) / K[0, 0],
                          (yf - K[1, 2]) / K[1, 1], np.ones_like(xf)], -1)
        d_w = d_cam @ c2w[:3, :3].T
        d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
        o_t = torch.from_numpy(np.broadcast_to(c2w[:3, 3], d_w.shape)
                               .copy().reshape(-1, 3)).float().to(device)
        d_t = torch.from_numpy(d_w.reshape(-1, 3)).float().to(device)

        with torch.no_grad():
            x_hit, t_hit, hit = trace_nograd(f, o_t, d_t)
            tdiag = _trace_diag(f, o_t, d_t, trace_cfg)   # per-ray ST numerics
            f_end = f(x_hit).abs().cpu().numpy().reshape(H, W)
            # tracer-independent min_t f over a stratified sweep
            ts = torch.linspace(0.05, trace_cfg.t_far, 64, device=device)
            sdf_min = torch.full((o_t.shape[0],), float("inf"), device=device)
            for tt in ts:
                sdf_min = torch.minimum(sdf_min, f(o_t + tt * d_t))
            sdf_min = sdf_min.cpu().numpy().reshape(H, W)
        # normals (chunked)
        xr = x_hit.detach(); g = []
        for i in range(0, xr.shape[0], 4096):
            xc = xr[i:i + 4096].requires_grad_(True)
            with torch.enable_grad():
                g.append(torch.autograd.grad(f(xc).sum(), xc)[0].detach())
        n = torch.cat(g); n = (n / n.norm(dim=-1, keepdim=True)
                               .clamp(min=1e-6)).cpu().numpy()
        hit_np = hit.cpu().numpy().reshape(H, W)

        gt = views["images"][vi].numpy()
        gm = (views["masks"][vi].numpy() if "masks" in views
              else gt.sum(-1) < 2.95)
        if down > 1:
            from PIL import Image as _PIL
            gt = np.array(_PIL.fromarray((gt * 255).astype(np.uint8))
                          .resize((W, H), _PIL.BILINEAR)).astype(np.float32) / 255
            gm = np.array(_PIL.fromarray(gm.astype(np.uint8) * 255)
                          .resize((W, H), _PIL.NEAREST)) > 127
        fg = gm.reshape(H, W)

        # ACTUAL training NCC — same gates as photo_loss:
        #   hit & in_frame & not_occl & cos_ok & fg_alt & fg_self,
        #   then pmvs valid+textured+keep(zncc>ncc_min). alts = alt_nn[vi].
        if alt_nn is not None:
            alts = [int(a) for a in alt_nn[vi].tolist()]
        else:                                   # fallback: nearest by centre
            others = [j for j in range(V) if j != vi]
            order = np.argsort(np.linalg.norm(cams[others] - cams[vi], axis=1))
            alts = [others[k] for k in order[:n_alt]]
        nrm_t = torch.from_numpy(n).float().to(device)        # (B,3) ray normals
        origins = c2w_all[:, :3, 3]                            # (V,3) cam centres
        cos_thr = getattr(train_cfg, "cos_thresh", 0.1) if train_cfg else 0.1
        occ_md = getattr(train_cfg, "occ_mode", "pinhole") if train_cfg else "pinhole"
        Bn = x_hit.shape[0]
        cost_sum = torch.zeros(Bn, device=device)
        used = torch.zeros(Bn, device=device)
        cost_sum_n = torch.zeros(Bn, device=device)
        used_n = torch.zeros(Bn, device=device)
        # Per-(pixel, alt) ZNCC for the position patch, retained for the
        # pool-stats PNG (NaN = gated out / non-finite / below ncc_min).
        Zpool = torch.full((Bn, len(alts)), float("nan"), device=device)
        fg_self = torch.from_numpy(fg.reshape(-1).copy()).to(device)
        for s in range(0, Bn, 8192):
            sl = slice(s, s + 8192)
            xs, ns = x_hit[sl], nrm_t[sl]
            fss = fg_self[sl]
            for ai, ak in enumerate(alts):
                op = origins[ak]                                # (3,)
                dirv = op.unsqueeze(0) - xs                      # (b,3) hit→altcam
                dist = dirv.norm(dim=-1).clamp(min=1e-6)
                dp = dirv / dist.unsqueeze(-1)
                # in_frame (project x into alt, full res)
                Rb = w2c_all[ak, :3, :3]; tb = w2c_all[ak, :3, 3]
                xc = xs @ Rb.T + tb
                ph = xc @ K_all[ak].T
                uv = ph[:, :2] / ph[:, 2:3].clamp(min=1e-6)
                in_fr = ((xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W_full)
                         & (uv[:, 1] >= 0) & (uv[:, 1] < H_full))
                cos_ok = (ns * dp).sum(-1).abs() > cos_thr
                # fg_alt: reprojected pixel inside alt fg mask
                uc = uv.long()
                uc0 = uc[:, 0].clamp(0, W_full - 1)
                uc1 = uc[:, 1].clamp(0, H_full - 1)
                fg_alt = masks_t[ak, uc1, uc0]
                # occlusion (same as photo_loss occ_mode)
                if occ_md == "from_hit":
                    _, tp, hp = trace_nograd(f, xs + 1e-2 * dp, dp, trace_cfg)
                    not_occl = (~hp) | (tp > dist - 0.1)
                else:                                            # pinhole
                    _, tp, hp = trace_nograd(f, op.unsqueeze(0).expand_as(xs),
                                             -dp, trace_cfg)
                    not_occl = hp & (dist <= tp + 0.1)
                gate = in_fr & cos_ok & fg_alt & not_occl & fss
                if not gate.any():
                    continue
                z = _zncc_per_ray(imgs_t, xs, ns, vi, ak, K_all, w2c_all,
                                  H_full, W_full, ncc_P, ncc_hp, ncc_clr, ncc_ga)
                u = gate & torch.isfinite(z) & (z > ncc_min)
                cost_sum[sl] += torch.where(u, 1.0 - z, torch.zeros_like(z))
                used[sl] += u.float()
                Zpool[sl, ai] = torch.where(u, z, Zpool[sl, ai])
                if ncc_n_split:
                    zn = _zncc_per_ray(imgs_t, xs, ns, vi, ak, K_all, w2c_all,
                                       H_full, W_full, ncc_nP, ncc_nhp,
                                       ncc_clr, ncc_ga)
                    un = gate & torch.isfinite(zn) & (zn > ncc_min)
                    cost_sum_n[sl] += torch.where(un, 1.0 - zn,
                                                  torch.zeros_like(zn))
                    used_n[sl] += un.float()
        ncc = (cost_sum / used.clamp(min=1)).cpu().numpy().reshape(H, W)
        used_np = used.cpu().numpy().reshape(H, W)
        ncc[used_np == 0] = np.nan
        num_alt = used_np
        ncc = np.where(fg & hit_np.astype(bool), ncc, np.nan)
        num_alt_m = np.where(fg, num_alt, np.nan)
        if ncc_n_split:
            used_n_np = used_n.cpu().numpy().reshape(H, W)
            ncc_n = (cost_sum_n / used_n.clamp(min=1)).cpu().numpy().reshape(H, W)
            ncc_n[used_n_np == 0] = np.nan
            ncc_n = np.where(fg & hit_np.astype(bool), ncc_n, np.nan)

        diff = np.clip((n * light).sum(-1), 0, 1).reshape(H, W)
        hmap = np.ones((H, W, 3), np.float32)
        hmap[fg & hit_np.astype(bool)] = [.4, .85, .4]
        hmap[fg & ~hit_np.astype(bool)] = [.9, .2, .2]

        # cause map: split holes into tracer-miss (surface present, min_t f<eps
        # but not registered → sphere-tracing failed) vs absent (min_t f≥eps →
        # geometry genuinely eroded / never formed).
        hb = hit_np.astype(bool)
        eps = trace_cfg.eps
        present = sdf_min < eps
        cause = np.ones((H, W, 3), np.float32)               # white = bg
        cause[fg & hb] = [.4, .85, .4]                       # green = correct
        cause[fg & ~hb & present] = [.25, .45, .95]          # blue  = tracer-miss
        cause[fg & ~hb & ~present] = [.9, .2, .2]            # red   = surface absent

        # clip |f(x_end)| display: miss rays sit at t_far (|f|~scene scale) and
        # crush the near-surface band to black. Cap at a robust small range.
        fend_fg = np.where(fg, f_end, np.nan)
        fmax = float(np.nanpercentile(fend_fg, 75)) if np.isfinite(fend_fg).any() else 1.0
        fmax = min(max(fmax, 5 * trace_cfg.eps), 0.5)

        P = [
            (np.where(hit_np[..., None], gt, 1.0).clip(0, 1), "GT×hit", None),
            (hmap, "hit-map (red=hole)", None),
            (np.where(fg, sdf_min, np.nan), "min_t f  (fg)", "coolwarm"),
            (np.where(fg, np.clip(f_end, 0, fmax), np.nan),
             f"|f(x_end)| (fg, ≤{fmax:.3f})", "magma"),
            (np.where(hit_np[..., None], (.35 + .65 * diff[..., None])
                      * np.array([.72, .72, .85]), 1.0).clip(0, 1), "normals", None),
            (ncc, f"pos NCC 1-ZNCC P{ncc_P}/h{ncc_hp:g} ({len(alts)}a)",
             "inferno"),
            (num_alt_m, f"num_alt_used (/{n_alt})", "viridis"),
            (cause, "cause: blue=tracer red=absent", None),
        ]
        # final |f| at the (post-Newton) hit point, in units of eps:
        #   ≪1  → Newton converges well below eps, eps is NOT the detail cap
        #   ≈1  → hit frozen at the eps shell → eps quantises the surface
        # (clipped to [0,2] so 1.0 = the eps reference; green = slack).
        hitm = fg & hit_np.astype(bool)
        feps = np.where(hitm, f_end / max(eps, 1e-12), np.nan)
        P.append((np.clip(feps, 0, 2), f"|f|/eps (hit)  eps={eps:g}",
                  "RdYlGn_r"))
        if ncc_n_split:
            P.insert(6, (ncc_n,
                         f"nrm NCC 1-ZNCC P{ncc_nP}/h{ncc_nhp:g} ({len(alts)}a)",
                         "inferno"))
        for col, (img, ti, cm) in enumerate(P):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cm) if cm else ax.imshow(img)
            if cm:
                fig.colorbar(im, ax=ax, fraction=.046, shrink=.8)
            ax.axis("off")
            if row == 0:
                ax.set_title(ti, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"view {vi}", fontsize=10)

        # collect region zoom crops for this view
        for region, box in zoom.get(str(vi), {}).items():
            bx0, by0, bx1, by1 = (v / down for v in box)
            mx = (bx1 - bx0) * 0.30; my = (by1 - by0) * 0.30   # 30% margin
            x0 = max(0, int(round(bx0 - mx))); x1 = min(W, int(round(bx1 + mx)))
            y0 = max(0, int(round(by0 - my))); y1 = min(H, int(round(by1 + my)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            sub = [(im[y0:y1, x0:x1], ti, cm) for im, ti, cm in P
                   if any(k in ti for k in ("GT×hit", "min_t f",
                                            "actual NCC", "cause"))]
            zoom_rows.append((vi, region, sub))

        n_fg = int(fg.sum())
        n_hit_fg = int((fg & hit_np.astype(bool)).sum())
        row_csv = dict(
            step=step, view=vi, n_fg=n_fg,
            recall=round(n_hit_fg / max(n_fg, 1), 4),
            n_hole=int((fg & ~hb).sum()),
            frac_tracer_miss=round(float((fg & ~hb & present).sum()
                                         / max(int((fg & ~hb).sum()), 1)), 4),
            frac_absent=round(float((fg & ~hb & ~present).sum()
                                    / max(int((fg & ~hb).sum()), 1)), 4),
            sdf_min_fg=round(float(np.nanmean(np.where(fg, sdf_min, np.nan))), 5),
            f_end_fg_hit=round(float(np.nanmean(
                np.where(fg & hit_np.astype(bool), f_end, np.nan))), 5),
            ncc_fg_hit=round(float(np.nanmean(ncc)), 4),
            num_alt_mean=round(float(np.nanmean(
                np.where(fg & hit_np.astype(bool), num_alt, np.nan))), 3),
        )
        # --- eps headroom split by texture (answers: does eps cap detail?) ---
        # Texture proxy = local GT luminance-gradient magnitude; split hit
        # pixels into textured (where fine detail lives) vs smooth at the
        # fg-hit median. If median |f|/eps ≈ 1 on TEXTURED pixels, eps is
        # quantising exactly the regions whose detail you want.
        if hitm.any():
            lum = gt.mean(-1)
            gy, gx = np.gradient(lum)
            gmag = np.hypot(gx, gy)
            r_eps = f_end / max(eps, 1e-12)
            gh = gmag[hitm]
            thr = float(np.median(gh)) if gh.size else 0.0
            tx = hitm & (gmag > thr)
            sm = hitm & (gmag <= thr)
            def _q(m):
                v = r_eps[m]
                if v.size == 0:
                    return (0.0, 0.0)
                return (round(float(np.median(v)), 3),
                        round(float(np.percentile(v, 90)), 3))
            tx_med, tx_p90 = _q(tx)
            sm_med, sm_p90 = _q(sm)
            row_csv.update(feps_tx_med=tx_med, feps_tx_p90=tx_p90,
                           feps_sm_med=sm_med, feps_sm_p90=sm_p90)
            print(f"  [eps] v{vi} |f|/eps (med/p90)  "
                  f"textured={tx_med:.2f}/{tx_p90:.2f}  "
                  f"smooth={sm_med:.2f}/{sm_p90:.2f}  "
                  f"(eps={eps:g}; ≈1 on textured → eps caps detail)")
        # --- sphere-tracing convergence numerics on HOLE pixels (fg & ~hit) ---
        hm = torch.from_numpy((fg & ~hb).reshape(-1).copy()).to(device)
        if hm.any():
            _m = lambda x: round(float(x[hm].float().mean()), 4)
            row_csv.update(
                hole_iters=_m(tdiag["iters"]),
                hole_min_absf=round(float(tdiag["min_abs"][hm].mean()), 5),
                hole_final_absf=round(float(tdiag["final_abs"][hm].mean()), 5),
                hole_t_over_tfar=round(float((tdiag["t"][hm]
                    / tdiag["t_far"][hm].clamp(min=1e-6)).mean()), 3),
                hole_escaped_frac=_m(tdiag["escaped"]),
                hole_stalled_frac=_m(tdiag["stalled"]),
                hole_signflips=round(float(tdiag["nflip"][hm].mean()), 3),
            )
            print(f"  [debug] v{vi} HOLES n={int(hm.sum())}: "
                  f"iters={row_csv['hole_iters']:.1f}/{trace_cfg.iters} "
                  f"min|f|={row_csv['hole_min_absf']:.4f} "
                  f"(eps={trace_cfg.eps}) final|f|={row_csv['hole_final_absf']:.4f} "
                  f"escaped={row_csv['hole_escaped_frac']:.0%} "
                  f"stalled={row_csv['hole_stalled_frac']:.0%} "
                  f"signflip={row_csv['hole_signflips']:.2f}")
        csvp = dbg / f"view{vi:02d}.csv"
        newf = not csvp.exists()
        with open(csvp, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row_csv))
            if newf:
                w.writeheader()
            w.writerow(row_csv)

        # --- per-view pool-stats PNG (position patch) ---
        # Robust-MVS pool diagnostics, separate from the main grid:
        #   ΔZ      = Z_top4 − Z_mean  (gain of top-K selection vs plain mean)
        #   K>0.3   = #{valid alt views with ZNCC>0.3}
        #   K>0.5   = #{valid alt views with ZNCC>0.5}
        #   K_valid = #{valid alt views} (= num_alt_used)
        # "valid" = same gate as training: in_frame & cos_ok & fg_alt &
        # not_occl & fg_self & finite & ZNCC>ncc_min.
        with torch.no_grad():
            valid = ~torch.isnan(Zpool)                       # (Bn, A)
            Kval = valid.sum(1)                                # (Bn,)
            Zf0 = torch.where(valid, Zpool, torch.zeros_like(Zpool))
            Zmean = Zf0.sum(1) / Kval.clamp(min=1)
            K4 = min(4, Zpool.shape[1])
            Zsent = torch.where(valid, Zpool,
                                torch.full_like(Zpool, -2.0))
            tvals, _ = Zsent.topk(K4, dim=1)                   # (Bn, K4)
            picks = tvals > -1.5
            cnt = picks.sum(1)
            Ztop4 = (torch.where(picks, tvals,
                                 torch.zeros_like(tvals)).sum(1)
                     / cnt.clamp(min=1))
            dZ = Ztop4 - Zmean
            Kg3 = (valid & (Zpool > 0.3)).sum(1)
            Kg5 = (valid & (Zpool > 0.5)).sum(1)
        none = Kval == 0
        mask2 = fg & hit_np.astype(bool)
        def _m2(t, blank_no_pool=True):
            a = t.float().cpu().numpy().reshape(H, W)
            if blank_no_pool:
                a = np.where(none.cpu().numpy().reshape(H, W), np.nan, a)
            return np.where(mask2, a, np.nan)
        dZ_m = _m2(dZ)
        zlim = float(np.nanpercentile(np.abs(dZ_m), 98)) if np.isfinite(dZ_m).any() else 0.1
        zlim = max(zlim, 1e-3)
        PS = [
            (dZ_m, f"ΔZ=Z_top4−Z_mean  P{ncc_P}/h{ncc_hp:g}", "coolwarm",
             (-zlim, zlim)),
            (_m2(Kg3, False), f"K>0.3 (/{len(alts)})", "viridis", (0, len(alts))),
            (_m2(Kg5, False), f"K>0.5 (/{len(alts)})", "viridis", (0, len(alts))),
            (_m2(Kval, False), f"K_valid (/{len(alts)})", "viridis",
             (0, len(alts))),
        ]
        psf, psax = plt.subplots(1, 4, figsize=(4 * 4, 4))
        for c, (img, ti, cm, lim) in enumerate(PS):
            im = psax[c].imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1])
            psax[c].set_title(ti, fontsize=9)
            psax[c].axis("off")
            psf.colorbar(im, ax=psax[c], fraction=0.046, pad=0.04)
        psf.suptitle(f"pool-stats v{vi}  step {step}", fontsize=11)
        psf.tight_layout()
        psout = dbg / f"poolstats_v{vi:02d}_{step:05d}.png"
        psf.savefig(psout, dpi=130, bbox_inches="tight"); plt.close(psf)
        print(f"  [debug] → debug/{psout.name}")

    fig.suptitle(f"debug step {step}  views {view_ids}", fontsize=11)
    fig.tight_layout()
    out = dbg / f"dbg_{step:05d}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [debug] → debug/{out.name}  (+ per-view csv)")

    if zoom_rows:
        nc = max(len(s) for _, _, s in zoom_rows)
        # one shared (vmin,vmax) per colour-mapped column so crops are comparable
        col_lim = {}
        for cidx in range(nc):
            vals = []
            for _, _, sub in zoom_rows:
                if cidx < len(sub) and sub[cidx][2] is not None:
                    a = sub[cidx][0]
                    vals.append(a[np.isfinite(a)])
            if vals:
                allv = np.concatenate(vals)
                if allv.size:
                    col_lim[cidx] = (float(np.nanpercentile(allv, 2)),
                                     float(np.nanpercentile(allv, 98)))
        zf, zax = plt.subplots(len(zoom_rows), nc,
                               figsize=(3.4 * nc, 3.4 * len(zoom_rows)))
        if len(zoom_rows) == 1:
            zax = zax[None, :]
        last_h = {}
        for r, (vi, region, sub) in enumerate(zoom_rows):
            for cidx in range(nc):
                ax = zax[r, cidx]
                if cidx < len(sub):
                    im, ti, cm = sub[cidx]
                    if cm and cidx in col_lim:
                        vmn, vmx = col_lim[cidx]
                        last_h[cidx] = ax.imshow(im, cmap=cm, vmin=vmn, vmax=vmx)
                    else:
                        ax.imshow(im, cmap=cm) if cm else ax.imshow(im)
                    ttl = f"v{vi} {region} | {ti}" if cidx == 0 else (
                        ti if r == 0 else "")
                    if ttl:
                        ax.set_title(ttl, fontsize=9)
                ax.axis("off")
        # single shared colorbar under each colour-mapped column
        for cidx, h in last_h.items():
            zf.colorbar(h, ax=list(zax[:, cidx]), location="bottom",
                        fraction=.04, pad=.02, shrink=.9)
        zf.suptitle(f"zoom step {step}  (shared scale per column)", fontsize=11)
        zout = dbg / f"zoom_{step:05d}.png"
        zf.savefig(zout, dpi=140, bbox_inches="tight"); plt.close(zf)
        print(f"  [debug] → debug/{zout.name}  ({len(zoom_rows)} region crops)")


# ---------- chamfer helpers ----------

def _chamfer(pred: torch.Tensor, gt: torch.Tensor, chunk: int = 4096) -> dict[str, float]:
    """Chamfer-L1 split into precision (pred->gt) and completeness (gt->pred)."""
    def _one_way(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mins = []
        for i in range(0, len(a), chunk):
            d2 = (a[i:i+chunk].unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
            mins.append(d2.min(dim=1).values.sqrt())   # L2, not L2²
        return torch.cat(mins).mean()
    precision = _one_way(pred, gt).item()
    completeness = _one_way(gt, pred).item()
    return {
        "precision": precision,
        "completeness": completeness,
        "chamfer": 0.5 * (precision + completeness),
    }


def _mc_chamfer(f, gt_pts: torch.Tensor, device: str,
                bound: float = 1.5, res: int = 128) -> dict[str, float] | None:
    """Marching-cubes surface extraction + split Chamfer vs GT point cloud.

    The MC mesh is reduced to its largest connected component (by surface area)
    before sampling, so disconnected floaters do not inflate Chamfer.
    """
    from skimage.measure import marching_cubes
    import trimesh
    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i+4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None
    spacing = 2 * bound / (res - 1)
    verts, faces, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts = (verts - bound).astype(np.float32)
    if len(verts) == 0 or len(faces) == 0:
        return None
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.area)
    if len(mesh.vertices) == 0:
        return None
    main_verts = np.asarray(mesh.vertices, dtype=np.float32)
    n = min(30_000, len(main_verts))
    idx = np.random.default_rng(0).choice(len(main_verts), n, replace=False)
    pred_pts = torch.from_numpy(main_verts[idx]).to(device)
    return _chamfer(pred_pts, gt_pts.to(device))


def _mc_sfm_surface_distance(
    f,
    sfm_pts: torch.Tensor,
    device: str,
    bound: float = 1.5,
    res: int = 128,
) -> dict[str, float] | None:
    """Approximate e_i = min_x_surface ||x_sfm_i - x_surface|| from MC vertices."""
    from skimage.measure import marching_cubes

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i + 4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None

    spacing = 2 * bound / (res - 1)
    verts, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts = (verts - bound).astype(np.float32)
    if len(verts) == 0:
        return None

    sfm_np = sfm_pts.detach().cpu().numpy().astype(np.float32)
    try:
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(verts).query(sfm_np, k=1, workers=-1)
    except Exception:
        v = torch.from_numpy(verts).to(device)
        q = sfm_pts.to(device)
        chunks = []
        for i in range(0, len(q), 4096):
            d2 = (q[i:i + 4096].unsqueeze(1) - v.unsqueeze(0)).pow(2).sum(-1)
            chunks.append(d2.min(dim=1).values.sqrt().detach().cpu())
        dist = torch.cat(chunks).numpy()

    return {
        "mean": float(dist.mean()),
        "p50": float(np.percentile(dist, 50)),
        "p90": float(np.percentile(dist, 90)),
        "p99": float(np.percentile(dist, 99)),
    }


def _load_dtu_eval_data(dtu_eval_dir: Path, scan_id: int, device: str) -> dict | None:
    """Load GT cloud + ObsMask for in-training DTU Chamfer. Returns None on error.

    dtu_eval_dir should be the 'SampleSet/MVS Data' directory from the official
    DTU SampleSet.zip download, containing Points/stl/ and ObsMask/ subdirs.
    GT points are pre-filtered by the ObsMask so NN search stays fast.
    """
    try:
        import open3d as o3d
        from scipy.io import loadmat
    except ImportError as e:
        print(f"  [dtu_chamfer] skipping — missing dependency: {e}"); return None
    # stl006 has better surface coverage; fall back to stl001
    ply = dtu_eval_dir / "Points" / "stl" / "stl006_total.ply"
    if not ply.exists():
        ply = dtu_eval_dir / "Points" / "stl" / "stl001_total.ply"
    mat = dtu_eval_dir / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    if not ply.exists() or not mat.exists():
        print(f"  [dtu_chamfer] eval files not found under {dtu_eval_dir}"); return None
    m = loadmat(str(mat))
    ObsMask = m["ObsMask"].astype(bool)
    BB      = m["BB"].astype(np.float64)
    Res     = float(m["Res"].flat[0])

    # load full GT cloud and pre-filter to this scene's ObsMask (avoids slow NN over all scenes)
    all_pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points, dtype=np.float32)
    in_bb   = np.all((all_pts >= BB[0]) & (all_pts <= BB[1]), axis=1)
    idx_bb  = np.clip(np.round((all_pts[in_bb] - BB[0]) / Res).astype(int),
                      0, np.array(ObsMask.shape) - 1)
    in_mask = ObsMask[idx_bb[:, 0], idx_bb[:, 1], idx_bb[:, 2]]
    gt_pts  = all_pts[in_bb][in_mask]
    print(f"  [dtu_chamfer] GT cloud (scan{scan_id}): {len(gt_pts):,} pts after ObsMask filter "
          f"(from {len(all_pts):,} total in {ply.name})")
    return {"gt_pts": torch.from_numpy(gt_pts).to(device),
            "ObsMask": ObsMask, "BB": BB, "Res": Res}


def _mc_chamfer_dtu(f, dtu_data: dict, scale_mat: np.ndarray, device: str,
                    bound: float = 1.5, res: int = 128) -> dict[str, float] | None:
    """Fast in-training DTU Chamfer: MC in normalised space → world → ObsMask-masked.

    Uses scipy cKDTree on CPU to avoid materialising the full distance matrix
    (GT cloud has ~2M points; brute-force GPU NN would OOM).
    """
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i+4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None
    spacing = 2 * bound / (res - 1)
    verts_norm, faces_norm, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts_norm = (verts_norm - bound).astype(np.float32)
    if len(verts_norm) == 0 or len(faces_norm) == 0:
        return None

    # largest connected component (by surface area) — drop MC floaters
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts_norm, faces=faces_norm, process=False)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.area)
    verts_norm = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts_norm) == 0:
        return None

    # normalised → DTU world
    v_h = np.concatenate([verts_norm, np.ones((len(verts_norm), 1), dtype=np.float32)], axis=1)
    verts_world = (scale_mat @ v_h.T).T[:, :3].astype(np.float32)

    ObsMask, BB, Res = dtu_data["ObsMask"], dtu_data["BB"], dtu_data["Res"]
    def _in_obs(pts):
        in_bb = np.all((pts >= BB[0]) & (pts <= BB[1]), axis=1)
        idx   = np.clip(np.round((pts - BB[0]) / Res).astype(int), 0, np.array(ObsMask.shape) - 1)
        return in_bb & ObsMask[idx[:, 0], idx[:, 1], idx[:, 2]]

    pred_np = verts_world[_in_obs(verts_world)]
    gt_np   = dtu_data["gt_pts"].cpu().numpy()   # already pre-filtered at load time
    if len(pred_np) == 0 or len(gt_np) == 0:
        return None

    tree_gt   = cKDTree(gt_np)
    tree_pred = cKDTree(pred_np)
    acc,  _ = tree_gt.query(pred_np, k=1, workers=-1)
    comp, _ = tree_pred.query(gt_np,  k=1, workers=-1)
    return {"precision": float(acc.mean()), "completeness": float(comp.mean()),
            "chamfer": 0.5 * (float(acc.mean()) + float(comp.mean()))}


def _extract_world_mesh_for_dtu(f, scale_mat: np.ndarray, device: str, out_ply: Path,
                                bound: float = 1.0, res: int = 384) -> Path | None:
    from skimage.measure import marching_cubes
    import trimesh

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i + 4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).detach().cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None

    spacing = 2 * bound / (res - 1)
    verts_norm, faces, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts_norm = (verts_norm - bound).astype(np.float32)
    if len(verts_norm) == 0:
        return None

    v_h = np.concatenate([verts_norm, np.ones((len(verts_norm), 1), dtype=np.float32)], axis=1)
    verts_world = (scale_mat @ v_h.T).T[:, :3].astype(np.float32)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=verts_world, faces=faces, process=False).export(str(out_ply))
    return out_ply


def _run_mvmannequin_official_eval(f, scene: Path, out_dir: Path,
                                   bound: float, res: int, device: str,
                                   ) -> dict[str, float] | None:
    """In-training MVMannequin Chamfer eval. Reproduces Inria-Morpheo's official
    protocol exactly: z>0.05 m slice + largest CC + ICP-p2l (Tukey, 25 mm)
    + pysdf point-to-mesh distance + clamp@100 mm.

    Two-stage: (1) extract pred mesh in WORLD frame from this venv, (2) subprocess
    to the pixi env which has pysdf + open3d.
    """
    eval_script = Path(__file__).resolve().parent.parent / "eval_mvmannequin_official.py"
    if not eval_script.exists():
        print(f"  [mvm_official] skipped — missing {eval_script}", flush=True)
        return None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(eval_script.parent))
        from eval_mvmannequin_official import extract_pred_mesh, write_gt_world_mesh
        pred_ply = out_dir / "pred_world_mesh.ply"
        gt_ply   = out_dir / "gt_world_mesh.ply"
        # Extract MC mesh (training venv has torch + lip_tracer); save in world frame.
        from skimage.measure import marching_cubes
        import trimesh
        vox = torch.linspace(-bound, bound, res, device=device)
        grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
        with torch.no_grad():
            vals = torch.cat([f(grid[i:i + 4096]) for i in range(0, len(grid), 4096)])
        vol = vals.reshape(res, res, res).detach().cpu().numpy()
        if vol.min() > 0 or vol.max() < 0:
            print(f"  [mvm_official] surface not in bounds (vol range "
                  f"[{vol.min():.3f},{vol.max():.3f}])", flush=True)
            return None
        spacing = 2 * bound / (res - 1)
        v_norm, faces, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
        v_norm = (v_norm - bound).astype(np.float64)
        scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
        v_h = np.concatenate([v_norm, np.ones((len(v_norm), 1))], axis=1)
        v_world = (scale_mat @ v_h.T).T[:, :3]
        trimesh.Trimesh(vertices=v_world, faces=faces, process=False).export(str(pred_ply))
        write_gt_world_mesh(scene, gt_ply)

        # Run the official protocol in the pixi env (pysdf + open3d)
        pixi_python = Path("/home/glenain/nerfstudio/.pixi/envs/default/bin/python3")
        eval_python = str(pixi_python) if pixi_python.exists() else sys.executable
        cmd = [eval_python, str(eval_script),
               "--mesh", str(pred_ply), "--scene", str(scene),
               "--out", str(out_dir), "--no-subprocess"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        (out_dir / "official_stdout.txt").write_text(result.stdout)
        (out_dir / "official_stderr.txt").write_text(result.stderr)
        if result.returncode != 0:
            print(f"  [mvm_official] failed code={result.returncode}; see {out_dir}", flush=True)
            return None
        try:
            payload = json.loads((out_dir / "mvmannequin_official.json").read_text())
        except FileNotFoundError:
            print(f"  [mvm_official] missing metrics json; see {out_dir}", flush=True)
            return None
        return {"accuracy": payload["accuracy_mm"],
                "completeness": payload["completeness_mm"],
                "chamfer": payload["chamfer_mm"]}
    except Exception as e:  # noqa: BLE001
        print(f"  [mvm_official] failed: {e}", flush=True)
        return None


def _run_dtu_official_eval(mesh_ply: Path, scan_id: int, dtu_eval_dir: Path,
                           out_dir: Path) -> dict[str, float] | None:
    eval_script = Path(__file__).resolve().parent.parent / "DTUeval-python" / "eval.py"
    if not eval_script.exists():
        print(f"  [dtu_official] skipped — missing {eval_script}", flush=True)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import open3d  # noqa: F401
        eval_python = sys.executable
    except ImportError:
        pixi_python = Path("/home/glenain/nerfstudio/.pixi/envs/default/bin/python3")
        eval_python = str(pixi_python) if pixi_python.exists() else sys.executable
    cmd = [
        eval_python, str(eval_script),
        "--data", str(mesh_ply),
        "--scan", str(scan_id),
        "--mode", "mesh",
        "--dataset_dir", str(dtu_eval_dir),
        "--vis_out_dir", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "official_stdout.txt").write_text(result.stdout)
    (out_dir / "official_stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        print(f"  [dtu_official] failed code={result.returncode}; see {out_dir}", flush=True)
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    try:
        acc, comp, chamfer = [float(v) for v in lines[-1].split()]
    except (IndexError, ValueError):
        print(f"  [dtu_official] failed to parse metrics; see {out_dir}", flush=True)
        return None
    payload = {"accuracy": acc, "completeness": comp, "chamfer": chamfer}
    (out_dir / "dtu_official.json").write_text(json.dumps(payload, indent=2))
    return payload


# ---------- debug region monitoring ----------

def _setup_debug_regions(region_str: str, views: dict, device: str) -> list[dict]:
    """Parse 'name,view,u0,v0,u1,v1;...' and precompute ray tensors for each region.

    Coordinates are full-resolution image pixels (column=u, row=v).
    Returns a list of dicts with keys: name, o (N,3), d (N,3), n_rays.
    """
    if not region_str.strip():
        return []
    H, W = views["H"], views["W"]
    regions = []
    for entry in region_str.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        if len(parts) != 6:
            print(f"  [debug_regions] bad entry (expected name,view,u0,v0,u1,v1): {entry!r}")
            continue
        name, vi, u0, v0, u1, v1 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        c2w = views["c2w"][vi]   # (4,4)
        K   = views["K"][vi]     # (3,3)
        us = torch.arange(u0, u1 + 1, dtype=torch.float32)
        vs = torch.arange(v0, v1 + 1, dtype=torch.float32)
        ug, vg = torch.meshgrid(us, vs, indexing="xy")   # (W_r, H_r)
        u_flat = ug.reshape(-1); v_flat = vg.reshape(-1)
        # camera-space directions
        dir_cam = torch.stack([
            (u_flat - K[0, 2]) / K[0, 0],
            (v_flat - K[1, 2]) / K[1, 1],
            torch.ones_like(u_flat),
        ], dim=-1)                                         # (N, 3)
        dir_w = dir_cam @ c2w[:3, :3].T
        dir_w = dir_w / dir_w.norm(dim=-1, keepdim=True)
        orig  = c2w[:3, 3].unsqueeze(0).expand(dir_w.shape[0], -1)
        regions.append({
            "name":   name,
            "view":   vi,
            "u0": u0, "v0": v0, "u1": u1, "v1": v1,
            "o":      orig.to(device),
            "d":      dir_w.to(device),
            "n_rays": dir_w.shape[0],
        })
        print(f"  [debug_regions] {name!r}: view={vi}  u=[{u0},{u1}]  v=[{v0},{v1}]  rays={dir_w.shape[0]}")
    return regions


@torch.no_grad()
def _log_debug_regions(f, regions: list[dict], cfg, step: int,
                       run_dir=None) -> None:
    """Trace each debug region, print one summary line, and save a depth image."""
    if not regions:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for reg in regions:
        o, d = reg["o"], reg["d"]
        B = o.shape[0]
        t         = torch.zeros(B, device=o.device)
        sdf_min   = torch.full((B,), float("inf"), device=o.device)
        converged = torch.zeros(B, dtype=torch.bool, device=o.device)
        iters_at  = torch.full((B,), cfg.iters, dtype=torch.long, device=o.device)
        mean_f_per_iter = []   # mean |f| of active rays at each iteration
        for i in range(cfg.iters):
            escaped = t >= cfg.t_far
            active  = ~(converged | escaped)
            if not active.any():
                break
            x   = o + t.unsqueeze(-1) * d
            sdf = f(x)
            sdf_abs = sdf.abs()
            mean_f_per_iter.append(sdf_abs[active].mean().item())
            sdf_min = torch.where(active, torch.minimum(sdf_min, sdf_abs), sdf_min)
            just_conv = active & (sdf_abs < cfg.eps)
            iters_at  = torch.where(just_conv & (iters_at == cfg.iters),
                                    torch.full_like(iters_at, i), iters_at)
            converged = converged | just_conv
            t = t + torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
        hit = converged & (t < cfg.t_far)
        n_hit  = hit.sum().item()
        n_miss = B - n_hit
        hit_rate = n_hit / B
        miss_sdf = sdf_min[~hit]
        mean_miss_sdf = miss_sdf.mean().item() if n_miss > 0 else float("nan")
        mean_iters    = iters_at[hit].float().mean().item() if n_hit > 0 else float("nan")

        t_hit = t[hit]
        t_mean = t_hit.mean().item() if n_hit > 0 else float("nan")
        t_std  = t_hit.std().item()  if n_hit > 1 else 0.0
        t_min  = t_hit.min().item()  if n_hit > 0 else float("nan")
        t_max  = t_hit.max().item()  if n_hit > 0 else float("nan")

        # iter histogram: count how many hit rays converged at each iteration
        if n_hit > 0:
            it_vals = iters_at[hit]
            iter_counts = torch.zeros(cfg.iters + 1, dtype=torch.long, device=o.device)
            for iv in range(cfg.iters):
                iter_counts[iv] = (it_vals == iv).sum()
            # compact: only print iters with at least 1 ray
            hist_parts = [f"{iv}:{iter_counts[iv].item()}"
                          for iv in range(cfg.iters) if iter_counts[iv].item() > 0]
            iter_hist_str = " ".join(hist_parts)
        else:
            iter_hist_str = "-"

        f_traj = "  ".join(f"{i}:{v:.4f}" for i, v in enumerate(mean_f_per_iter))
        print(f"  [region/{reg['name']}@{step:6d}]  "
              f"hit={hit_rate:.2%} ({n_hit}/{B})  "
              f"miss_sdf_min={mean_miss_sdf:.4f}  "
              f"t_hit={t_mean:.3f}±{t_std:.3f} [{t_min:.3f},{t_max:.3f}]  "
              f"conv_iters: {iter_hist_str}")
        print(f"    f@iter: {f_traj}")

        # ── depth + iters image ───────────────────────────────────────────────
        if run_dir is None:
            continue
        # infer grid shape from bounding box stored at setup time
        u0, v0, u1, v1 = reg["u0"], reg["v0"], reg["u1"], reg["v1"]
        RW = u1 - u0 + 1   # cols
        RH = v1 - v0 + 1   # rows
        if RW * RH != B:
            continue        # shape mismatch — skip image

        t_cpu   = t.cpu().numpy().reshape(RH, RW)
        hit_cpu = hit.cpu().numpy().reshape(RH, RW)
        it_cpu  = iters_at.cpu().numpy().reshape(RH, RW).astype(float)
        it_cpu[~hit_cpu] = float("nan")

        # normalise depth within [t_min, t_max] for colour; misses → grey
        t_vis = t_cpu.copy()
        t_vis[~hit_cpu] = float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5 * RH / RW + 0.6),
                                 gridspec_kw={"wspace": 0.05})
        im0 = axes[0].imshow(t_vis, cmap="plasma", interpolation="nearest")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, label="t (depth)")
        axes[0].set_title("depth", fontsize=8); axes[0].axis("off")

        im1 = axes[1].imshow(it_cpu, cmap="viridis", interpolation="nearest")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, label="iters")
        axes[1].set_title("conv iters", fontsize=8); axes[1].axis("off")

        fig.suptitle(f"{reg['name']}  step={step}  hit={hit_rate:.1%}  "
                     f"t={t_mean:.3f}±{t_std:.3f}", fontsize=8)
        out = run_dir / "diag" / f"dbg_{reg['name']}_{step:06d}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------- training ----------

def train(cfg: Config = None, use_wandb: bool = False, resume: Path | None = None,
          run_dir: Path | None = None) -> Path:
    """Train the 1-Lip SDF and save checkpoints to a timestamped run directory.

    Returns the path to the final checkpoint.
    """
    import os
    cfg = cfg or Config()
    # unpack for convenience
    model_cfg = cfg.model
    trace_cfg = cfg.trace
    init_cfg  = cfg.init
    train_cfg = cfg.train
    eval_cfg  = cfg.eval
    scene     = cfg.scene
    out_dir   = cfg.out_dir

    # Skip the differentiable soft-min compute only when no active loss consumes
    # sdf_min. In IDR tracing the hard sdf_min is collected under no_grad, so
    # idr_mask_loss also needs sdf_min_beta to get a gradient signal.
    if train_cfg.w_sil == 0 and train_cfg.w_idr_mask == 0 and trace_cfg.sdf_min_beta > 0:
        trace_cfg = dataclasses.replace(trace_cfg, sdf_min_beta=0.0)

    # --- SLURM preemption recovery ---
    slurm_job_id      = os.environ.get("SLURM_JOB_ID")
    slurm_restart_cnt = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if slurm_restart_cnt > 0 and resume is None and slurm_job_id:
        sentinel = out_dir / f"job_{slurm_job_id}.rundir"
        if sentinel.exists():
            prev_run_dir = Path(sentinel.read_text().strip())
            # checkpoint_latest.pt lives under ckpt/ (new layout) or run root (legacy).
            for cand in (prev_run_dir / "ckpt" / "checkpoint_latest.pt",
                         prev_run_dir / "checkpoint_latest.pt"):
                if cand.exists():
                    resume = cand
                    print(f"  [preemption] SLURM_RESTART_COUNT={slurm_restart_cnt}, "
                          f"resuming from {resume}")
                    break

    # Reuse the existing run_dir when resuming (avoids cluttering outputs/).
    if resume is not None:
        # resume path may be <run>/ckpt/foo.pt (new) or <run>/foo.pt (legacy)
        run_dir = resume.parent.parent if resume.parent.name == "ckpt" else resume.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [resume] reusing run dir {run_dir}")
    elif run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if slurm_job_id:
            (out_dir / f"job_{slurm_job_id}.rundir").write_text(str(run_dir))
    else:
        ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag     = scene.name if scene is not None else ("blender" if train_cfg.use_blender else "dtu")
        run_dir = out_dir / f"run_{ts}_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Write sentinel so preemption recovery can find this directory.
        if slurm_job_id:
            (out_dir / f"job_{slurm_job_id}.rundir").write_text(str(run_dir))

    # Subfolders: ckpt/, render/, diag/. config.json + train.log stay at root.
    ckpt_dir   = run_dir / "ckpt";   ckpt_dir.mkdir(exist_ok=True)
    render_dir = run_dir / "render"; render_dir.mkdir(exist_ok=True)
    diag_dir   = run_dir / "diag";   diag_dir.mkdir(exist_ok=True)

    OUT         = ckpt_dir   / "checkpoint.pt"
    RENDER_OUT  = render_dir / "render.png"
    SURFACE_OUT = render_dir / "surface.png"

    full_cfg = {**cfg.to_dict(), "cmd": " ".join(sys.argv)}
    (run_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))
    print(f"  run dir → {run_dir}")

    if use_wandb:
        import wandb
        wandb.init(project="1lip-tracer", name=run_dir.name, config=full_cfg, dir=str(run_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    if train_cfg.use_blender:
        views = load_blender_views(scene=scene, down=train_cfg.down)
    else:
        views = load_views(scene)
    images      = views["images"].to(device).half()
    masks       = views["masks"].to(device) if "masks" in views else None
    c2w_all     = views["c2w"].to(device)
    K_all       = views["K"].to(device)
    H, W        = views["H"], views["W"]
    V           = images.shape[0]
    print(f"  views: {V} cameras  {H}×{W}")
    origins_all = c2w_all[:, :3, 3]
    w2c_all     = torch.linalg.inv(c2w_all)

    # --- bundle adjustment (joint photometric, see bundle_adjustment.py) ---
    cam_params = None
    if train_cfg.bundle.enabled:
        from .bundle_adjustment import CameraParams, rays_from_pixels
        cam_params = CameraParams(c2w_all, lock_first=train_cfg.bundle.lock_first).to(device)
        print(f"  BA enabled  lr={train_cfg.bundle.lr}  freeze_steps={train_cfg.bundle.freeze_steps}  "
              f"lock_first={train_cfg.bundle.lock_first}")

    feature_maps = None
    if train_cfg.w_feature > 0 and train_cfg.feature_maps is None:
        raise ValueError("--w-feature > 0 requires --feature-maps /path/to/features.pt")
    if train_cfg.feature_maps is not None and train_cfg.w_feature > 0:
        payload = torch.load(train_cfg.feature_maps, map_location="cpu")
        feats = payload["features"].float()
        if feats.ndim != 4:
            raise ValueError(f"expected feature maps as (V,C,H,W), got {tuple(feats.shape)}")
        if feats.shape[0] != V:
            raise ValueError(f"feature map view count {feats.shape[0]} != training views {V}")
        orig_hw = tuple(payload.get("orig_hw", (H, W)))
        if orig_hw != (H, W):
            print(f"  warning: feature orig_hw={orig_hw} differs from training image H,W={(H, W)}; "
                  "UVs will be scaled to feature resolution")
        feature_maps = F.normalize(feats, dim=1).permute(0, 2, 3, 1).contiguous().to(device).half()
        print(f"  feature maps: {train_cfg.feature_maps}  {tuple(feature_maps.shape)}  "
              f"w_feature={train_cfg.w_feature}")

    if train_cfg.use_blender:
        try:
            gt_pts = load_blender_gt_points(scene=scene)
            sfm_pts = gt_pts.to(device)
            print(f"  blender GT pts → sfm_pts: {sfm_pts.shape[0]} surface points")
        except (FileNotFoundError, ValueError) as e:
            print(f"  blender GT pts: not available ({e}) — sfm_sdf_loss disabled")
            gt_pts  = None
            sfm_pts = torch.zeros(1, 3, device=device)
        sfm_origins  = torch.empty(0, 3, device=device)
        sfm_targets  = torch.empty(0, 3, device=device)
        n_sfm_pairs  = 0
        mvs_data     = None
    else:
        gt_pts = None
        # official DTU eval data (GT cloud + ObsMask) for in-training Chamfer
        dtu_eval_data  = None
        dtu_scale_mat  = None
        dtu_scan_id    = None
        if eval_cfg.dtu_eval_dir is not None and (eval_cfg.dtu_chamfer_freq > 0 or eval_cfg.dtu_official_freq > 0):
            import re
            m = re.search(r"scan(\d+)", str(scene))
            if m:
                dtu_scan_id = int(m.group(1))
                dtu_scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
                if eval_cfg.dtu_chamfer_freq > 0:
                    dtu_eval_data = _load_dtu_eval_data(eval_cfg.dtu_eval_dir, dtu_scan_id, device)
        # MVMannequin scene auto-detect: presence of GT mesh next to cameras.npz.
        # Reuses dtu_official_freq for cadence; gt_mesh.ply lives in normalized frame.
        is_mvm = (scene / "gt_mesh.ply").exists() and (scene / "cameras.npz").exists() and not dtu_scan_id
        if train_cfg.w_sfm > 0:
            try:
                sfm_pts_all = load_colmap_points(scene)
                if train_cfg.sfm_min_views > 0:
                    sfm_vis = colmap_visibility_counts(sfm_pts_all, views)
                    sfm_pts = sfm_pts_all[sfm_vis >= train_cfg.sfm_min_views].to(device)
                    print(f"  loaded {sfm_pts.shape[0]}/{sfm_pts_all.shape[0]} SFM points "
                          f"(>={train_cfg.sfm_min_views} views)")
                else:
                    sfm_pts = sfm_pts_all.to(device)
                    print(f"  loaded {sfm_pts.shape[0]} SFM points")
            except FileNotFoundError as e:
                print(f"  sparse_sfm_points.txt not found — w_sfm disabled ({e})")
                train_cfg = dataclasses.replace(train_cfg, w_sfm=0.0)
                sfm_pts = torch.zeros(1, 3, device=device)
        else:
            sfm_pts = torch.zeros(1, 3, device=device)
        if train_cfg.w_sfm > 0:
            try:
                sfm_origins_cpu, sfm_targets_cpu = load_sfm_pairs(scene, allowed_points=sfm_pts)
                sfm_origins = sfm_origins_cpu.to(device)
                sfm_targets = sfm_targets_cpu.to(device)
                n_sfm_pairs = sfm_origins.shape[0]
                print(f"  {n_sfm_pairs} cleaned (cam, sfm_point) pairs for SFM free-space "
                      f"(same >= {train_cfg.sfm_min_views} view filter)")
            except (FileNotFoundError, KeyError) as e:
                print(f"  sfm_pairs/meta_data not found — SFM surface term only ({e})")
                sfm_origins = torch.empty(0, 3, device=device)
                sfm_targets = torch.empty(0, 3, device=device)
                n_sfm_pairs = 0
        elif train_cfg.w_free > 0:
            try:
                sfm_origins, sfm_targets = load_sfm_pairs(scene)
                sfm_origins, sfm_targets = sfm_origins.to(device), sfm_targets.to(device)
                n_sfm_pairs = sfm_origins.shape[0]
                print(f"  {n_sfm_pairs} (cam, sfm_point) pairs for free-space loss")
            except (FileNotFoundError, KeyError) as e:
                print(f"  meta_data.json not found — w_free disabled ({e})")
                train_cfg = dataclasses.replace(train_cfg, w_free=0.0)
                sfm_origins = torch.empty(0, 3, device=device)
                sfm_targets = torch.empty(0, 3, device=device)
                n_sfm_pairs = 0
        else:
            sfm_origins = torch.empty(0, 3, device=device)
            sfm_targets = torch.empty(0, 3, device=device)
            n_sfm_pairs = 0
        if train_cfg.w_surf > 0 or train_cfg.w_mvs > 0 or train_cfg.w_mvs_sdf > 0 or train_cfg.w_normal > 0 or train_cfg.mvsdf_schedule.enabled:
            if train_cfg.mvsformer_depth_dir is not None:
                from .geomvs import load_mvsformer_depths_idr
                mvs_data = load_mvsformer_depths_idr(
                    scene, train_cfg.mvsformer_depth_dir,
                    conf_thresh=train_cfg.mvsformer_conf_thr)
            elif train_cfg.mvs_depth_dir is not None:
                from .geomvs import load_mast3r_depths_idr
                mvs_data = load_mast3r_depths_idr(scene, train_cfg.mvs_depth_dir)
            else:
                from .geomvs import load_aligned_depths
                mvs_data = load_aligned_depths(scene)
            if mvs_data is not None:
                print(f"  loaded {len(mvs_data['depths'])} aligned depth maps")
        else:
            mvs_data = None

    # --- model ---
    bound = EvalConfig().bound(train_cfg.use_blender)
    mvs_sdf_bounds_np = None
    _resume_ckpt = None
    if resume is not None:
        print(f"  resuming from {resume} — skipping init")
        _resume_ckpt = torch.load(resume, map_location="cpu")
        f = make_model(hidden=model_cfg.hidden, depth=model_cfg.depth,
                       group_size=model_cfg.group_size, activation=model_cfg.activation,
                       input_encoding=model_cfg.input_encoding, multires=model_cfg.multires,
                       architecture=model_cfg.architecture)
        f.load_state_dict(_resume_ckpt["f"], strict=False)
    elif init_cfg.init == "hull":
        f, hull_occ = fit_hull_init(
            model_cfg=model_cfg, init_cfg=init_cfg, scene=scene, bound=bound,
            return_info=True,
        )
        mvs_sdf_bounds_np = _visual_hull_sample_bounds(hull_occ, bound)
        if mvs_sdf_bounds_np is not None:
            lo_np, hi_np = mvs_sdf_bounds_np
            print(f"  mvs-sdf samples: visual-hull AABB "
                  f"x=[{lo_np[0]:+.3f},{hi_np[0]:+.3f}] "
                  f"y=[{lo_np[1]:+.3f},{hi_np[1]:+.3f}] "
                  f"z=[{lo_np[2]:+.3f},{hi_np[2]:+.3f}]")
        torch.cuda.empty_cache()
    elif init_cfg.init == "colmap":
        f, hull_occ = fit_colmap_init(
            model_cfg=model_cfg, init_cfg=init_cfg, scene=scene, bound=bound,
        )
        mvs_sdf_bounds_np = _visual_hull_sample_bounds(hull_occ, bound)
        if mvs_sdf_bounds_np is not None:
            lo_np, hi_np = mvs_sdf_bounds_np
            print(f"  mvs-sdf samples: COLMAP-hull AABB "
                  f"x=[{lo_np[0]:+.3f},{hi_np[0]:+.3f}] "
                  f"y=[{lo_np[1]:+.3f},{hi_np[1]:+.3f}] "
                  f"z=[{lo_np[2]:+.3f},{hi_np[2]:+.3f}]")
        torch.cuda.empty_cache()
    else:
        _init_cfg = InitConfig(radius=1.0) if train_cfg.use_blender else init_cfg
        if train_cfg.use_blender:
            print("  [blender] sphere-init radius=1.0")
        f, _ = fit_sphere_init(model_cfg=model_cfg, init_cfg=_init_cfg, scene=scene)
    f = f.to(device)
    total_params = sum(p.numel() for p in f.parameters())
    n_cpl = sum(1 for m in f.net if isinstance(m, ConvexPotentialLayer)) if hasattr(f, "net") else 0
    arch_tag = getattr(f, "architecture", "mlp")
    act_tag = getattr(f, "activation", "-")
    print(f"  model: hidden={f.hidden}  depth={n_cpl or f.depth}  params={total_params:,}  "
          f"arch={arch_tag}  act={act_tag}  enc={f.input_encoding}  multires={f.multires}")
    if train_cfg.profile:
        from .profiling import profile_model
        _prof = profile_model(f, device, batch=train_cfg.batch,
                              iters=trace_cfg.iters)
        if use_wandb:
            import wandb
            wandb.summary.update(_prof)
    if mvs_sdf_bounds_np is not None:
        mvs_sdf_lo = torch.from_numpy(mvs_sdf_bounds_np[0]).to(device)
        mvs_sdf_hi = torch.from_numpy(mvs_sdf_bounds_np[1]).to(device)
    else:
        mvs_sdf_lo = torch.full((3,), -bound, device=device)
        mvs_sdf_hi = torch.full((3,),  bound, device=device)

    # --- deterministic rays ---
    det        = make_deterministic_rays(views, down=train_cfg.down, device=device)
    total_rays = det["o"].shape[0]
    rpv        = det["rays_per_view"]
    print(f"  det rays: {total_rays} total, {rpv}/view, {V} views, down={train_cfg.down}")
    fg_frac = det["fg"].float().mean().item()
    print(f"  fg rays: {det['fg'].sum():.0f}/{total_rays} ({fg_frac:.1%})"
          f"{'  [NO MASKS]' if masks is None else ''}")
    fg_idx = det["fg"].nonzero(as_tuple=True)[0]
    bg_idx = (~det["fg"]).nonzero(as_tuple=True)[0]
    _bg_loss_active = any(w > 0 for w in (
        train_cfg.w_mask_bg, train_cfg.w_idr_mask, train_cfg.w_sil,
        train_cfg.w_behind_hit, train_cfg.w_ray_free,
    ))
    if len(bg_idx) == 0 or not _bg_loss_active:
        n_fg = train_cfg.batch
        n_bg = 0
    elif len(fg_idx) == 0:
        n_fg = 0
        n_bg = train_cfg.batch
    else:
        n_fg = int(train_cfg.batch * 0.7)
        n_bg = train_cfg.batch - n_fg
    print(f"  batch split: n_fg={n_fg}  n_bg={n_bg}  "
          f"(bg_loss_active={_bg_loss_active}, |bg_idx|={len(bg_idx)})")

    # --- MVS depth alignment ---
    if mvs_data is None:
        mvs_depth_flat  = torch.zeros(total_rays, device=device)
        mvs_valid_flat  = torch.zeros(total_rays, dtype=torch.bool, device=device)
        mvs_normal_flat = torch.zeros(total_rays, 3, device=device)
        H_d, W_d = H // train_cfg.down, W // train_cfg.down
        mvs_depth_maps  = torch.zeros(V, H_d, W_d, device=device)
        mvs_valid_maps  = torch.zeros(V, H_d, W_d, dtype=torch.bool, device=device)
        mvs_normal_maps = torch.zeros(V, H_d, W_d, 3, device=device)
    else:
        H_d, W_d = H // train_cfg.down, W // train_cfg.down
        dep_all, val_all, nor_all = [], [], []
        for v in range(V):
            d_full = mvs_data["depths"][v]; v_full = mvs_data["valid"][v]
            n_full = mvs_data["normals"][v]   # (H, W, 3) camera-space
            ys_d = torch.arange(H_d); xs_d = torch.arange(W_d)
            ys_f = ((ys_d.float() + 0.5) * train_cfg.down - 0.5).long().clamp(0, H - 1)
            xs_f = ((xs_d.float() + 0.5) * train_cfg.down - 0.5).long().clamp(0, W - 1)
            yy, xx = torch.meshgrid(ys_f, xs_f, indexing="ij")
            dep_all.append(d_full[yy, xx].reshape(-1))
            val_all.append(v_full[yy, xx].reshape(-1))
            nor_all.append(n_full[yy, xx].reshape(-1, 3))
        mvs_depth_flat  = torch.cat(dep_all).to(device)
        mvs_valid_flat  = torch.cat(val_all).to(device)
        mvs_normal_flat = torch.cat(nor_all).to(device)   # (total_rays, 3) cam-space
        _dv = mvs_depth_flat[mvs_valid_flat]
        if _dv.numel() > 2 ** 24:
            _dv = _dv[torch.randperm(_dv.numel(), device=_dv.device)[:2 ** 24]]
        depth_p95       = _dv.quantile(0.95)
        mvs_valid_flat  = mvs_valid_flat & (mvs_depth_flat < depth_p95)
        mvs_depth_maps  = mvs_depth_flat.reshape(V, H_d, W_d)
        mvs_valid_maps  = mvs_valid_flat.reshape(V, H_d, W_d)
        mvs_normal_maps = mvs_normal_flat.reshape(V, H_d, W_d, 3)
        print(f"  MVS depth: {mvs_valid_flat.sum()}/{total_rays} valid, p95={depth_p95:.3f}")

    if train_cfg.single_view >= 0:
        sv_mask = (det["vi"] == train_cfg.single_view)   # CPU
        print(f"  [single-view] view {train_cfg.single_view}: "
              f"{sv_mask.sum()} / {total_rays} rays")
        det             = {k: (v[sv_mask] if isinstance(v, torch.Tensor) else v)
                           for k, v in det.items()}
        sv_mask_dev     = sv_mask.to(device)
        mvs_depth_flat  = mvs_depth_flat[sv_mask_dev]
        mvs_valid_flat  = mvs_valid_flat[sv_mask_dev]
        mvs_normal_flat = mvs_normal_flat[sv_mask_dev]
        total_rays      = det["o"].shape[0]
        fg_idx = det["fg"].nonzero(as_tuple=True)[0]
        bg_idx = (~det["fg"]).nonzero(as_tuple=True)[0]

    alt_nn = precompute_alt_cameras(views, train_cfg.n_alt).to(device)
    print(f"  alt cameras: {train_cfg.n_alt} NN per view")

    # --- optional image-gradient-weighted fg sampling ---
    fg_cdf = None   # CDF for grad-weighted sampling (searchsorted, no 2^24 limit)
    if train_cfg.grad_weighted_sampling:
        H_d, W_d = views["H"] // train_cfg.down, views["W"] // train_cfg.down
        _ray_grad = _image_grad_ray_weights(det, H_d, W_d)        # (N,) cpu
        _g = _ray_grad[fg_idx]
        _g = _g / _g.sum().clamp(min=1e-12)
        _u = torch.full_like(_g, 1.0 / max(len(_g), 1))
        a  = float(train_cfg.grad_sampling_alpha)
        _p = a * _g + (1.0 - a) * _u
        _p = _p / _p.sum()
        fg_cdf = torch.cumsum(_p, dim=0)                          # (|fg|,) — reused each step
        print(f"  grad-weighted fg sampling: α={a:.2f}  "
              f"grad[min/mean/max]={_ray_grad[fg_idx].min():.4f}/"
              f"{_ray_grad[fg_idx].mean():.4f}/{_ray_grad[fg_idx].max():.4f}")

    # --- debug regions ---
    debug_region_data = _setup_debug_regions(train_cfg.debug_regions, views, device)

    # --- held-out log subset ---
    n_log       = min(4096, total_rays)
    log_idx     = torch.randperm(total_rays)[:n_log]          # CPU — used to index det
    log_idx_dev = log_idx.to(device)
    log_o   = det["o"][log_idx].to(device);  log_d  = det["d"][log_idx].to(device)
    log_vi  = det["vi"][log_idx].to(device); log_fg = det["fg"][log_idx].to(device)
    log_mvs_d = mvs_depth_flat[log_idx_dev]; log_mvs_v = mvs_valid_flat[log_idx_dev]
    z_cams_all  = c2w_all[:, :3, 2]
    log_cos     = (log_d * z_cams_all[log_vi]).sum(-1).abs().clamp(min=1e-6)
    log_t_target = log_mvs_d / log_cos

    # --- post-init sanity ---
    with torch.no_grad():
        test_o = origins_all[:5]
        print(f"  POST-INIT  f(cams[:5])={f(test_o).tolist()}")
        print(f"  POST-INIT  f(0,0,0)={f(torch.zeros(1,3,device=device)).item():.4f}")
        o_dbg = origins_all[0:1]
        d_dbg = -o_dbg / o_dbg.norm(dim=-1, keepdim=True)
        t_dbg = torch.tensor([0.0], device=device)
        print(f"  TRACE-DBG  o={o_dbg[0].tolist()}  f(o)={f(o_dbg).item():.4f}")
        for i in range(20):
            p = o_dbg + t_dbg.unsqueeze(-1) * d_dbg
            sv = f(p).item()
            print(f"    iter {i:2d}  t={t_dbg.item():.4f}  f(p)={sv:.4f}")
            if abs(sv) < 1e-3: print(f"    HIT"); break
            t_dbg = t_dbg + abs(sv)
            if t_dbg.item() > 10.0: print("    MISS"); break

    # --- optimiser ---
    opt       = torch.optim.Adam(f.parameters(), lr=train_cfg.lr)
    if cam_params is not None:
        opt.add_param_group({"params": list(cam_params.parameters()),
                             "lr": train_cfg.bundle.lr})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.steps,
                                                             eta_min=train_cfg.lr / 10)
    start_step = 0
    if _resume_ckpt is not None:
        start_step = int(_resume_ckpt.get("step", -1)) + 1
        if "opt" in _resume_ckpt:
            opt.load_state_dict(_resume_ckpt["opt"])
        if "scheduler" in _resume_ckpt:
            scheduler.load_state_dict(_resume_ckpt["scheduler"])
        else:
            for _ in range(start_step):
                scheduler.step()
        print(f"  [resume] continuing from step {start_step} / {train_cfg.steps}")
    best_score        = float("inf"); best_step = -1
    best_photo_score  = float("inf")
    best_loss_score   = float("inf")
    BEST_OUT       = ckpt_dir / "checkpoint_best_geo.pt"
    BEST_PHOTO_OUT = ckpt_dir / "checkpoint_best_photo.pt"
    BEST_LOSS_OUT  = ckpt_dir / "checkpoint_best_loss.pt"
    LATEST_OUT     = ckpt_dir / "checkpoint_latest.pt"
    LATEST_FREQ    = 500
    STEP_CKPT_FREQ = 10000
    NORMAL_DUMP_VIEWS  = [16, 32]   # hardcoded views for the periodic normal-map dump
    NORMAL_DUMP_MC_RES = 256        # 10k-step mesh preview; down=1 (full-res cameras)
    NORMAL_DUMP_COVERAGE_RES = 500  # target height for K_valid heatmaps
    photo_history: list[tuple[int, float]] = []

    # ─── profile: static accounting + always-on per-phase timing ─────────────
    _prof_dir = run_dir / "profile"
    dump_static_accounting(_prof_dir, f, model_cfg, train_cfg, trace_cfg)
    prof = StepProfiler(_prof_dir, flush_every=1000, rays_per_step=train_cfg.batch)
    mem_snap = MemorySnapshot(_prof_dir, at_step=max(start_step + 50, 200))

    # ------------------------------------------------------------------ loop --
    for step in range(start_step, train_cfg.steps):
        prof.step_begin()
        parts = []
        if n_fg > 0 and len(fg_idx) > 0:
            if fg_cdf is not None:
                fg_draw = torch.searchsorted(fg_cdf, torch.rand(n_fg))
            else:
                fg_draw = torch.randint(0, len(fg_idx), (n_fg,))
            parts.append(fg_idx[fg_draw])
        if n_bg > 0 and len(bg_idx) > 0:
            parts.append(bg_idx[torch.randint(0, len(bg_idx), (n_bg,))])
        idx      = torch.cat(parts)                            # CPU — indexes det on CPU
        idx_dev  = idx.to(device)
        vi       = det["vi"][idx].to(device);  gt = det["gt"][idx].to(device)
        fg_self  = det["fg"][idx].to(device)
        if cam_params is not None and step >= train_cfg.bundle.freeze_steps:
            # Refresh world-frame cameras from the current BA parameters and
            # rebuild this batch's rays so gradients flow into the extrinsics.
            c2w_all     = cam_params()
            origins_all = c2w_all[:, :3, 3]
            w2c_all     = torch.linalg.inv(c2w_all)
            px = det["px"][idx].to(device);  py = det["py"][idx].to(device)
            o, u = rays_from_pixels(c2w_all, K_all, px, py, vi)
        else:
            o = det["o"][idx].to(device);   u  = det["d"][idx].to(device)

        _need_eik = train_cfg.w_eikonal > 0 or train_cfg.w_mvs_sdf > 0 or train_cfg.mvsdf_schedule.enabled
        _trace_fn = trace_idr if trace_cfg.grad_mode == "idr" else trace_unrolled
        with prof.timed("trace"):
            x_theta, t, hit, eik_pts, n_raw, sdf_min, hit_bg = _trace_fn(f, o, u, trace_cfg,
                                                                          collect_eik=_need_eik,
                                                                          diff_normal=train_cfg.w_ncc_normal > 0)
        neus_trace_stats = get_last_trace_stats() if f.architecture == "neus" else None
        neus_trace_str = ""
        # hit_real = real convergence; hit_bg = reached bounding sphere exit.
        # For photo loss: include both (bg points are photometrically inconsistent → gradient).
        # For geometry losses (mask, eikonal, behind-hit): real hits only.
        hit_for_photo = hit | (hit_bg & fg_self)

        if step % 50 == 0:
            neus_trace_str = _format_neus_trace_stats(neus_trace_stats) if neus_trace_stats is not None else ""
            print(f"step {step:5d}  |∇f| mean={n_raw.norm(dim=-1).mean():.4f}{neus_trace_str}")
        n = n_raw / n_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # primary-camera uv (for NCC and photo)
        w2c_self  = w2c_all[vi]
        xc_self   = torch.einsum("bij,bj->bi", w2c_self[:, :3, :3], x_theta) + w2c_self[:, :3, 3]
        uv_h_self = torch.einsum("bij,bj->bi", K_all[vi], xc_self)
        uv_self   = uv_h_self[:, :2] / uv_h_self[:, 2:3].clamp(min=1e-6)

        # --- MVSDF weight schedule (see config.py MvsdfScheduleConfig) ---
        # wR (w_photo) and wE (w_eikonal) are always fixed per MVSDF paper.
        # Only wD (w_mvs_sdf) and wF (w_feature) follow the schedule.
        if train_cfg.mvsdf_schedule.enabled:
            _prog = step / max(train_cfg.steps - 1, 1)
            _eff_w_msdf, _eff_w_feat = train_cfg.mvsdf_schedule.weights(_prog)
            if step % 50 == 0:
                _ph = train_cfg.mvsdf_schedule.phase(_prog)
                print(f"  [schedule] phase={_ph}  w_msdf={_eff_w_msdf}  w_feat={_eff_w_feat}")
        else:
            _eff_w_msdf = train_cfg.w_mvs_sdf
            _eff_w_feat = train_cfg.w_feature
        _eff_w_photo = train_cfg.w_photo  # always fixed (wR)

        # gaussian sigma schedule: exponential decay sigma_start → sigma_end
        _sigma_start = train_cfg.gaussian_sigma
        _sigma_end   = train_cfg.gaussian_sigma_end
        if train_cfg.sample_mode == "gaussian" and _sigma_end != _sigma_start and _sigma_end > 0:
            _t = step / max(train_cfg.steps - 1, 1)
            _eff_sigma = _sigma_start * (_sigma_end / _sigma_start) ** _t
        else:
            _eff_sigma = _sigma_start
        _eff_radius = max(1, int(math.ceil(2.0 * _eff_sigma)))

        # spatial patch-weight α schedule (exp decay), independent of sample_mode
        _wsig_start = getattr(train_cfg, "ncc_patch_wsigma", 0.0)
        _wsig_end   = getattr(train_cfg, "ncc_patch_wsigma_end", 0.0)
        if _wsig_start > 0 and _wsig_end > 0 and _wsig_end != _wsig_start:
            _t = step / max(train_cfg.steps - 1, 1)
            _eff_wsigma = _wsig_start * (_wsig_end / _wsig_start) ** _t
        else:
            _eff_wsigma = _wsig_start

        # --- losses ---
        with prof.timed("photo_ncc"):
            ph, ph_stats = photo_loss(
                f, x_theta, hit_for_photo, n,
                vi, alt_nn, origins_all,
                images, K_all, w2c_all, feature_maps, masks, fg_self,
                H, W, uv_self,
                train_cfg.n_alt, train_cfg.cos_thresh,
                _eff_w_photo, _eff_w_feat, train_cfg.w_ncc, train_cfg.ncc_patch, train_cfg.ncc_half_pix,
                train_cfg.sample_mode, _eff_sigma, _eff_radius,
                step, train_cfg.ncc_min, train_cfg.occ_mode,
                hit_bg=hit_bg if trace_cfg.bsphere_radius > 0 else None,
                w_ncc_normal=train_cfg.w_ncc_normal,
                ncc_topk=train_cfg.ncc_topk,
                ncc_color=train_cfg.ncc_color,
                ncc_grad_alpha=train_cfg.ncc_grad_alpha,
                ncc_normal_patch=train_cfg.ncc_normal_patch,
                ncc_normal_half_pix=train_cfg.ncc_normal_half_pix,
                ncc_patch_wsigma=_eff_wsigma,
                trace_cfg=trace_cfg,
                prof=prof,
            )

        with prof.timed("eikonal"):
            eik  = (eikonal_loss(f, eik_pts, train_cfg.n_eik_vol, device)
                    if train_cfg.w_eikonal > 0 else torch.zeros(1, device=device).squeeze())
        cfr  = (cam_free_loss(f, o)
                if train_cfg.w_cam_free > 0 else torch.zeros(1, device=device).squeeze())

        if train_cfg.w_ray_free > 0 and hit.any():
            t_samp = torch.rand(hit.sum(), train_cfg.n_ray_free, device=device)
            t_samp = t_samp * t[hit].detach().unsqueeze(1)          # (n_hit, n_ray_free)
            pts_rf = o[hit].unsqueeze(1) + t_samp.unsqueeze(-1) * u[hit].unsqueeze(1)
            rf = F.relu(-f(pts_rf.reshape(-1, 3))).mean()
        elif train_cfg.w_ray_free > 0:
            # no hits this step — zero with grad_fn so backward stays valid
            rf = f(o[:1].detach()).sum() * 0.0
        else:
            rf = torch.zeros(1, device=device).squeeze()

        # normal supervision: compare predicted world-space normal with GT cam-space normal
        if train_cfg.w_normal > 0 and hit.any():
            hit_valid_n = hit & mvs_valid_flat[idx_dev] & fg_self
            if hit_valid_n.any():
                # rotate GT cam-space normal to world space: n_world = R_c2w @ n_cam
                R_c2w = c2w_all[vi[hit_valid_n], :3, :3]        # (M, 3, 3)
                n_cam = mvs_normal_flat[idx_dev[hit_valid_n]]    # (M, 3)
                n_gt  = torch.einsum("bij,bj->bi", R_c2w, n_cam)
                n_gt  = F.normalize(n_gt, dim=-1)
                n_pred = F.normalize(n_raw[hit_valid_n], dim=-1)
                nrm = (1.0 - (n_pred * n_gt).sum(-1).clamp(-1, 1)).mean()
            else:
                nrm = f(x_theta.detach()[:1]).sum() * 0.0
        else:
            nrm = torch.zeros(1, device=device).squeeze()

        loss = (ph + train_cfg.w_eikonal * eik
                + train_cfg.w_cam_free * cfr
                + train_cfg.w_ray_free * rf
                + train_cfg.w_normal * nrm)
        photo_history.append((step, ph.item()))

        opt.zero_grad(set_to_none=True)
        with prof.timed("backward"):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
        with prof.timed("opt_step"):
            opt.step(); scheduler.step()
        _trace_stats = get_last_trace_stats()
        if "mean_iters" in _trace_stats:
            prof.record_trace_iters(_trace_stats["mean_iters"])
        prof.step_end(step)
        mem_snap.tick(step)

        # --- periodic latest checkpoint (atomic, includes opt/scheduler for preemption) ---
        if step > 0 and step % LATEST_FREQ == 0:
            _ckpt_payload = {"f": f.state_dict(), "step": step,
                             "architecture": f.architecture, "group_size": f.group_size,
                             "depth": f.depth, "activation": f.activation,
                             "input_encoding": f.input_encoding,
                             "multires": f.multires,
                             "opt": opt.state_dict(),
                             "scheduler": scheduler.state_dict()}
            tmp = LATEST_OUT.with_suffix(".pt.tmp")
            torch.save(_ckpt_payload, tmp)
            tmp.replace(LATEST_OUT)

        # --- step checkpoint every STEP_CKPT_FREQ steps ---
        if step > 0 and step % STEP_CKPT_FREQ == 0:
            step_out = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save({"f": f.state_dict(), "step": step,
                        "architecture": f.architecture, "group_size": f.group_size,
                        "depth": f.depth, "activation": f.activation,
                        "input_encoding": f.input_encoding,
                        "multires": f.multires,
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict()}, step_out)
            print(f"  [ckpt] saved step checkpoint → {step_out.name}", flush=True)
            _dump_mc_normal_maps(f, views, NORMAL_DUMP_VIEWS, step, run_dir,
                                 device, mc_res=NORMAL_DUMP_MC_RES,
                                 bound=eval_cfg.bound(train_cfg.use_blender),
                                 trace_cfg=trace_cfg, train_cfg=train_cfg,
                                 alt_nn=alt_nn,
                                 coverage_res=NORMAL_DUMP_COVERAGE_RES)

        # --- logging every 50 steps ---
        if step % 50 == 0:
            hw_grad    = getattr(f, "head_weight", None)
            hw_grad    = hw_grad.grad if hw_grad is not None else None
            grad_norm  = hw_grad.norm().item() if hw_grad is not None else float("nan")
            with torch.no_grad():
                frac_fo_neg = (f(o) < 0).float().mean().item()
                frac_t_far  = (t >= 10.0 - 1e-3).float().mean().item()
                if hit.any():
                    _r = x_theta[hit].detach().norm(dim=-1)
                    xh_str = f"[{_r.min():.3f},{_r.mean():.3f},{_r.max():.3f}]"
                else:
                    xh_str = "[-]"
            pm = ph_stats
            photo_str = f"photo {ph.item():.4f}[w={_eff_w_photo}]"
            if _eff_w_feat > 0 or train_cfg.w_ncc > 0:
                photo_str += (f"  (l1 {pm['l1']:.4f}  feat {pm['feature']:.4f}[w={_eff_w_feat}]"
                              f"  ncc {pm['ncc']:.4f}[w={train_cfg.w_ncc}"
                              f" -> {pm.get('ncc_weighted', 0.0):.4f}])")
            if train_cfg.w_ncc > 0:
                _v = pm['ncc_valid']; _t = pm['ncc_textured']; _k = pm['ncc_kept']
                _kf = (_k / _t) if _t > 0 else 0.0
                _tf = (_t / _v) if _v > 0 else 0.0
                photo_str += f"  ncc[zncc={pm['ncc_zncc']:.3f} kept={_k}/{_t}/{_v} ({_kf:.2f}|{_tf:.2f})]"
                if train_cfg.ncc_topk > 0:
                    photo_str += (f"  ZNCC[mean{train_cfg.n_alt}="
                                  f"{pm.get('ncc_zncc_mean', 0.0):.3f} "
                                  f"top{train_cfg.ncc_topk}="
                                  f"{pm.get('ncc_zncc_topk', 0.0):.3f}]")
                if train_cfg.ncc_grad_alpha > 0:
                    _zi = pm.get('ncc_zncc_I', 0.0)
                    _zg = pm.get('ncc_zncc_grad', 0.0)
                    _a = train_cfg.ncc_grad_alpha
                    photo_str += (f"  grad[α={_a:.2f} I={_zi:.3f} ∇={_zg:.3f}"
                                  f" Δ={_zg - _zi:+.3f}"
                                  f" comb={(1 - _a) * _zi + _a * _zg:.3f}]")
            if train_cfg.w_ncc_normal > 0:
                _nv = pm.get('ncc_n_valid', 0); _nt = pm.get('ncc_n_textured', 0); _nk = pm.get('ncc_n_kept', 0)
                _nkf = (_nk / _nt) if _nt > 0 else 0.0
                _ntf = (_nt / _nv) if _nv > 0 else 0.0
                photo_str += (f"  ncc_normal {pm.get('ncc_normal', 0.0):.4f}"
                              f"[w={train_cfg.w_ncc_normal} -> {pm.get('ncc_normal_weighted', 0.0):.4f}]"
                              f" kept={_nk}/{_nt}/{_nv} ({_nkf:.2f}|{_ntf:.2f})")
            if (train_cfg.w_ncc > 0 or train_cfg.w_ncc_normal > 0) and step % 50 == 0:
                photo_str += (
                    f"  ncc_grad[ΔNCC/Δworld  "
                    f"pos(x)={pm.get('ncc_grad_pos', 0.0):.2e}/"
                    f"{pm.get('ncc_grad_pos_median', 0.0):.2e}/"
                    f"{pm.get('ncc_grad_pos_p90', 0.0):.2e}  "
                    f"nrm(n)={pm.get('ncc_grad_n_xeq', 0.0):.2e}/"
                    f"{pm.get('ncc_grad_n_xeq_median', 0.0):.2e}/"
                    f"{pm.get('ncc_grad_n_xeq_p90', 0.0):.2e}  mean/med/p90  "
                    f"n/x={pm.get('ncc_grad_ratio', 0.0):.2f}  "
                    f"(raw ∂n={pm.get('ncc_grad_n', 0.0):.2e}/rad)]")
            _hit_bg_str = (f"+bg{hit_bg.sum()}(ph:{ph_stats['n_mask_bg']})"
                           if trace_cfg.bsphere_radius > 0 else "")
            print(f"step {step:5d}  cams {vi.unique().numel():2d}  "
                  f"loss {loss.item():.4f}  {photo_str}  "
                  f"hit {hit.sum()}{_hit_bg_str}/{train_cfg.batch}  x_r {xh_str}  "
                  f"mask {pm['n_mask']}/{pm['n_in_frame']}/{pm['n_not_occl']}/"
                  f"{pm['n_cos_ok']}/{pm['n_total']}  "
                  f"f(o)<0 {frac_fo_neg:.2f}  t_far {frac_t_far:.2f}  "
                  f"rf {rf.item():.4f}  eik {eik.item():.4f}[w={train_cfg.w_eikonal}]  "
                  f"nrm {nrm.item():.4f}  ∇head {grad_norm:.6f}{neus_trace_str}")

            # geometry metrics on held-out subset
            with torch.no_grad():
                f_sfm_all = f(sfm_pts).abs()
                sfm_mean  = f_sfm_all.mean().item()
                sfm_p90   = torch.quantile(f_sfm_all, 0.9).item()
                _, t_log, hit_log = trace_nograd(f, log_o, log_d, trace_cfg)
                hr_log    = hit_log.float().mean().item()
                valid     = log_mvs_v & hit_log
                dt_str    = "n=0"
                dt_mean   = float("inf")
                if valid.any():
                    dt      = (t_log[valid] - log_t_target[valid]).abs()
                    dt_mean = dt.mean().item()
                    dt_str  = f"mean={dt_mean:.4f} p90={dt.quantile(.9):.4f} n={int(valid.sum())}"
                cams_out  = (f(origins_all) > 0).float().mean().item()

            sil_str = ""
            if log_fg.any():
                tp_s = (hit_log & log_fg).float().sum()
                fp_s = (hit_log & ~log_fg).float().sum()
                fn_s = (~hit_log & log_fg).float().sum()
                sil_iou  = (tp_s / (tp_s + fp_s + fn_s).clamp(1)).item()
                sil_prec = (tp_s / (tp_s + fp_s).clamp(1)).item()
                sil_rec  = (tp_s / (tp_s + fn_s).clamp(1)).item()
                sil_f1   = 2 * sil_prec * sil_rec / max(sil_prec + sil_rec, 1e-6)
                vol = (2 * torch.rand(1024, 3, device=device) - 1) * 2.0
                vol.requires_grad_(True)
                with torch.enable_grad():
                    gv = torch.autograd.grad(f(vol).sum(), vol)[0]
                gnorm     = gv.norm(dim=-1)
                grad_mean = gnorm.mean().item()
                grad_std  = gnorm.std().item()
                r_str = "n/a"; normal_cons = 0.0; n_ray = 0.0; depth_err_str = "n=0"
                if hit_log.any():
                    x_h  = log_o[hit_log] + t_log[hit_log].unsqueeze(-1) * log_d[hit_log]
                    r_str = f"{x_h.norm(dim=-1).mean():.3f}"
                    xr_h = x_h.detach().requires_grad_(True)
                    with torch.enable_grad():
                        nh = torch.autograd.grad(f(xr_h).sum(), xr_h)[0]
                    nh = nh / nh.norm(dim=-1, keepdim=True).clamp(1e-6)
                    ia = torch.randint(0, nh.shape[0], (min(512, nh.shape[0]),), device=device)
                    ib = torch.randint(0, nh.shape[0], (min(512, nh.shape[0]),), device=device)
                    normal_cons = (nh[ia] * nh[ib]).sum(-1).abs().mean().item()
                    # normal-ray angle: mean |cos(n, -d)| on hit rays (1=perfect facing)
                    n_ray = (nh * (-log_d[hit_log])).sum(-1).abs().mean().item()
                    # depth error vs MVS on hit+valid rays
                    hit_valid = hit_log & log_mvs_v
                    if hit_valid.any():
                        dt_hv = (t_log[hit_valid] - log_t_target[hit_valid]).abs()
                        depth_err_str = f"mean={dt_hv.mean():.4f} p90={dt_hv.quantile(.9):.4f}"
                    else:
                        depth_err_str = "n=0"
                sil_str = (f"  sil_IoU={sil_iou:.3f} P={sil_prec:.3f} R={sil_rec:.3f} "
                           f"F1={sil_f1:.3f}  |∇f|={grad_mean:.3f}±{grad_std:.3f}  "
                           f"n_cons={normal_cons:.3f}  n_ray={n_ray:.3f}  "
                           f"depth_err={depth_err_str}  r={r_str}")

            print(f"  [geom@{step:5d}]  |f(sfm)| mean={sfm_mean:.4f} p90={sfm_p90:.4f}  "
                  f"hit={hr_log:.2%}  mvs_|Δt| {dt_str}  cams_out={cams_out:.1%}{sil_str}")

            score = float("inf")
            if train_cfg.use_blender and log_fg.any():
                score = 1.0 - sil_iou
            elif valid.any():
                score = dt_mean

            if score < best_score:
                best_score = score; best_step = step
                torch.save({"f": f.state_dict(), "step": step, "score": score,
                            "architecture": f.architecture, "group_size": f.group_size,
                            "depth": f.depth, "activation": f.activation,
                            "input_encoding": f.input_encoding,
                            "multires": f.multires}, BEST_OUT)
                print(f"  [best_geo@{step}] score={score:.4f} → {BEST_OUT.name}")
                torch.cuda.empty_cache()
                _render_poses(f, views, step, run_dir, device, trace_cfg=trace_cfg)
                render_src = render_dir / f"render_{step:05d}.png"
                if render_src.exists():
                    shutil.copy(render_src, render_dir / "render_best_geo.png")

            # best_photo block disabled: with a single active loss (e.g. NCC-only,
            # w_photo=0), best_photo and best_loss track the same quantity, so
            # firing both was double-saving the ckpt and double-rendering 4
            # sphere-traced views every 50 steps. Keep best_loss only.

            _cur_loss = loss.item()
            loss_ckpt_ok = train_cfg.use_blender or hr_log > 0.01
            if _cur_loss < best_loss_score and loss_ckpt_ok:
                best_loss_score = _cur_loss
                torch.save({"f": f.state_dict(), "step": step, "loss": _cur_loss,
                            "architecture": f.architecture, "group_size": f.group_size,
                            "depth": f.depth, "activation": f.activation,
                            "input_encoding": f.input_encoding,
                            "multires": f.multires}, BEST_LOSS_OUT)
                print(f"  [best_loss@{step}] loss={_cur_loss:.4f} → {BEST_LOSS_OUT.name}")
                torch.cuda.empty_cache()
                _render_poses(f, views, step, run_dir, device, trace_cfg=trace_cfg)
                render_src = render_dir / f"render_{step:05d}.png"
                if render_src.exists():
                    shutil.copy(render_src, render_dir / "render_best_loss.png")
            elif _cur_loss < best_loss_score and not loss_ckpt_ok:
                print(f"  [best_loss@{step}] skipped: loss={_cur_loss:.4f} but hit={hr_log:.2%}")

            if debug_region_data and step % train_cfg.debug_region_freq == 0:
                _log_debug_regions(f, debug_region_data, trace_cfg, step, run_dir)

            if train_cfg.debug_views and step % train_cfg.debug_every == 0:
                _dbg_ids = [int(s) for s in train_cfg.debug_views.split(",") if s.strip()]
                _debug_views(f, views, _dbg_ids, step, run_dir, device,
                             trace_cfg, zoom_json=train_cfg.debug_zoom_json,
                             train_cfg=train_cfg, alt_nn=alt_nn)

            if train_cfg.single_view >= 0 and step % 500 == 0:
                H_d = H // train_cfg.down; W_d = W // train_cfg.down
                _render_residual_map(
                    f, det, train_cfg.single_view,
                    images, K_all, w2c_all, origins_all, alt_nn,
                    H, W, H_d, W_d,
                    train_cfg.n_alt, train_cfg.cos_thresh,
                    trace_cfg, run_dir, step,
                )

            cd = None
            sfm_surf = None
            dtu_official = None
            mvm_official = None
            if train_cfg.use_blender and gt_pts is not None and step % 500 == 0:
                cd = _mc_chamfer(f, gt_pts, device, bound=eval_cfg.bound_blender)
                if cd is not None:
                    print(f"  [chamfer@{step:5d}] sym={cd['chamfer']:.6f}  "
                          f"precision={cd['precision']:.6f}  completeness={cd['completeness']:.6f}")
                else:
                    print(f"  [chamfer@{step:5d}] n/a (surface not in bounds)")
            elif not train_cfg.use_blender:
                fast_dtu_due = eval_cfg.dtu_chamfer_freq > 0 and step % eval_cfg.dtu_chamfer_freq == 0
                official_due = (
                    eval_cfg.dtu_official_freq > 0 and step > 0
                    and step % eval_cfg.dtu_official_freq == 0
                    and dtu_scale_mat is not None and dtu_scan_id is not None
                    and eval_cfg.dtu_eval_dir is not None
                )
                if fast_dtu_due and train_cfg.w_sfm > 0 and sfm_pts.numel() > 3:
                    sfm_surf = _mc_sfm_surface_distance(
                        f, sfm_pts, device,
                        bound=eval_cfg.bound_dtu,
                        res=eval_cfg.dtu_chamfer_res,
                    )
                    if sfm_surf is not None:
                        print(f"  [sfm_surf@{step:5d}] mean={sfm_surf['mean']:.4f}  "
                              f"p50={sfm_surf['p50']:.4f}  p90={sfm_surf['p90']:.4f}  "
                              f"p99={sfm_surf['p99']:.4f}")
                    else:
                        print(f"  [sfm_surf@{step:5d}] n/a (surface not in bounds)")
                if fast_dtu_due and dtu_eval_data is not None:
                    cd = _mc_chamfer_dtu(f, dtu_eval_data, dtu_scale_mat, device,
                                         bound=eval_cfg.dtu_chamfer_bound, res=eval_cfg.dtu_chamfer_res)
                    if cd is not None:
                        print(f"  [dtu_chamfer@{step:5d}] sym={cd['chamfer']:.4f}  "
                              f"acc={cd['precision']:.4f}  comp={cd['completeness']:.4f}")
                    else:
                        print(f"  [dtu_chamfer@{step:5d}] n/a (surface not in bounds)")
                if official_due:
                    off_dir = run_dir / "dtu_official" / f"step_{step:06d}"
                    mesh_ply = _extract_world_mesh_for_dtu(
                        f, dtu_scale_mat, device, off_dir / "pred_world_mesh.ply",
                        bound=eval_cfg.dtu_official_bound,
                        res=eval_cfg.dtu_official_res,
                    )
                    if mesh_ply is None:
                        print(f"  [dtu_official@{step:5d}] n/a (surface not in bounds)")
                    else:
                        dtu_official = _run_dtu_official_eval(
                            mesh_ply, dtu_scan_id, eval_cfg.dtu_eval_dir, off_dir,
                        )
                        if dtu_official is not None:
                            print(f"  [dtu_official@{step:5d}] chamfer={dtu_official['chamfer']:.4f}mm  "
                                  f"acc={dtu_official['accuracy']:.4f}mm  "
                                  f"comp={dtu_official['completeness']:.4f}mm  out={off_dir}")
                # MVMannequin parallel path: same cadence, different eval recipe.
                mvm_official_due = (
                    eval_cfg.dtu_official_freq > 0 and step > 0
                    and step % eval_cfg.dtu_official_freq == 0 and is_mvm
                )
                if mvm_official_due:
                    off_dir = run_dir / "mvmannequin_official" / f"step_{step:06d}"
                    mvm_official = _run_mvmannequin_official_eval(
                        f, scene, off_dir,
                        bound=eval_cfg.dtu_official_bound,
                        res=eval_cfg.dtu_official_res,
                        device=device,
                    )
                    if mvm_official is not None:
                        print(f"  [mvm_official@{step:5d}] chamfer={mvm_official['chamfer']:.4f}mm  "
                              f"acc={mvm_official['accuracy']:.4f}mm  "
                              f"comp={mvm_official['completeness']:.4f}mm  out={off_dir}")

            if use_wandb:
                import wandb
                log = {"loss": loss.item(), "photo": ph.item(),
                       "photo_l1": ph_stats["l1"], "photo_ncc": ph_stats["ncc"],
                       "photo_ncc_weighted": ph_stats.get("ncc_weighted", 0.0),
                       "photo_ncc_normal": ph_stats.get("ncc_normal", 0.0),
                       "photo_ncc_normal_weighted": ph_stats.get("ncc_normal_weighted", 0.0),
                       "photo_ncc_zncc": ph_stats.get("ncc_zncc", 0.0),
                       "photo_ncc_zncc_mean": ph_stats.get("ncc_zncc_mean", 0.0),
                       "photo_ncc_zncc_topk": ph_stats.get("ncc_zncc_topk", 0.0),
                       "photo_ncc_zncc_I": ph_stats.get("ncc_zncc_I", 0.0),
                       "photo_ncc_zncc_grad": ph_stats.get("ncc_zncc_grad", 0.0),
                       "photo_ncc_kept_frac": (ph_stats.get("ncc_kept", 0) / max(ph_stats.get("ncc_textured", 0), 1)),
                       "photo_ncc_textured_frac": (ph_stats.get("ncc_textured", 0) / max(ph_stats.get("ncc_valid", 0), 1)),
                       "photo_ncc_normal_kept_frac": (ph_stats.get("ncc_n_kept", 0) / max(ph_stats.get("ncc_n_textured", 0), 1)),
                       "photo_ncc_normal_textured_frac": (ph_stats.get("ncc_n_textured", 0) / max(ph_stats.get("ncc_n_valid", 0), 1)),
                       "photo_feature": ph_stats["feature"],
                       "hit_rate": hit.float().mean().item(), "cams_out": cams_out,
                       "grad_norm_mean": grad_mean, "sil_iou": sil_iou,
                       "best_geo_score": best_score}
                if torch.cuda.is_available():
                    log["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
                    torch.cuda.reset_peak_memory_stats(device)
                if cd is not None:
                    log["chamfer"] = cd["chamfer"]
                    log["chamfer_acc"] = cd["precision"]
                    log["chamfer_comp"] = cd["completeness"]
                if sfm_surf is not None:
                    log["sfm_surf_mean"] = sfm_surf["mean"]
                    log["sfm_surf_p50"] = sfm_surf["p50"]
                    log["sfm_surf_p90"] = sfm_surf["p90"]
                    log["sfm_surf_p99"] = sfm_surf["p99"]
                if dtu_official is not None:
                    log["dtu_official_chamfer"] = dtu_official["chamfer"]
                    log["dtu_official_acc"] = dtu_official["accuracy"]
                    log["dtu_official_comp"] = dtu_official["completeness"]
                if mvm_official is not None:
                    log["mvm_official_chamfer"] = mvm_official["chamfer"]
                    log["mvm_official_acc"] = mvm_official["accuracy"]
                    log["mvm_official_comp"] = mvm_official["completeness"]
                if (render_dir / "render_{:05d}.png".format(step)).exists():
                    log["render"] = wandb.Image(str(render_dir / "render_{:05d}.png".format(step)))
                wandb.log(log, step=step)

    # --- save final ---
    final_payload = {"f": f.state_dict(), "architecture": f.architecture,
                     "group_size": f.group_size, "depth": f.depth,
                     "activation": f.activation,
                     "input_encoding": f.input_encoding,
                     "multires": f.multires, "step": step}
    torch.save(final_payload, OUT)
    FINAL_OUT = ckpt_dir / "checkpoint_final.pt"
    torch.save(final_payload, FINAL_OUT)
    print(f"saved → {OUT}")
    print(f"final → {FINAL_OUT.name}")
    try:
        _render_poses(f, views, step, run_dir, device, trace_cfg=trace_cfg)
        final_render = render_dir / f"render_{step:05d}.png"
        if final_render.exists():
            shutil.copy(final_render, render_dir / "render_final.png")
            print(f"final render → render_final.png")
    except Exception as e:
        print(f"  [final render] skipped: {e}")
    if best_step >= 0:
        print(f"best_geo:   step {best_step}  score {best_score:.4f}  → {BEST_OUT.name}")
        print(f"best_photo: photo {best_photo_score:.4f}              → {BEST_PHOTO_OUT.name}")

    # --- photo loss curve ---
    if photo_history:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps_h, vals_h = zip(*photo_history)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps_h, vals_h, linewidth=0.8)
        ax.set_xlabel("step"); ax.set_ylabel("photo loss")
        ax.set_title(f"photo loss — hidden={f.hidden} depth={f.depth} gs={f.group_size}")
        ax.grid(True, alpha=0.3)
        plot_out = diag_dir / "photo_loss.png"
        fig.savefig(plot_out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"photo loss plot → {plot_out}")

    return OUT


# ---------- entry point ----------

if __name__ == "__main__":
    import argparse

    _tc = TrainConfig()
    _mc = ModelConfig()
    ap = argparse.ArgumentParser(description="1-Lip sphere-tracing trainer")
    ap.add_argument("--no-train",   action="store_true", help="skip training")
    ap.add_argument("--steps",      type=int,   default=_tc.steps)
    ap.add_argument("--batch",      type=int,   default=_tc.batch)
    ap.add_argument("--lr",         type=float, default=_tc.lr)
    ap.add_argument("--down",       type=int,   default=_tc.down)
    ap.add_argument("--grad-weighted-sampling", action="store_true",
                    default=_tc.grad_weighted_sampling,
                    help="sample fg rays ∝ image-gradient magnitude (fine-detail focus)")
    ap.add_argument("--grad-sampling-alpha", type=float,
                    default=_tc.grad_sampling_alpha,
                    help="grad/uniform mix for --grad-weighted-sampling (0=uniform, 1=pure grad)")
    ap.add_argument("--blender",    action="store_true", help="use a Blender synthetic dataset")
    ap.add_argument("--dataset",    choices=["dtu", "skull", "lego"], default=None)
    ap.add_argument("--scene",      type=Path, default=None,
                    help="override scene path (useful for non-default Blender objects)")
    ap.add_argument("--config",     type=Path, default=None,
                    help="load the full training config from a saved run config.json")
    ap.add_argument("--hidden",     type=int,   default=_mc.hidden)
    ap.add_argument("--depth",      type=int,   default=_mc.depth)
    ap.add_argument("--group-size", type=int,   default=_mc.group_size)
    ap.add_argument("--activation", type=str,   default=_mc.activation,
                    choices=["groupsort", "nact"], help="activation: groupsort or nact (N-Activation)")
    ap.add_argument("--input-encoding", type=str, default=_mc.input_encoding,
                    choices=["identity", "pe"], help="input encoding before the 1-Lipschitz backbone")
    ap.add_argument("--architecture", type=str, default=_mc.architecture, choices=["cpl", "mlp", "neus"],
                    help="cpl: 1-Lipschitz CPL (default); neus: Softplus+skip MLP; mlp: plain ReLU MLP")
    ap.add_argument("--multires", type=int, default=_mc.multires,
                    help="number of positional encoding frequencies (input_encoding=pe)")
    _trc = TraceConfig()
    ap.add_argument("--trace-iters", type=int, default=_trc.iters,
                    help="max sphere-tracing iterations (increase when using PE)")
    ap.add_argument("--bsphere-radius", type=float, default=_trc.bsphere_radius,
                    help=">0: use per-ray bounding-sphere exit as t_far; rays that reach "
                         "the sphere become hit_bg and participate in photo loss with "
                         "photometrically inconsistent colors → gradient fills holes. "
                         "Set to ~1.5× the object bounding-sphere radius.")
    ap.add_argument("--t-far", type=float, default=_trc.t_far,
                    help="global ray cut-off distance (used when --bsphere-radius=0). "
                         "Must exceed max camera distance + object radius.")
    _ic = InitConfig()
    ap.add_argument("--init",       type=str,   default=_ic.init,
                    choices=["sphere", "hull", "colmap"],
                    help="warm-start: sphere, silhouette visual hull, or "
                         "voxelisation of a COLMAP dense mesh (needs --init-mesh)")
    ap.add_argument("--init-mesh",  type=str,   default=_ic.init_mesh,
                    help="path to a triangle mesh in NSVF-COLMAP frame "
                         "(e.g. outputs/colmap_barn/poisson.ply); only used "
                         "when --init colmap")
    ap.add_argument("--init-steps", type=int,   default=_ic.steps,
                    help="gradient steps for the hull/sphere warm-start")
    ap.add_argument("--radius",     type=float, default=_ic.radius,
                    help="sphere-init radius (None = auto from COLMAP p60)")
    ap.add_argument("--hull-res",   type=int,   default=_ic.hull_res,
                    help="voxel resolution for hull carving")
    ap.add_argument("--w-depth-surface", type=float, default=_ic.w_depth_surface,
                    help="blender hull-init only: weight on GT depth surface samples")
    ap.add_argument("--w-photo",    type=float, default=_tc.w_photo)
    ap.add_argument("--w-feature",  type=float, default=_tc.w_feature,
                    help="weight for cosine distance on precomputed feature maps")
    ap.add_argument("--feature-maps", type=Path, default=_tc.feature_maps,
                    help="path to a .pt file produced by precompute_mast3r_features.py")
    ap.add_argument("--n-alt",      type=int,   default=_tc.n_alt,
                    help="nearest-neighbour alt cameras per ray (pool size; "
                         "e.g. 10 with --ncc-topk 4, or 6 for the legacy pool)")
    ap.add_argument("--w-ncc",      type=float, default=_tc.w_ncc)
    ap.add_argument("--w-ncc-normal", type=float, default=_tc.w_ncc_normal,
                    help="weight of the normal-branch PMVS NCC term "
                         "L=w_ncc·NCC(x,detach(n))+w_ncc_normal·NCC(detach(x),n); "
                         ">0 enables a differentiable normal (double-backward)")
    ap.add_argument("--ncc-half-pix", type=float, default=_tc.ncc_half_pix,
                    help="PMVS patch half-width in reference-view pixels")
    ap.add_argument("--ncc-normal-patch", type=int, default=_tc.ncc_normal_patch,
                    help="normal-branch patch P (<0 → share --ncc-patch); the "
                         "normal-branch ZNCC leverage scales with patch extent, "
                         "so it usually wants a larger patch, e.g. 9 or 11")
    ap.add_argument("--ncc-normal-half-pix", type=float, default=_tc.ncc_normal_half_pix,
                    help="normal-branch patch half-width in px (<0 → share "
                         "--ncc-half-pix); pair with --ncc-normal-patch≈2·hp+1")
    ap.add_argument("--ncc-min", type=float, default=_tc.ncc_min,
                    help="PMVS photometric gate: drop pairs with ZNCC below this")
    ap.add_argument("--ncc-topk", type=int, default=_tc.ncc_topk,
                    help="0: mean over all valid alt views; >0: per-point top-K "
                         "best ZNCC across the n_alt pool (robust MVS, use 3-4 "
                         "with n_alt~10)")
    ap.add_argument("--ncc-color", type=str, default=_tc.ncc_color,
                    choices=["gray", "rgb"],
                    help="ZNCC on Rec.601 luminance (gray, DTU default) or "
                         "per-channel RGB averaged (rgb, legacy)")
    ap.add_argument("--ncc-grad-alpha", type=float, default=_tc.ncc_grad_alpha,
                    help="0: intensity ZNCC only; >0 (≈0.3-0.5): blend a "
                         "gradient-magnitude ZNCC (Gipuma-style edge term)")
    ap.add_argument("--sample-mode", type=str, default=_tc.sample_mode,
                    choices=["bilinear", "gaussian"],
                    help="image sampler for photo and NCC losses")
    ap.add_argument("--gaussian-sigma",      type=float, default=_tc.gaussian_sigma)
    ap.add_argument("--gaussian-sigma-end",  type=float, default=_tc.gaussian_sigma_end,
                    help="anneal sigma exponentially to this value by end of training (default: no anneal)")
    ap.add_argument("--gaussian-radius", type=int,   default=_tc.gaussian_radius)
    ap.add_argument("--ncc-patch-wsigma", type=float, default=_tc.ncc_patch_wsigma,
                    help="spatial patch weight α in w=exp(-r/α) (grid units); 0=uniform/legacy")
    ap.add_argument("--ncc-patch-wsigma-end", type=float, default=_tc.ncc_patch_wsigma_end,
                    help="anneal patch-weight α exponentially to this by end of training")
    ap.add_argument("--debug-views",    type=str,   default=_tc.debug_views,
                    help='opt-in thorough per-view hole diagnostics, e.g. "16,32"')
    ap.add_argument("--debug-every",    type=int,   default=_tc.debug_every,
                    help="steps between --debug-views dumps")
    ap.add_argument("--debug-zoom-json", type=str,  default=_tc.debug_zoom_json,
                    help='JSON of per-view region boxes for zoom crops')
    ap.add_argument("--w-idr-mask",     type=float, default=_tc.w_idr_mask,
                    help="IDR mask loss weight (Yariv 2020); paper uses ρ=100")
    ap.add_argument("--idr-n-samples",  type=int,   default=_tc.idr_n_samples,
                    help="uniform ray samples for IDR hard min_t f (paper: 100)")
    ap.add_argument("--sil-s",           type=float, default=_tc.sil_s,
                    help="starting α for idr_mask/sil; IDR=50")
    ap.add_argument("--sil-s-interval", type=int,   default=_tc.sil_s_interval,
                    help="steps between α doublings (0=fixed); IDR doubles every 250 epochs")
    ap.add_argument("--sil-s-max-mults", type=int,  default=_tc.sil_s_max_mults,
                    help="max doublings of α (IDR=5 → α_max=1600)")
    ap.add_argument("--w-sil",          type=float, default=_tc.w_sil)
    ap.add_argument("--sil-fg-offset",  type=float, default=_tc.sil_fg_offset,
                    help="sdf_min shift for fg rays in mask loss; pushes σ away from 0.5 on hits")
    ap.add_argument("--sil-bg-offset",  type=float, default=_tc.sil_bg_offset,
                    help="symmetric offset for bg rays so near-surface bg gets low loss (mask-noise tolerance)")
    ap.add_argument("--sil-focal-gamma", type=float, default=_tc.sil_focal_gamma,
                    help="focal weighting (1-p_correct)^γ; γ=2 concentrates gradient on missing-piece rays")
    ap.add_argument("--sil-no-balance", action="store_true",
                    help="disable class balancing (default: rescale fg/bg to 1/class_frac)")
    ap.add_argument("--sil-norm-alpha", action="store_true",
                    help="divide loss by α (legacy IDR scaling); off by default so α scheduling actually bites")
    _trc = TraceConfig()
    ap.add_argument("--sdf-min-beta",   type=float, default=_trc.sdf_min_beta,
                    help="β for soft-min over trace iterations (0 = hard min)")
    ap.add_argument("--w-mask-fg", type=float, default=_tc.w_mask_fg,
                    help="DVR-style foreground mask loss weight")
    ap.add_argument("--w-mask-bg", type=float, default=_tc.w_mask_bg,
                    help="DVR-style background free-space loss weight")
    ap.add_argument("--mask-fg-margin", type=float, default=_tc.mask_fg_margin,
                    help="fg rays: min SDF along ray must be <= this (0 = surface touched)")
    ap.add_argument("--mask-bg-margin", type=float, default=_tc.mask_bg_margin,
                    help="bg rays: all sampled SDFs must be >= this (free-space clearance)")
    ap.add_argument("--n-mask-fg", type=int, default=_tc.n_mask_fg,
                    help="stratified samples per foreground ray for DVR-style mask loss")
    ap.add_argument("--n-mask-bg", type=int, default=_tc.n_mask_bg,
                    help="stratified samples per background ray for DVR-style free-space loss")
    ap.add_argument("--w-eikonal",  type=float, default=_tc.w_eikonal)
    ap.add_argument("--n-eik-vol",  type=int, default=_tc.n_eik_vol,
                    help="random volume points for eikonal loss, separate from trace samples")
    ap.add_argument("--w-sfm",          type=float, default=_tc.w_sfm)
    ap.add_argument("--sfm-min-views",  type=int,   default=_tc.sfm_min_views,
                    help="filter COLMAP pts visible in fewer cameras (0=keep all)")
    ap.add_argument("--sfm-behind-eps", type=float, default=_tc.sfm_behind_eps,
                    help="step behind SFM point along camera ray; require f <= 0 there (0=disabled)")
    ap.add_argument("--w-free",     type=float, default=_tc.w_free)
    ap.add_argument("--w-surf",     type=float, default=_tc.w_surf)
    ap.add_argument("--w-mvs",      type=float, default=_tc.w_mvs)
    ap.add_argument("--mvs-depth-dir", type=Path, default=None,
                    help="directory with MASt3R depth manifest.json (IDR scans); "
                         "enables load_mast3r_depths_idr instead of load_aligned_depths")
    ap.add_argument("--mvsformer-depth-dir", type=Path, default=None,
                    help="directory with MVSFormer++ depths (depth_est/*.pfm + confidence/*.npy); "
                         "takes priority over --mvs-depth-dir")
    ap.add_argument("--mvsformer-conf-thr", type=float, default=_tc.mvsformer_conf_thr,
                    help="confidence threshold for MVSFormer++ valid mask (default 0.5)")
    _sc = MvsdfScheduleConfig()
    ap.add_argument("--mvsdf-schedule",    action="store_true",   help="enable MVSDF 3-phase weight schedule")
    ap.add_argument("--schedule-p1-end",   type=float, default=_sc.phase1_end,  help="phase 1 end (fraction of steps, default 1/6)")
    ap.add_argument("--schedule-p2-end",   type=float, default=_sc.phase2_end,  help="phase 2 end (fraction of steps, default 1/2)")
    ap.add_argument("--schedule-p1-msdf",  type=float, default=_sc.p1_w_msdf,   help="phase 1 w_msdf (wD)")
    ap.add_argument("--schedule-p1-feat",  type=float, default=_sc.p1_w_feat,   help="phase 1 w_feat (wF)")
    ap.add_argument("--schedule-p2-msdf",  type=float, default=_sc.p2_w_msdf,   help="phase 2 w_msdf (wD)")
    ap.add_argument("--schedule-p2-feat",  type=float, default=_sc.p2_w_feat,   help="phase 2 w_feat (wF)")
    ap.add_argument("--schedule-p3-msdf",  type=float, default=_sc.p3_w_msdf,   help="phase 3 w_msdf (wD)")
    ap.add_argument("--schedule-p3-feat",  type=float, default=_sc.p3_w_feat,   help="phase 3 w_feat (wF)")
    ap.add_argument("--w-mvs-sdf",  type=float, default=_tc.w_mvs_sdf,
                    help="MVSDF carving loss weight (overridden by --mvsdf-schedule)")
    ap.add_argument("--n-mvs-sdf",  type=int,   default=_tc.n_mvs_sdf)
    ap.add_argument("--mvs-sdf-out-thresh", type=float, default=_tc.mvs_sdf_out_thresh,
                    help="fraction of views that must agree 'outside' to label a point outside (0.5=majority)")
    ap.add_argument("--mvs-sdf-trunc", type=float, default=_tc.mvs_sdf_trunc)
    ap.add_argument("--mvs-sdf-smooth", type=float, default=_tc.mvs_sdf_smooth,
                    help="SmoothL1 beta-like scale; 0 uses L1")
    ap.add_argument("--mvs-sdf-far-thresh", type=float, default=_tc.mvs_sdf_far_thresh)
    ap.add_argument("--mvs-sdf-far-att", type=float, default=_tc.mvs_sdf_far_att)
    ap.add_argument("--mvs-sdf-near-thresh", type=float, default=_tc.mvs_sdf_near_thresh)
    ap.add_argument("--mvs-sdf-near-att", type=float, default=_tc.mvs_sdf_near_att)
    ap.add_argument("--w-normal",      type=float, default=_tc.w_normal)
    ap.add_argument("--w-behind-hit",  type=float, default=_tc.w_behind_hit)
    ap.add_argument("--behind-eps",    type=float, default=_tc.behind_eps)
    ap.add_argument("--w-ray-free",    type=float, default=_tc.w_ray_free)
    ap.add_argument("--n-ray-free",    type=int,   default=_tc.n_ray_free)
    ap.add_argument("--single-view", type=int, default=_tc.single_view,
                    help="restrict training to this view index (-1=all, overfit diagnostic)")
    ap.add_argument("--pt",         default=None, help="checkpoint to evaluate")
    ap.add_argument("--resume",     type=Path, default=None,
                    help="resume training from this checkpoint (skips init)")
    ap.add_argument("--dtu-eval-dir", default=None,
                    help="path to DTU evaluation data (SampleSet/ + ObsMask/ subdirs)")
    _ec = EvalConfig()
    ap.add_argument("--dtu-chamfer-freq", type=int, default=_ec.dtu_chamfer_freq,
                    help="fast in-training DTU Chamfer frequency in steps (0=off)")
    ap.add_argument("--dtu-chamfer-res", type=int, default=_ec.dtu_chamfer_res,
                    help="MC resolution for fast in-training DTU Chamfer")
    ap.add_argument("--dtu-official-freq", type=int, default=_ec.dtu_official_freq,
                    help="DTUeval-python official Chamfer frequency in steps (0=off)")
    ap.add_argument("--dtu-official-res", type=int, default=_ec.dtu_official_res,
                    help="MC resolution for periodic DTUeval-python official Chamfer")
    ap.add_argument("--dtu-official-bound", type=float, default=_ec.dtu_official_bound,
                    help="MC bound for periodic DTUeval-python official Chamfer")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="explicit run directory (overrides auto-timestamped name); ignored on --resume")
    ap.add_argument("--viewer",      action="store_true")
    ap.add_argument("--viewer-res",  type=int, default=256)
    ap.add_argument("--profile", action="store_true",
                    help="one-shot compute/memory breakdown of the model at startup")
    ap.add_argument("--viewer-port", type=int, default=8080)
    ap.add_argument("--render-down", type=int, default=1,
                    help="downsample for sphere-traced PNG renders (1=full res, 2=half)")
    ap.add_argument("--mc-res",           type=int,   default=256,
                    help="marching-cubes grid resolution for eval/viewer")
    ap.add_argument("--dtu-chamfer-bound", type=float, default=_ec.dtu_chamfer_bound,
                    help="MC bound for in-training DTU Chamfer (tighter than SDF grid for better voxel precision)")
    ap.add_argument("--wandb",       action="store_true", help="log to Weights & Biases")
    ap.add_argument("--debug-regions", type=str, default="",
                    help='semicolon-separated "name,view,u0,v0,u1,v1" pixel rectangles to monitor during training')
    ap.add_argument("--debug-region-freq", type=int, default=_tc.debug_region_freq,
                    help="log debug regions every N steps (default 500)")
    args = ap.parse_args()

    if args.resume is not None and args.config is None:
        resume_run_dir = args.resume.parent.parent if args.resume.parent.name == "ckpt" else args.resume.parent
        resume_config = resume_run_dir / "config.json"
        if not resume_config.exists():
            raise SystemExit(f"--resume requires the checkpoint run config: missing {resume_config}")
        args.config = resume_config
        print(f"  inferred resume config: {args.config}")

    if args.config is not None:
        run_cfg = load_config_json(args.config)
        if args.scene is not None:
            run_cfg.scene = args.scene
        run_cfg = dataclasses.replace(
            run_cfg,
            eval=dataclasses.replace(
                run_cfg.eval,
                dtu_eval_dir=Path(args.dtu_eval_dir) if args.dtu_eval_dir else run_cfg.eval.dtu_eval_dir,
                dtu_chamfer_freq=args.dtu_chamfer_freq,
                dtu_chamfer_res=args.dtu_chamfer_res,
                dtu_chamfer_bound=args.dtu_chamfer_bound,
                dtu_official_freq=args.dtu_official_freq,
                dtu_official_res=args.dtu_official_res,
                dtu_official_bound=args.dtu_official_bound,
            ),
        )
        if args.profile:
            run_cfg = dataclasses.replace(
                run_cfg, train=dataclasses.replace(run_cfg.train, profile=True))
        if args.debug_regions:
            run_cfg = dataclasses.replace(run_cfg,
                train=dataclasses.replace(run_cfg.train,
                    debug_regions=args.debug_regions,
                    debug_region_freq=args.debug_region_freq))
    else:
        if args.dataset == "lego":
            args.blender = True
        elif args.dataset in ("dtu", "skull"):
            args.blender = False

        scene_path = args.scene
        if scene_path is None:
            scene_path = BLENDER_SCENE if args.blender else SCENE

        run_cfg = Config(
            model=ModelConfig(hidden=args.hidden, depth=args.depth, group_size=args.group_size,
                              activation=args.activation,
                              input_encoding=args.input_encoding,
                              multires=args.multires,
                              architecture=args.architecture),
            trace=TraceConfig(iters=args.trace_iters, bsphere_radius=args.bsphere_radius,
                              t_far=args.t_far, sdf_min_beta=args.sdf_min_beta),
            init=InitConfig(
                init=args.init,
                steps=args.init_steps,
                radius=args.radius,
                hull_res=args.hull_res,
                init_mesh=args.init_mesh,
            ),
            train=TrainConfig(
                steps=args.steps, batch=args.batch, lr=args.lr, down=args.down,
                profile=args.profile,
                grad_weighted_sampling=args.grad_weighted_sampling,
                grad_sampling_alpha=args.grad_sampling_alpha,
                use_blender=args.blender, single_view=args.single_view,
                w_photo=args.w_photo, w_feature=args.w_feature, feature_maps=args.feature_maps,
                n_alt=args.n_alt,
                w_ncc=args.w_ncc, w_ncc_normal=args.w_ncc_normal,
                ncc_half_pix=args.ncc_half_pix, ncc_min=args.ncc_min,
                ncc_topk=args.ncc_topk,
                ncc_color=args.ncc_color,
                ncc_grad_alpha=args.ncc_grad_alpha,
                ncc_normal_patch=args.ncc_normal_patch,
                ncc_normal_half_pix=args.ncc_normal_half_pix,
                sample_mode=args.sample_mode,
                gaussian_sigma=args.gaussian_sigma, gaussian_sigma_end=args.gaussian_sigma_end,
                gaussian_radius=args.gaussian_radius,
                ncc_patch_wsigma=args.ncc_patch_wsigma,
                ncc_patch_wsigma_end=args.ncc_patch_wsigma_end,
                debug_views=args.debug_views, debug_every=args.debug_every,
                debug_zoom_json=args.debug_zoom_json,
                w_idr_mask=args.w_idr_mask, idr_n_samples=args.idr_n_samples,
                sil_s=args.sil_s, sil_s_interval=args.sil_s_interval, sil_s_max_mults=args.sil_s_max_mults,
                w_sil=args.w_sil, sil_fg_offset=args.sil_fg_offset,
                sil_bg_offset=args.sil_bg_offset,
                sil_focal_gamma=args.sil_focal_gamma,
                sil_balance=not args.sil_no_balance,
                sil_norm_alpha=args.sil_norm_alpha,
                w_mask_fg=args.w_mask_fg, w_mask_bg=args.w_mask_bg,
                mask_fg_margin=args.mask_fg_margin, mask_bg_margin=args.mask_bg_margin,
                n_mask_fg=args.n_mask_fg, n_mask_bg=args.n_mask_bg,
                w_eikonal=args.w_eikonal, n_eik_vol=args.n_eik_vol, w_sfm=args.w_sfm,
                sfm_min_views=args.sfm_min_views, sfm_behind_eps=args.sfm_behind_eps, w_free=args.w_free,
                w_surf=args.w_surf, w_mvs=args.w_mvs, mvs_depth_dir=args.mvs_depth_dir,
                mvsformer_depth_dir=args.mvsformer_depth_dir,
                mvsformer_conf_thr=args.mvsformer_conf_thr,
                mvsdf_schedule=MvsdfScheduleConfig(
                    enabled=args.mvsdf_schedule,
                    phase1_end=args.schedule_p1_end,  phase2_end=args.schedule_p2_end,
                    p1_w_msdf=args.schedule_p1_msdf,  p1_w_feat=args.schedule_p1_feat,
                    p2_w_msdf=args.schedule_p2_msdf,  p2_w_feat=args.schedule_p2_feat,
                    p3_w_msdf=args.schedule_p3_msdf,  p3_w_feat=args.schedule_p3_feat,
                ),
                w_mvs_sdf=args.w_mvs_sdf, n_mvs_sdf=args.n_mvs_sdf,
                mvs_sdf_out_thresh=args.mvs_sdf_out_thresh,
                mvs_sdf_trunc=args.mvs_sdf_trunc, mvs_sdf_smooth=args.mvs_sdf_smooth,
                mvs_sdf_far_thresh=args.mvs_sdf_far_thresh, mvs_sdf_far_att=args.mvs_sdf_far_att,
                mvs_sdf_near_thresh=args.mvs_sdf_near_thresh, mvs_sdf_near_att=args.mvs_sdf_near_att,
                w_normal=args.w_normal,
                w_behind_hit=args.w_behind_hit, behind_eps=args.behind_eps,
                w_ray_free=args.w_ray_free, n_ray_free=args.n_ray_free,
                debug_regions=args.debug_regions,
                debug_region_freq=args.debug_region_freq,
            ),
            eval=EvalConfig(
                dtu_eval_dir=Path(args.dtu_eval_dir) if args.dtu_eval_dir else None,
                dtu_chamfer_freq=args.dtu_chamfer_freq,
                dtu_chamfer_res=args.dtu_chamfer_res,
                dtu_chamfer_bound=args.dtu_chamfer_bound,
                dtu_official_freq=args.dtu_official_freq,
                dtu_official_res=args.dtu_official_res,
                dtu_official_bound=args.dtu_official_bound,
            ),
            scene=scene_path,
        )

    ckpt_path: Path | None = None
    if not args.no_train:
        ckpt_path = train(run_cfg, use_wandb=args.wandb, resume=args.resume,
                          run_dir=args.run_dir)

    if args.pt is not None:
        ckpt_path = Path(args.pt)
    elif ckpt_path is None:
        # fall back to best available in output dir
        for name in ("checkpoint_best_photo.pt", "checkpoint_best_geo.pt", "checkpoint.pt"):
            cand = OUT_DIR / name
            if cand.exists():
                ckpt_path = cand; break

    if ckpt_path is None:
        print("No checkpoint found — run without --no-train first."); sys.exit(1)

    print(f"loading {ckpt_path}")
    ckpt       = torch.load(ckpt_path, map_location="cpu")
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    architecture = ckpt.get("architecture", args.architecture)
    if architecture == "neus":
        hidden = ckpt["f"]["layers.0.weight"].shape[0]
    elif "head_weight" in ckpt["f"]:
        hidden = ckpt["f"]["head_weight"].shape[0]
    else:
        # RegularMLP: last Linear layer maps hidden→1
        hidden = next(v.shape[1] for k, v in ckpt["f"].items()
                      if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
                      and not k.startswith("encoder"))
    group_size = ckpt.get("group_size", args.group_size)
    activation = ckpt.get("activation", "groupsort")
    input_encoding = ckpt.get("input_encoding", args.input_encoding)
    multires = ckpt.get("multires", args.multires)
    if architecture == "neus":
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("layers.") and k.endswith(".weight")))
    else:
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("net.") and k.endswith(".weight")
                                      and "_u" not in k))
    print(f"  arch={architecture}  hidden={hidden}  depth={depth}  group_size={group_size}  "
          f"activation={activation}  input_encoding={input_encoding}  multires={multires}")
    f = make_model(hidden=hidden, depth=depth, group_size=group_size, activation=activation,
                   input_encoding=input_encoding, multires=multires,
                   architecture=architecture).to(device)
    f.load_state_dict(ckpt["f"], strict=False)  # strict=False: cache buffers missing in old ckpts
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))     # populate caches

    from .visualize import (visualize, render_vs_reference, view_in_viser,
                            lego_stats, geom_stats, chamfer_stats, dtu_official_chamfer)

    eval_cfg = EvalConfig(mc_res=args.mc_res, render_down=args.render_down,
                          dtu_chamfer_bound=args.dtu_chamfer_bound,
                          dtu_chamfer_res=args.dtu_chamfer_res,
                          dtu_chamfer_freq=args.dtu_chamfer_freq,
                          dtu_official_freq=args.dtu_official_freq,
                          dtu_official_res=args.dtu_official_res,
                          dtu_official_bound=args.dtu_official_bound)
    if args.viewer:
        view_in_viser(f, res=args.viewer_res, bound=eval_cfg.bound(args.blender),
                      port=args.viewer_port, use_blender=args.blender)
        sys.exit(0)

    visualize(f, eval_cfg=eval_cfg, use_blender=args.blender)
    if args.blender:
        lego_stats(f, label=ckpt_path.name)
    else:
        render_vs_reference(f)
        geom_stats(f)
        chamfer_stats(f)
        if args.dtu_eval_dir is not None:
            import re
            scan_id = int(re.search(r"scan(\d+)", str(run_cfg.scene)).group(1))
            dtu_official_chamfer(f, scan_id, Path(args.dtu_eval_dir),
                                 scene=run_cfg.scene, eval_cfg=eval_cfg)
