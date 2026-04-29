"""Training loop and sphere-SDF initialisation for the 1-Lip tracer."""
from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import SCENE, BLENDER_SCENE, OUT_DIR, Config, ModelConfig, InitConfig, TrainConfig, TraceConfig, EvalConfig, MvsdfScheduleConfig
from .data import (load_colmap_points, load_camera_centers, load_sfm_pairs,
                   load_views, load_blender_views, load_blender_gt_points,
                   make_deterministic_rays, precompute_alt_cameras)
from .loss import (photo_loss, mask_loss_min_sdf, silhouette_loss, eikonal_loss, cam_free_loss,
                   sfm_sdf_loss, free_space_loss, surface_loss, mvs_depth_loss,
                   mvsdf_carving_loss, behind_hit_loss)
from .model import FTheta, ConvexPotentialLayer, make_model
from .sphere_tracing import trace_unrolled, trace_nograd


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
) -> FTheta:
    """Warm-start f_θ by fitting to the visual hull SDF."""
    from .visual_hull import carve, carve_octree, fit_to_hull
    model_cfg = model_cfg or ModelConfig()
    init_cfg  = init_cfg  or InitConfig()
    print(f"  hull init: carving at res={init_cfg.hull_res} mode={init_cfg.hull_mode} …")
    if init_cfg.hull_mode == "octree":
        occ = carve_octree(
            scene=scene,
            res=init_cfg.hull_res,
            bound=bound,
            min_depth=init_cfg.octree_min_depth,
        )
    else:
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
    hull_lr = init_cfg.lr if getattr(model_cfg, "architecture", "cpl") == "cpl" else min(init_cfg.lr, 1e-3)
    f = fit_to_hull(occ, bound=bound, steps=init_cfg.steps, batch=init_cfg.batch,
                    lr=hull_lr, cfg=model_cfg,
                    depth_points=depth_pts, w_depth_surface=w_depth_surface,
                    cam_origins=cam_origins_np, w_cam_free=0.0 if is_blender else 1.0)
    print("  hull init done")
    return f


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
            xh, t, hit = trace_nograd(f, sv_det["o"][i:i+chunk], sv_det["d"][i:i+chunk], trace_cfg)
            x_hits.append(xh); ts.append(t); hits.append(hit)
    x_theta = torch.cat(x_hits)
    hit     = torch.cat(hits)
    vi      = sv_det["vi"]   # all == view_idx

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
    out = run_dir / f"residual_map_{step:05d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [residual_map] → {out.name}  "
          f"mean={res_mean[hit.cpu()].mean():.4f}  "
          f"std={res_std[valid_px].mean():.4f}")


# ---------- periodic render ----------

def _render_poses(f, views, step: int, run_dir: Path, device: str,
                  res: int = 400) -> None:
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

    imgs_phong, imgs_color = [], []
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

        x_hit, _, hit = trace_nograd(f, o_t, d_t)
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

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    labels = ["view A", "view B", "view C", f"back (v{back_id})"]
    for ax, img, lbl in zip(axes[0], imgs_phong, labels):
        ax.imshow(img.clip(0, 1)); ax.axis("off"); ax.set_title(lbl, fontsize=9)
    for ax, img in zip(axes[1], imgs_color):
        ax.imshow(img.clip(0, 1)); ax.axis("off")
    axes[0][1].set_title("Phong", fontsize=10)
    axes[1][1].set_title("Colour (GT × hit)", fontsize=10)
    fig.suptitle(f"step {step}", fontsize=11)
    fig.tight_layout()
    out = run_dir / f"render_{step:05d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [render] → {out.name}")


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
    """Marching-cubes surface extraction + split Chamfer vs GT point cloud."""
    from skimage.measure import marching_cubes
    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i+4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None
    spacing = 2 * bound / (res - 1)
    verts, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts = (verts - bound).astype(np.float32)
    if len(verts) == 0:
        return None
    # sample mesh surface uniformly up to 30K pts (matches GT cloud size)
    n = min(30_000, len(verts))
    idx = np.random.default_rng(0).choice(len(verts), n, replace=False)
    pred_pts = torch.from_numpy(verts[idx]).to(device)
    return _chamfer(pred_pts, gt_pts.to(device))


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
    verts_norm, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts_norm = (verts_norm - bound).astype(np.float32)
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


# ---------- training ----------

def train(cfg: Config = None, use_wandb: bool = False) -> Path:
    """Train the 1-Lip SDF and save checkpoints to a timestamped run directory.

    Returns the path to the final checkpoint.
    """
    cfg = cfg or Config()
    # unpack for convenience
    model_cfg = cfg.model
    trace_cfg = cfg.trace
    init_cfg  = cfg.init
    train_cfg = cfg.train
    eval_cfg  = cfg.eval
    scene     = cfg.scene
    out_dir   = cfg.out_dir

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = "blender" if train_cfg.use_blender else "dtu"
    run_dir = out_dir / f"run_{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT         = run_dir / "checkpoint.pt"
    RENDER_OUT  = run_dir / "render.png"
    SURFACE_OUT = run_dir / "surface.png"

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
        sfm_origins  = torch.zeros(1, 3, device=device)
        sfm_targets  = torch.zeros(1, 3, device=device)
        n_sfm_pairs  = 1
        mvs_data     = None
    else:
        gt_pts = None
        # official DTU eval data (GT cloud + ObsMask) for in-training Chamfer
        dtu_eval_data  = None
        dtu_scale_mat  = None
        if eval_cfg.dtu_eval_dir is not None and eval_cfg.dtu_chamfer_freq > 0:
            import re
            m = re.search(r"scan(\d+)", str(scene))
            if m:
                _scan_id = int(m.group(1))
                dtu_eval_data = _load_dtu_eval_data(eval_cfg.dtu_eval_dir, _scan_id, device)
                if dtu_eval_data is not None:
                    dtu_scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
        if train_cfg.w_sfm > 0:
            sfm_pts = load_colmap_points(scene).to(device)
            print(f"  loaded {sfm_pts.shape[0]} SFM points for sdf loss")
        else:
            sfm_pts = torch.zeros(1, 3, device=device)
        if train_cfg.w_free > 0:
            sfm_origins, sfm_targets = load_sfm_pairs(scene)
            sfm_origins, sfm_targets = sfm_origins.to(device), sfm_targets.to(device)
            n_sfm_pairs = sfm_origins.shape[0]
            print(f"  {n_sfm_pairs} (cam, sfm_point) pairs for free-space loss")
        else:
            sfm_origins = torch.zeros(1, 3, device=device)
            sfm_targets = torch.zeros(1, 3, device=device)
            n_sfm_pairs = 1
        if train_cfg.w_surf > 0 or train_cfg.w_mvs > 0 or train_cfg.w_mvs_sdf > 0 or train_cfg.w_normal > 0 or train_cfg.mvsdf_schedule.enabled:
            if train_cfg.mvs_depth_dir is not None:
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
    if init_cfg.init == "hull":
        f = fit_hull_init(model_cfg=model_cfg, init_cfg=init_cfg, scene=scene, bound=bound)
        torch.cuda.empty_cache()
    else:
        _init_cfg = InitConfig(radius=1.0) if train_cfg.use_blender else init_cfg
        if train_cfg.use_blender:
            print("  [blender] sphere-init radius=1.0")
        f, _ = fit_sphere_init(model_cfg=model_cfg, init_cfg=_init_cfg, scene=scene)
    f = f.to(device)
    total_params = sum(p.numel() for p in f.parameters())
    n_cpl = sum(1 for m in f.net if isinstance(m, ConvexPotentialLayer)) if hasattr(f, "net") else 0
    arch_tag = getattr(f, "activation", "mlp")
    print(f"  model: hidden={f.hidden}  depth={n_cpl or f.depth}  params={total_params:,}  "
          f"arch={arch_tag}  enc={f.input_encoding}  multires={f.multires}")

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
    n_fg   = int(train_cfg.batch * 0.7)   # 70% foreground, 30% background
    n_bg   = train_cfg.batch - n_fg

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
        depth_p95       = mvs_depth_flat[mvs_valid_flat].quantile(0.95)
        mvs_valid_flat  = mvs_valid_flat & (mvs_depth_flat < depth_p95)
        mvs_depth_maps  = mvs_depth_flat.reshape(V, H_d, W_d)
        mvs_valid_maps  = mvs_valid_flat.reshape(V, H_d, W_d)
        mvs_normal_maps = mvs_normal_flat.reshape(V, H_d, W_d, 3)
        print(f"  MVS depth: {mvs_valid_flat.sum()}/{total_rays} valid, p95={depth_p95:.3f}")

    if train_cfg.single_view >= 0:
        sv_mask = (det["vi"] == train_cfg.single_view)
        print(f"  [single-view] view {train_cfg.single_view}: "
              f"{sv_mask.sum()} / {total_rays} rays")
        det             = {k: v[sv_mask] for k, v in det.items()}
        mvs_depth_flat  = mvs_depth_flat[sv_mask]
        mvs_valid_flat  = mvs_valid_flat[sv_mask]
        mvs_normal_flat = mvs_normal_flat[sv_mask]
        total_rays      = det["o"].shape[0]
        fg_idx = det["fg"].nonzero(as_tuple=True)[0]
        bg_idx = (~det["fg"]).nonzero(as_tuple=True)[0]

    alt_nn = precompute_alt_cameras(views, train_cfg.n_alt).to(device)
    print(f"  alt cameras: {train_cfg.n_alt} NN per view")

    # --- held-out log subset ---
    n_log   = min(4096, total_rays)
    log_idx = torch.randperm(total_rays, device=device)[:n_log]
    log_o   = det["o"][log_idx];  log_d  = det["d"][log_idx]
    log_vi  = det["vi"][log_idx]; log_fg = det["fg"][log_idx]
    log_mvs_d = mvs_depth_flat[log_idx]; log_mvs_v = mvs_valid_flat[log_idx]
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.steps,
                                                             eta_min=train_cfg.lr / 10)
    best_score        = float("inf"); best_step = -1
    best_photo_score  = float("inf")
    BEST_OUT       = run_dir / "checkpoint_best_geo.pt"
    BEST_PHOTO_OUT = run_dir / "checkpoint_best_photo.pt"
    photo_history: list[tuple[int, float]] = []

    # ------------------------------------------------------------------ loop --
    for step in range(train_cfg.steps):
        idx      = torch.cat([
            fg_idx[torch.randint(0, len(fg_idx), (n_fg,), device=device)],
            bg_idx[torch.randint(0, len(bg_idx), (n_bg,), device=device)],
        ])
        o        = det["o"][idx];  u  = det["d"][idx]
        vi       = det["vi"][idx]; gt = det["gt"][idx]
        fg_self  = det["fg"][idx]

        _need_eik = train_cfg.w_eikonal > 0 or train_cfg.w_mvs_sdf > 0 or train_cfg.mvsdf_schedule.enabled
        x_theta, t, hit, eik_pts, n_raw, sdf_min = trace_unrolled(f, o, u, trace_cfg,
                                                                   collect_eik=_need_eik)

        if step % 50 == 0:
            print(f"step {step:5d}  |∇f| mean={n_raw.norm(dim=-1).mean():.4f}")
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

        # --- losses ---
        ph, ph_stats = photo_loss(
            f, x_theta, hit, n,
            vi, alt_nn, origins_all,
            images, K_all, w2c_all, feature_maps, masks, fg_self,
            H, W, uv_self,
            train_cfg.n_alt, train_cfg.cos_thresh,
            _eff_w_photo, _eff_w_feat, train_cfg.w_ncc, train_cfg.ncc_patch, train_cfg.ncc_half_pix,
            train_cfg.sample_mode, train_cfg.gaussian_sigma, train_cfg.gaussian_radius,
            step,
        )

        sil  = (mask_loss_min_sdf(sdf_min, fg_self, train_cfg.sil_s, train_cfg.sil_fg_offset)
                if train_cfg.w_sil > 0 else torch.zeros(1, device=device).squeeze())
        eik  = (eikonal_loss(f, eik_pts, train_cfg.n_eik_vol, device)
                if train_cfg.w_eikonal > 0 else torch.zeros(1, device=device).squeeze())
        cfr  = (cam_free_loss(f, o)
                if train_cfg.w_cam_free > 0 else torch.zeros(1, device=device).squeeze())
        sfm  = (sfm_sdf_loss(f, sfm_pts, train_cfg.batch)
                if train_cfg.w_sfm > 0 else torch.zeros(1, device=device).squeeze())
        fs   = (free_space_loss(f, sfm_origins, sfm_targets, n_sfm_pairs,
                                train_cfg.batch, train_cfg.n_free)
                if train_cfg.w_free > 0 else torch.zeros(1, device=device).squeeze())
        surf = (surface_loss(f, o, u, vi, c2w_all, mvs_depth_flat, mvs_valid_flat, idx)
                if train_cfg.w_surf > 0 else torch.zeros(1, device=device).squeeze())
        mvs  = (mvs_depth_loss(x_theta, o, u, vi, hit, c2w_all,
                               mvs_depth_flat, mvs_valid_flat, idx, step)
                if train_cfg.w_mvs > 0 else torch.zeros(1, device=device).squeeze())
        if _eff_w_msdf > 0:
            if eik_pts.numel() > 0:
                mvs_sdf_idx = torch.randint(0, eik_pts.shape[0], (train_cfg.n_mvs_sdf,), device=device)
                mvs_sdf_pts = eik_pts[mvs_sdf_idx]
            else:
                mvs_sdf_pts = (2 * torch.rand(train_cfg.n_mvs_sdf, 3, device=device) - 1) * bound
            msdf = mvsdf_carving_loss(
                f, mvs_sdf_pts,
                w2c_all, K_all,
                mvs_depth_maps, mvs_valid_maps,
                H, W, train_cfg.down,
                train_cfg.mvs_sdf_out_thresh,
                train_cfg.mvs_sdf_trunc, train_cfg.mvs_sdf_smooth,
                train_cfg.mvs_sdf_far_thresh, train_cfg.mvs_sdf_far_att,
                train_cfg.mvs_sdf_near_thresh, train_cfg.mvs_sdf_near_att,
                step,
            )
        else:
            msdf = torch.zeros(1, device=device).squeeze()
        beh  = (behind_hit_loss(f, x_theta, hit, u, train_cfg.behind_eps)
                if train_cfg.w_behind_hit > 0 else torch.zeros(1, device=device).squeeze())

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
            hit_valid_n = hit & mvs_valid_flat[idx]
            if hit_valid_n.any():
                # rotate GT cam-space normal to world space: n_world = R_c2w @ n_cam
                R_c2w = c2w_all[vi[hit_valid_n], :3, :3]   # (M, 3, 3)
                n_cam = mvs_normal_flat[idx[hit_valid_n]]   # (M, 3)
                n_gt  = torch.einsum("bij,bj->bi", R_c2w, n_cam)
                n_gt  = F.normalize(n_gt, dim=-1)
                n_pred = F.normalize(n_raw[hit_valid_n], dim=-1)
                nrm = (1.0 - (n_pred * n_gt).sum(-1).clamp(-1, 1)).mean()
            else:
                nrm = f(x_theta.detach()[:1]).sum() * 0.0
        else:
            nrm = torch.zeros(1, device=device).squeeze()

        loss = (ph + train_cfg.w_sil * sil + train_cfg.w_eikonal * eik
                + train_cfg.w_cam_free * cfr + train_cfg.w_sfm * sfm + train_cfg.w_free * fs
                + train_cfg.w_surf * surf + train_cfg.w_mvs * mvs + _eff_w_msdf * msdf
                + train_cfg.w_behind_hit * beh + train_cfg.w_ray_free * rf
                + train_cfg.w_normal * nrm)
        photo_history.append((step, ph.item()))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
        opt.step(); scheduler.step()

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
                photo_str += f"  (l1 {pm['l1']:.4f}  feat {pm['feature']:.4f}[w={_eff_w_feat}]  ncc {pm['ncc']:.4f})"
            print(f"step {step:5d}  cams {vi.unique().numel():2d}  "
                  f"loss {loss.item():.4f}  {photo_str}  sil {sil.item():.4f}  "
                  f"hit {hit.sum()}/{train_cfg.batch}  x_r {xh_str}  "
                  f"mask {pm['n_mask']}/{pm['n_in_frame']}/{pm['n_not_occl']}/"
                  f"{pm['n_cos_ok']}/{pm['n_total']}  "
                  f"f(o)<0 {frac_fo_neg:.2f}  t_far {frac_t_far:.2f}  "
                  f"sfm {sfm.item():.4f}  free {fs.item():.4f}  "
                  f"surf {surf.item():.4f}  mvs {mvs.item():.4f}  "
                  f"msdf {msdf.item():.4f}[w={_eff_w_msdf}]  beh {beh.item():.4f}  "
                  f"rf {rf.item():.4f}  eik {eik.item():.4f}[w={train_cfg.w_eikonal}]  "
                  f"nrm {nrm.item():.4f}  ∇head {grad_norm:.6f}")

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
                _render_poses(f, views, step, run_dir, device)
                render_src = run_dir / f"render_{step:05d}.png"
                if render_src.exists():
                    shutil.copy(render_src, run_dir / "render_best_geo.png")

            if ph.item() > 1e-6 and ph.item() < best_photo_score:
                best_photo_score = ph.item()
                torch.save({"f": f.state_dict(), "step": step, "photo": ph.item(),
                            "architecture": f.architecture, "group_size": f.group_size,
                            "depth": f.depth, "activation": f.activation,
                            "input_encoding": f.input_encoding,
                            "multires": f.multires}, BEST_PHOTO_OUT)
                print(f"  [best_photo@{step}] photo={ph.item():.4f} → {BEST_PHOTO_OUT.name}")
                torch.cuda.empty_cache()
                _render_poses(f, views, step, run_dir, device)
                render_src = run_dir / f"render_{step:05d}.png"
                if render_src.exists():
                    shutil.copy(render_src, run_dir / "render_best_photo.png")

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
            if train_cfg.use_blender and gt_pts is not None and step % 500 == 0:
                cd = _mc_chamfer(f, gt_pts, device, bound=eval_cfg.bound_blender)
                if cd is not None:
                    print(f"  [chamfer@{step:5d}] sym={cd['chamfer']:.6f}  "
                          f"precision={cd['precision']:.6f}  completeness={cd['completeness']:.6f}")
                else:
                    print(f"  [chamfer@{step:5d}] n/a (surface not in bounds)")
            elif (not train_cfg.use_blender and dtu_eval_data is not None
                  and eval_cfg.dtu_chamfer_freq > 0 and step % eval_cfg.dtu_chamfer_freq == 0):
                cd = _mc_chamfer_dtu(f, dtu_eval_data, dtu_scale_mat, device,
                                     bound=eval_cfg.dtu_chamfer_bound, res=eval_cfg.dtu_chamfer_res)
                if cd is not None:
                    print(f"  [dtu_chamfer@{step:5d}] sym={cd['chamfer']:.4f}  "
                          f"acc={cd['precision']:.4f}  comp={cd['completeness']:.4f}")
                else:
                    print(f"  [dtu_chamfer@{step:5d}] n/a (surface not in bounds)")

            if use_wandb:
                import wandb
                log = {"loss": loss.item(), "photo": ph.item(), "sil": sil.item(),
                       "photo_l1": ph_stats["l1"], "photo_ncc": ph_stats["ncc"],
                       "photo_feature": ph_stats["feature"],
                       "hit_rate": hit.float().mean().item(), "cams_out": cams_out,
                       "grad_norm_mean": grad_mean, "sil_iou": sil_iou,
                       "best_geo_score": best_score}
                if cd is not None:
                    log["chamfer"] = cd["chamfer"]
                    log["chamfer_acc"] = cd["precision"]
                    log["chamfer_comp"] = cd["completeness"]
                if (run_dir / "render_{:05d}.png".format(step)).exists():
                    log["render"] = wandb.Image(str(run_dir / "render_{:05d}.png".format(step)))
                wandb.log(log, step=step)

    # --- save final ---
    torch.save({"f": f.state_dict(), "architecture": f.architecture,
                "group_size": f.group_size, "depth": f.depth,
                "activation": f.activation,
                "input_encoding": f.input_encoding,
                "multires": f.multires}, OUT)
    print(f"saved → {OUT}")
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
        plot_out = run_dir / "photo_loss.png"
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
    ap.add_argument("--blender",    action="store_true", help="use a Blender synthetic dataset")
    ap.add_argument("--dataset",    choices=["skull", "lego"], default=None)
    ap.add_argument("--scene",      type=Path, default=None,
                    help="override scene path (useful for non-default Blender objects)")
    ap.add_argument("--hidden",     type=int,   default=_mc.hidden)
    ap.add_argument("--depth",      type=int,   default=_mc.depth)
    ap.add_argument("--group-size", type=int,   default=_mc.group_size)
    ap.add_argument("--activation", type=str,   default=_mc.activation,
                    choices=["groupsort", "nact"], help="activation: groupsort or nact (N-Activation)")
    ap.add_argument("--input-encoding", type=str, default=_mc.input_encoding,
                    choices=["identity", "neus"], help="input encoding before the 1-Lipschitz backbone")
    ap.add_argument("--architecture", type=str, default="cpl", choices=["cpl", "mlp"],
                    help="cpl: 1-Lipschitz CPL network (default); mlp: unconstrained MLP baseline")
    ap.add_argument("--multires", type=int, default=_mc.multires,
                    help="number of NeuS-style positional encoding frequencies")
    _trc = TraceConfig()
    ap.add_argument("--trace-iters", type=int, default=_trc.iters,
                    help="max sphere-tracing iterations (increase when using PE)")
    _ic = InitConfig()
    ap.add_argument("--init",       type=str,   default=_ic.init,
                    choices=["sphere", "hull"], help="warm-start: sphere or visual hull (blender only)")
    ap.add_argument("--radius",     type=float, default=_ic.radius,
                    help="sphere-init radius (None = auto from COLMAP p60)")
    ap.add_argument("--hull-res",   type=int,   default=_ic.hull_res,
                    help="voxel resolution for hull carving")
    ap.add_argument("--hull-mode",  type=str,   default=_ic.hull_mode,
                    choices=["grid", "octree"], help="visual-hull carving backend")
    ap.add_argument("--octree-min-depth", type=int, default=_ic.octree_min_depth,
                    help="for octree hulls: accept fully-inside cells at/after this depth")
    ap.add_argument("--w-depth-surface-init", type=float, default=_ic.w_depth_surface,
                    help="blender hull-init only: weight on GT depth surface samples")
    ap.add_argument("--w-photo",    type=float, default=_tc.w_photo)
    ap.add_argument("--w-feature",  type=float, default=_tc.w_feature,
                    help="weight for cosine distance on precomputed feature maps")
    ap.add_argument("--feature-maps", type=Path, default=_tc.feature_maps,
                    help="path to a .pt file produced by precompute_mast3r_features.py")
    ap.add_argument("--w-ncc",      type=float, default=_tc.w_ncc)
    ap.add_argument("--ncc-half-pix", type=float, default=_tc.ncc_half_pix,
                    help="PMVS patch half-width in reference-view pixels")
    ap.add_argument("--sample-mode", type=str, default=_tc.sample_mode,
                    choices=["bilinear", "gaussian"],
                    help="image sampler for photo and NCC losses")
    ap.add_argument("--gaussian-sigma",  type=float, default=_tc.gaussian_sigma)
    ap.add_argument("--gaussian-radius", type=int,   default=_tc.gaussian_radius)
    ap.add_argument("--w-sil",          type=float, default=_tc.w_sil)
    ap.add_argument("--sil-fg-offset",  type=float, default=_tc.sil_fg_offset,
                    help="sdf_min shift for fg rays in mask loss; pushes σ away from 0.5 on hits")
    ap.add_argument("--w-eikonal",  type=float, default=_tc.w_eikonal)
    ap.add_argument("--w-sfm",      type=float, default=_tc.w_sfm)
    ap.add_argument("--w-free",     type=float, default=_tc.w_free)
    ap.add_argument("--w-surf",     type=float, default=_tc.w_surf)
    ap.add_argument("--w-mvs",      type=float, default=_tc.w_mvs)
    ap.add_argument("--mvs-depth-dir", type=Path, default=None,
                    help="directory with MASt3R depth manifest.json (IDR scans); "
                         "enables load_mast3r_depths_idr instead of load_aligned_depths")
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
    ap.add_argument("--w-normal",   type=float, default=_tc.w_normal)
    ap.add_argument("--w-ray-free", type=float, default=_tc.w_ray_free)
    ap.add_argument("--n-ray-free", type=int,   default=_tc.n_ray_free)
    ap.add_argument("--single-view", type=int, default=_tc.single_view,
                    help="restrict training to this view index (-1=all, overfit diagnostic)")
    ap.add_argument("--pt",         default=None, help="checkpoint to evaluate")
    ap.add_argument("--dtu-eval-dir", default=None,
                    help="path to DTU evaluation data (SampleSet/ + ObsMask/ subdirs)")
    ap.add_argument("--viewer",      action="store_true")
    ap.add_argument("--viewer-res",  type=int, default=256)
    ap.add_argument("--viewer-port", type=int, default=8080)
    ap.add_argument("--render-down", type=int, default=1,
                    help="downsample for sphere-traced PNG renders (1=full res, 2=half)")
    ap.add_argument("--mc-res",           type=int,   default=256,
                    help="marching-cubes grid resolution for eval/viewer")
    ap.add_argument("--dtu-chamfer-bound", type=float, default=0.8,
                    help="MC bound for in-training DTU Chamfer (tighter than SDF grid for better voxel precision)")
    ap.add_argument("--wandb",       action="store_true", help="log to Weights & Biases")
    args = ap.parse_args()

    if args.dataset == "lego":
        args.blender = True
    elif args.dataset == "skull":
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
        trace=TraceConfig(iters=args.trace_iters),
        init=InitConfig(
            init=args.init,
            radius=args.radius,
            hull_res=args.hull_res,
            hull_mode=args.hull_mode,
            octree_min_depth=args.octree_min_depth,
            w_depth_surface=args.w_depth_surface_init,
        ),
        train=TrainConfig(
            steps=args.steps, batch=args.batch, lr=args.lr, down=args.down,
            use_blender=args.blender, single_view=args.single_view,
            w_photo=args.w_photo, w_feature=args.w_feature, feature_maps=args.feature_maps,
            w_ncc=args.w_ncc, ncc_half_pix=args.ncc_half_pix,
            sample_mode=args.sample_mode,
            gaussian_sigma=args.gaussian_sigma, gaussian_radius=args.gaussian_radius,
            w_sil=args.w_sil, sil_fg_offset=args.sil_fg_offset,
            w_eikonal=args.w_eikonal, w_sfm=args.w_sfm, w_free=args.w_free,
            w_surf=args.w_surf, w_mvs=args.w_mvs, mvs_depth_dir=args.mvs_depth_dir,
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
            w_ray_free=args.w_ray_free, n_ray_free=args.n_ray_free,
        ),
        eval=EvalConfig(
            dtu_eval_dir=Path(args.dtu_eval_dir) if args.dtu_eval_dir else None,
        ),
        scene=scene_path,
    )

    ckpt_path: Path | None = None
    if not args.no_train:
        ckpt_path = train(run_cfg, use_wandb=args.wandb)

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
    architecture = ckpt.get("architecture", "cpl")
    if "head_weight" in ckpt["f"]:
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
    depth      = ckpt.get("depth", sum(1 for k in ckpt["f"]
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
                          dtu_chamfer_bound=args.dtu_chamfer_bound)
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
