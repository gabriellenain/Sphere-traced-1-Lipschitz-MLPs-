"""Visualisation: static PNG panels, interactive Viser viewer, eval metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import SCENE, BLENDER_SCENE, OUT_DIR, EvalConfig, TraceConfig
from .data import (load_colmap_points, load_camera_centers,
                   load_views, load_blender_views,
                   make_deterministic_rays, load_sfm_pairs)
from .model import FTheta
from .sphere_tracing import trace_nograd


# ---------- helpers ----------

def _nn_dist(A: Tensor, B: Tensor, chunk: int = EvalConfig().nn_chunk) -> Tensor:
    """For each row of A, distance to its nearest neighbour in B (chunked)."""
    out = torch.empty(A.shape[0], device=A.device)
    for i in range(0, A.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(A[i:i + chunk], B).min(dim=1).values
    return out


def _marching_cubes(f: FTheta, bound: float, res: int, device: str):
    """Extract iso-surface; returns (verts, faces) or (None, None) if no crossing."""
    from skimage import measure
    grid_t = torch.linspace(-bound, bound, res, device=device)
    xs, ys, zs = torch.meshgrid(grid_t, grid_t, grid_t, indexing="ij")
    pts   = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
    total = pts.shape[0]
    print(f"  marching cubes: evaluating {total:,} points at res={res}…", flush=True)
    vals = torch.cat([f(p) for p in pts.split(65536)]).reshape(res, res, res).detach().cpu().numpy()
    print(f"  marching cubes: done", flush=True)
    if vals.min() > 0 or vals.max() < 0:
        return None, None, vals
    verts, faces, _, _ = measure.marching_cubes(vals, level=0.0)
    verts = verts / (res - 1) * (2 * bound) - bound
    return verts, faces, vals


def _trace_view(f: FTheta, c2w: np.ndarray, K: np.ndarray,
                H_d: int, W_d: int, down: int, device: str):
    """Ray-trace a downsampled view and return (n_img, hit_img) as numpy arrays."""
    ys_g, xs_g = np.meshgrid(np.arange(H_d), np.arange(W_d), indexing="ij")
    xs_f = (xs_g + 0.5) * down - 0.5
    ys_f = (ys_g + 0.5) * down - 0.5
    d_cam = np.stack([(xs_f - K[0, 2]) / K[0, 0],
                      (ys_f - K[1, 2]) / K[1, 1],
                      np.ones_like(xs_f)], axis=-1)
    d = d_cam @ c2w[:3, :3].T
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    o_t = torch.from_numpy(np.broadcast_to(c2w[:3, 3], d.shape).copy().reshape(-1, 3)).float().to(device)
    d_t = torch.from_numpy(d.reshape(-1, 3)).float().to(device)
    x_hit, _, hit = trace_nograd(f, o_t, d_t)
    with torch.enable_grad():
        xr = x_hit.detach().clone().requires_grad_(True)
        n  = torch.autograd.grad(f(xr).sum(), xr)[0]
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    n_img   = (0.5 * (n + 1.0)).clamp(0, 1).reshape(H_d, W_d, 3).detach().cpu().numpy()
    hit_img = hit.reshape(H_d, W_d).detach().cpu().numpy()
    return n_img, hit_img


# ---------- main visualisation ----------

@torch.no_grad()
def visualize(f: FTheta, eval_cfg: EvalConfig = None,
              use_blender: bool = False,
              out: Path | None = None) -> None:
    """Marching-cubes surface + per-view normal overlays → PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    eval_cfg = eval_cfg or EvalConfig()
    device   = next(f.parameters()).device
    bound    = eval_cfg.bound(use_blender)
    verts, faces, vals = _marching_cubes(f, bound, eval_cfg.mc_res, str(device))
    if verts is None:
        print(f"WARNING: no zero crossing (min={vals.min():.4f} max={vals.max():.4f})")
        return

    if use_blender:
        colmap = np.zeros((0, 3))
        views  = load_blender_views(down=2)
        cams   = views["c2w"].numpy()[:, :3, 3]
    else:
        colmap = load_colmap_points().numpy()
        cams   = load_camera_centers().numpy()
        views  = load_views()

    H, W   = views["H"], views["W"]
    down   = eval_cfg.render_down
    H_d, W_d = H // down, W // down
    V      = views["images"].shape[0]
    n_grid = eval_cfg.n_views
    view_ids = [int(round(i * (V - 1) / (n_grid - 1))) for i in range(n_grid)]

    overlays = []
    for vi_ref in view_ids:
        K   = views["K"][vi_ref].numpy()
        c2w = views["c2w"][vi_ref].numpy()
        n_img, hit_img = _trace_view(f, c2w, K, H_d, W_d, down, str(device))
        ref_np  = views["images"][vi_ref].numpy()
        ref_ds  = ref_np[::down, ::down][:H_d, :W_d]
        overlay = ref_ds.copy()
        overlay[hit_img] = 0.45 * ref_ds[hit_img] + 0.55 * n_img[hit_img]
        if use_blender and "masks" in views:
            gt_mask = views["masks"][vi_ref].numpy()
            gt_ds   = gt_mask[::down, ::down][:H_d, :W_d]
            bnd     = np.zeros_like(gt_ds, dtype=bool)
            bnd[1:]  |= gt_ds[1:]  != gt_ds[:-1]
            bnd[:-1] |= gt_ds[:-1] != gt_ds[1:]
            bnd[:, 1:]  |= gt_ds[:, 1:]  != gt_ds[:, :-1]
            bnd[:, :-1] |= gt_ds[:, :-1] != gt_ds[:, 1:]
            overlay[bnd] = np.array([1.0, 0.55, 0.0])
        overlays.append((vi_ref, overlay))

    n_top = n_grid + (2 if use_blender else 0)
    fig = plt.figure(figsize=(18 + (4 if use_blender else 0), 12))
    gs  = GridSpec(2, n_top, figure=fig, height_ratios=[1.4, 1.0])

    for i, (elev, azim) in enumerate([(20, 30), (20, 120)]):
        span = n_grid // 2
        ax   = fig.add_subplot(gs[0, i * span:(i + 1) * span], projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        color="lightsteelblue", alpha=0.5, linewidth=0)
        if not use_blender:
            ax.scatter(colmap[:, 0], colmap[:, 1], colmap[:, 2], s=1, c="red",  alpha=0.6)
            ax.scatter(cams[:, 0],   cams[:, 1],   cams[:, 2],   s=15, c="blue",
                       marker="^", alpha=0.8)
        ax.view_init(elev=elev, azim=azim); ax.set_box_aspect((1, 1, 1))
        if len(verts):
            ctr  = verts.mean(0)
            half = max(0.55 * (verts.max(0) - verts.min(0)).max(), 0.1)
            ax.set_xlim(ctr[0]-half, ctr[0]+half)
            ax.set_ylim(ctr[1]-half, ctr[1]+half)
            ax.set_zlim(ctr[2]-half, ctr[2]+half)

    if use_blender and "masks" in views:
        for k, vi_gt in enumerate([view_ids[0], view_ids[n_grid // 2]]):
            ax = fig.add_subplot(gs[0, n_grid + k])
            img_np = views["images"][vi_gt].numpy()
            msk_np = views["masks"][vi_gt].numpy()
            gray   = img_np.mean(-1, keepdims=True).repeat(3, axis=-1)
            gray[~msk_np] = 1.0
            ys_obj, xs_obj = np.where(msk_np)
            if len(ys_obj):
                pad = 10
                crop = gray[max(ys_obj.min()-pad,0):min(ys_obj.max()+pad,gray.shape[0]),
                            max(xs_obj.min()-pad,0):min(xs_obj.max()+pad,gray.shape[1])]
            else:
                crop = gray
            ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"GT view {vi_gt}", fontsize=9); ax.axis("off")

    for j, (vi_ref, overlay) in enumerate(overlays):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(overlay); ax.set_title(f"view {vi_ref}", fontsize=9); ax.axis("off")

    fig.tight_layout()
    tag     = "lego" if use_blender else "skull"
    out_png = out or (OUT_DIR / f"surface_{tag}.png")
    fig.savefig(out_png, dpi=150); print(f"png → {out_png}")


# ---------- render vs reference ----------

@torch.no_grad()
def render_vs_reference(f: FTheta, views_list: tuple[int, ...] = (0, 15, 30, 45),
                        down: int = 2, out: Path | None = None) -> None:
    """Bilinear-reproject sphere-traced hits into the source image + normals."""
    import matplotlib.pyplot as plt
    views  = load_views()
    device = next(f.parameters()).device
    V      = views["images"].shape[0]
    views_list = tuple(int(round(i * (V-1) / (len(views_list)-1)))
                       for i in range(len(views_list)))

    fig, axes = plt.subplots(len(views_list), 3, figsize=(12, 3 * len(views_list)))
    for row, vi in zip(axes, views_list):
        img_np = views["images"][vi].numpy()
        img_t  = views["images"][vi].to(device)
        H, W   = img_np.shape[:2]
        H_d, W_d = H // down, W // down
        K      = views["K"][vi].numpy();   K_t = views["K"][vi].to(device)
        c2w    = views["c2w"][vi].numpy()
        w2c_t  = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device)
        ys, xs = np.meshgrid(np.arange(H_d), np.arange(W_d), indexing="ij")
        xs_f   = (xs + 0.5) * down - 0.5; ys_f = (ys + 0.5) * down - 0.5
        d_cam  = np.stack([(xs_f-K[0,2])/K[0,0], (ys_f-K[1,2])/K[1,1], np.ones_like(xs_f)], -1)
        d      = d_cam @ c2w[:3, :3].T; d /= np.linalg.norm(d, axis=-1, keepdims=True)
        o_t    = torch.from_numpy(np.broadcast_to(c2w[:3,3], d.shape).copy().reshape(-1, 3)).float().to(device)
        d_t    = torch.from_numpy(d.reshape(-1, 3)).float().to(device)
        x_theta, _, hit = trace_nograd(f, o_t, d_t)
        # round-trip reproject into same view
        xc   = (w2c_t[:3,:3] @ x_theta.T).T + w2c_t[:3, 3]
        uv_h = (K_t @ xc.T).T
        uv   = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
        grid = torch.stack([uv[:,0]/(W-1)*2-1, uv[:,1]/(H-1)*2-1], dim=-1)
        N    = o_t.shape[0]
        rgb  = F.grid_sample(img_t.permute(2,0,1).unsqueeze(0).expand(N,-1,-1,-1),
                              grid.view(N,1,1,2), mode="bilinear", align_corners=True).view(N,3)
        rgb  = torch.where(hit.unsqueeze(-1), rgb, torch.ones_like(rgb))
        with torch.enable_grad():
            xr = x_theta.detach().clone().requires_grad_(True)
            n  = torch.autograd.grad(f(xr).sum(), xr)[0]
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        n_img = torch.where(hit.unsqueeze(-1), (0.5*(n+1)).clamp(0,1), torch.ones_like(n))
        row[0].imshow(img_np);                          row[0].set_title(f"view {vi}");      row[0].axis("off")
        row[1].imshow(rgb.reshape(H_d,W_d,3).cpu());   row[1].set_title("reprojected");      row[1].axis("off")
        row[2].imshow(n_img.reshape(H_d,W_d,3).cpu()); row[2].set_title("normals");          row[2].axis("off")
    fig.tight_layout()
    out_png = out or (OUT_DIR / "render_vs_ref.png")
    fig.savefig(out_png, dpi=150); print(f"render → {out_png}")


# ---------- interactive Viser viewer ----------

@torch.no_grad()
def view_in_viser(f: FTheta, res: int = 256, bound: float = 1.5,
                  port: int = 8080, use_blender: bool = False) -> None:
    """Extract iso-surface and open an interactive Viser 3-D session."""
    import time
    import viser

    device = next(f.parameters()).device
    verts = faces = None
    for b in [bound, bound * 1.5, bound * 2.0, bound * 3.0]:
        v_cur, f_cur, _ = _marching_cubes(f, b, res, str(device))
        if v_cur is not None:
            verts, faces, bound = v_cur, f_cur, b; break
        print(f"  no zero crossing at bound={b:.2f}, trying larger...")
    if verts is None:
        print("  WARNING: no zero crossing found"); return
    verts = verts.astype("float32"); faces = faces.astype("uint32")
    print(f"  mesh: {len(verts)} verts  {len(faces)} faces")

    views = load_blender_views(down=2) if use_blender else load_views()
    c2ws  = views["c2w"].numpy(); imgs = views["images"].numpy()
    V, H, W = imgs.shape[:3]

    server = viser.ViserServer(port=port)
    server.scene.add_mesh_simple("/surface", vertices=verts, faces=faces,
                                 wxyz=(1,0,0,0), position=(0,0,0),
                                 color=(180,180,220), side="double")
    from scipy.spatial.transform import Rotation
    for v, (c2w, img) in enumerate(zip(c2ws, imgs)):
        R = c2w[:3, :3]; t = c2w[:3, 3]
        wxyz = Rotation.from_matrix(R).as_quat()[[3,0,1,2]].astype("float32")
        server.scene.add_camera_frustum(f"/cameras/{v:03d}", fov=np.deg2rad(60.0),
                                        aspect=W/H, scale=0.08, wxyz=wxyz,
                                        position=t.astype("float32"), color=(255,180,60))
        thumb = (img[::max(H//64,1), ::max(W//64,1)].clip(0,1) * 255).astype(np.uint8)
        server.scene.add_image(f"/images/{v:03d}", image=thumb,
                               render_width=0.15, render_height=0.15*H/W,
                               wxyz=wxyz, position=(t + R[:,2]*0.16).astype("float32"))
    print(f"  {V} cameras  |  http://localhost:{port}  (Ctrl-C to quit)")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass


# ---------- evaluation metrics ----------

def chamfer_stats(f: FTheta, eval_cfg: EvalConfig = None,
                  scene: Path = SCENE, use_clean: bool = True) -> dict:
    """Bidirectional Chamfer distance: reconstructed surface vs SFM cloud."""
    eval_cfg = eval_cfg or EvalConfig()
    device   = next(f.parameters()).device
    verts, _, _ = _marching_cubes(f, eval_cfg.bound_dtu, eval_cfg.mc_res, str(device))
    if verts is None:
        print("  chamfer: no zero crossing"); return {}
    A = torch.from_numpy(verts.astype(np.float32)).to(device)

    clean    = scene / "sparse_sfm_points_clean.txt"
    pts_file = clean if (use_clean and clean.exists()) else scene / "sparse_sfm_points.txt"
    B = torch.from_numpy(np.loadtxt(pts_file, dtype=np.float32)).to(device)

    d_ab = _nn_dist(A, B, eval_cfg.nn_chunk); d_ba = _nn_dist(B, A, eval_cfg.nn_chunk)
    acc  = d_ab.mean().item(); comp = d_ba.mean().item()
    stats = {
        "n_recon_verts": int(A.shape[0]), "n_gt_points": int(B.shape[0]),
        "gt_source": pts_file.name,
        "precision_mean": acc,   "precision_p50": d_ab.quantile(0.5).item(),
        "precision_p90":  d_ab.quantile(0.9).item(),
        "completeness_mean": comp, "completeness_p50": d_ba.quantile(0.5).item(),
        "completeness_p90":  d_ba.quantile(0.9).item(),
        "chamfer": 0.5 * (acc + comp),
    }
    print("\n=== chamfer ===")
    print(f"  recon {stats['n_recon_verts']} verts  gt {stats['n_gt_points']} pts  ({pts_file.name})")
    print(f"  prec  mean={acc:.4f}  p50={stats['precision_p50']:.4f}  p90={stats['precision_p90']:.4f}")
    print(f"  comp  mean={comp:.4f}  p50={stats['completeness_p50']:.4f}  p90={stats['completeness_p90']:.4f}")
    print(f"  chamfer (sym): {stats['chamfer']:.4f}")
    return stats


def dtu_official_chamfer(
    f: FTheta,
    scan_id: int,
    dtu_eval_dir: Path,
    scene: Path = SCENE,
    eval_cfg: EvalConfig = None,
) -> dict:
    """Official DTU surface evaluation (accuracy + completeness + Chamfer).

    Mirrors the evaluation used in NeuS/VolSDF/MonoSDF:
      - GT point cloud: <dtu_eval_dir>/SampleSet/STL/stl{scan_id:03d}_total.ply
      - Observation mask: <dtu_eval_dir>/ObsMask/ObsMask{scan_id}_10.mat

    The predicted mesh is extracted in normalised SDF space, then mapped back to
    DTU world coordinates via scale_mat_0 from the scene's cameras.npz.
    """
    import open3d as o3d
    from scipy.io import loadmat

    eval_cfg = eval_cfg or EvalConfig()
    device   = next(f.parameters()).device

    # ---- 1. extract mesh in normalised space ----------------------------
    verts_norm, faces, _ = _marching_cubes(f, eval_cfg.bound_dtu, eval_cfg.mc_res, str(device))
    if verts_norm is None:
        print("  dtu_chamfer: no zero crossing"); return {}

    # ---- 2. transform verts to DTU world coords via scale_mat_0 --------
    cam_dict   = np.load(scene / "cameras.npz")
    scale_mat  = cam_dict["scale_mat_0"].astype(np.float64)   # (4,4)
    v_h        = np.concatenate([verts_norm, np.ones((len(verts_norm), 1))], axis=1)  # (N,4)
    verts_world = (scale_mat @ v_h.T).T[:, :3].astype(np.float32)

    # ---- 3. load GT point cloud + ObsMask -----------------------------
    # dtu_eval_dir = 'SampleSet/MVS Data' from DTU's SampleSet.zip
    ply_path = dtu_eval_dir / "Points" / "stl" / "stl006_total.ply"
    if not ply_path.exists():
        ply_path = dtu_eval_dir / "Points" / "stl" / "stl001_total.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"GT PLY not found under {dtu_eval_dir}/Points/stl/")
    mat_path = dtu_eval_dir / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"ObsMask not found: {mat_path}")

    mat     = loadmat(str(mat_path))
    ObsMask = mat["ObsMask"].astype(bool)
    BB      = mat["BB"].astype(np.float64)
    Res     = float(mat["Res"].flat[0])

    def _in_obs(pts: np.ndarray) -> np.ndarray:
        in_bb  = np.all((pts >= BB[0]) & (pts <= BB[1]), axis=1)
        idx    = np.clip(np.round((pts - BB[0]) / Res).astype(int), 0, np.array(ObsMask.shape) - 1)
        return in_bb & ObsMask[idx[:, 0], idx[:, 1], idx[:, 2]]

    all_pts = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points, dtype=np.float32)
    gt_pts  = all_pts[_in_obs(all_pts)]   # pre-filter to this scene's region
    print(f"  GT cloud: {len(gt_pts):,} pts after ObsMask filter "
          f"(from {len(all_pts):,} in {ply_path.name})")

    # ---- 4. chunked nearest-neighbour distances in world coords --------
    pred_t = torch.from_numpy(verts_world).to(device)
    gt_t   = torch.from_numpy(gt_pts).to(device)

    chunk = eval_cfg.nn_chunk
    d_pred_gt = _nn_dist(pred_t, gt_t, chunk).cpu().numpy()   # pred → gt
    d_gt_pred = _nn_dist(gt_t, pred_t, chunk).cpu().numpy()   # gt   → pred

    # accuracy: pred points inside obs mask
    obs_pred  = _in_obs(verts_world)
    acc_dists = d_pred_gt[obs_pred]
    accuracy  = float(acc_dists.mean()) if len(acc_dists) else 0.0

    # completeness: gt already filtered, so all gt_pts are in obs mask
    comp_dists   = d_gt_pred
    completeness = float(comp_dists.mean()) if len(comp_dists) else 0.0

    chamfer = 0.5 * (accuracy + completeness)

    stats = {
        "scan_id":        scan_id,
        "n_pred_verts":   int(len(verts_world)),
        "n_pred_in_obs":  int(obs_pred.sum()),
        "n_gt_pts":       int(len(gt_pts)),
        "n_gt_in_obs":    int(obs_gt.sum()),
        "accuracy":       accuracy,
        "completeness":   completeness,
        "chamfer":        chamfer,
        "acc_p50":   float(np.median(acc_dists))  if len(acc_dists)  else 0.0,
        "acc_p90":   float(np.percentile(acc_dists,  90)) if len(acc_dists)  else 0.0,
        "comp_p50":  float(np.median(comp_dists)) if len(comp_dists) else 0.0,
        "comp_p90":  float(np.percentile(comp_dists, 90)) if len(comp_dists) else 0.0,
    }

    print(f"\n=== DTU official Chamfer (scan{scan_id}) ===")
    print(f"  pred: {stats['n_pred_verts']:,} verts  in_obs: {stats['n_pred_in_obs']:,}")
    print(f"  gt:   {stats['n_gt_pts']:,} pts   in_obs: {stats['n_gt_in_obs']:,}")
    print(f"  accuracy    (pred→gt):    mean={accuracy:.4f}  p50={stats['acc_p50']:.4f}  p90={stats['acc_p90']:.4f}")
    print(f"  completeness (gt→pred):  mean={completeness:.4f}  p50={stats['comp_p50']:.4f}  p90={stats['comp_p90']:.4f}")
    print(f"  Chamfer (sym mean):       {chamfer:.4f}")
    return stats


def geom_stats(f: FTheta, down: int = 2, scene: Path = SCENE) -> dict:
    """Final geometric-quality report: SFM SDF, free-space, hit rate, MVS depth."""
    device   = next(f.parameters()).device
    sfm_pts  = load_colmap_points(scene).to(device)
    cams     = load_camera_centers(scene).to(device)
    sfm_o, sfm_x = load_sfm_pairs(scene)
    sfm_o, sfm_x = sfm_o.to(device), sfm_x.to(device)

    with torch.no_grad():
        f_sfm = f(sfm_pts).abs()
    sfm_q = torch.quantile(f_sfm, torch.tensor([0.5, 0.9, 0.99], device=device))

    with torch.no_grad():
        f_cam = f(cams)
    frac_cam_out = (f_cam > 0).float().mean().item()

    n_free = 16
    t_fs   = torch.rand(sfm_o.shape[0], n_free, 1, device=device) * 0.9 + 0.05
    pts_fs = sfm_o.unsqueeze(1) + t_fs * (sfm_x - sfm_o).unsqueeze(1)
    with torch.no_grad():
        f_fs = f(pts_fs.reshape(-1, 3))
    frac_free_viol = (f_fs < 0).float().mean().item()
    mean_free_viol = F.relu(-f_fs).mean().item()

    x_rand = (2 * torch.rand(8192, 3, device=device) - 1) * 1.5
    x_rand.requires_grad_(True)
    g = torch.autograd.grad(f.sdf(x_rand).sum(), x_rand, create_graph=False)[0]
    grad_norm = g.norm(dim=-1)

    views = load_views(scene)
    det   = make_deterministic_rays(views, down=down, device=str(device))
    n_hit_tot = n_tot = 0
    mvs_errs: list[Tensor] = []
    from .geomvs import load_aligned_depths
    mvs_data  = load_aligned_depths(scene)
    V, H_v, W_v = views["images"].shape[0], views["H"], views["W"]
    H_d, W_d = H_v // down, W_v // down
    for v in range(V):
        s = v * det["rays_per_view"]; e = s + det["rays_per_view"]
        o = det["o"][s:e]; d = det["d"][s:e]
        _, t, hit = trace_nograd(f, o, d)
        n_hit_tot += int(hit.sum()); n_tot += hit.numel()
        d_full = mvs_data["depths"][v]; v_full = mvs_data["valid"][v]
        ys_d = torch.arange(H_d); xs_d = torch.arange(W_d)
        ys_f = ((ys_d.float()+0.5)*down-0.5).long().clamp(0,H_v-1)
        xs_f = ((xs_d.float()+0.5)*down-0.5).long().clamp(0,W_v-1)
        yy, xx = torch.meshgrid(ys_f, xs_f, indexing="ij")
        mvs_d = d_full[yy, xx].reshape(-1).to(device)
        mvs_v = v_full[yy, xx].reshape(-1).to(device) & hit
        if mvs_v.any():
            z_cam     = views["c2w"][v, :3, 2].to(device)
            cos_theta = (d * z_cam).sum(-1).abs().clamp(min=1e-6)
            t_target  = mvs_d / cos_theta
            t_pred    = (trace_nograd(f, o, d)[0] - o).mul(d).sum(-1)
            mvs_errs.append((t_pred[mvs_v] - t_target[mvs_v]).abs())

    mvs_err = torch.cat(mvs_errs) if mvs_errs else torch.tensor([])
    stats = {
        "n_sfm_points":         int(sfm_pts.shape[0]),
        "sfm_sdf_mean":         f_sfm.mean().item(),
        "sfm_sdf_median":       sfm_q[0].item(),
        "sfm_sdf_p90":          sfm_q[1].item(),
        "sfm_sdf_p99":          sfm_q[2].item(),
        "cam_outside_frac":     frac_cam_out,
        "cam_f_mean":           f_cam.mean().item(),
        "cam_f_min":            f_cam.min().item(),
        "free_space_viol_frac": frac_free_viol,
        "free_space_viol_mean": mean_free_viol,
        "grad_norm_mean":       grad_norm.mean().item(),
        "grad_norm_max":        grad_norm.max().item(),
        "ray_hit_rate":         n_hit_tot / max(n_tot, 1),
        "mvs_depth_err_mean":   mvs_err.mean().item() if mvs_err.numel() else float("nan"),
        "mvs_depth_err_p50":    mvs_err.quantile(0.5).item() if mvs_err.numel() else float("nan"),
        "mvs_depth_err_p90":    mvs_err.quantile(0.9).item() if mvs_err.numel() else float("nan"),
    }
    print("\n=== geometric quality ===")
    print(f"  SFM (n={stats['n_sfm_points']}):  |f| mean={stats['sfm_sdf_mean']:.4f}  "
          f"med={stats['sfm_sdf_median']:.4f}  p90={stats['sfm_sdf_p90']:.4f}  p99={stats['sfm_sdf_p99']:.4f}")
    print(f"  cams outside:    {stats['cam_outside_frac']:.1%}  "
          f"(f mean={stats['cam_f_mean']:.3f} min={stats['cam_f_min']:.3f})")
    print(f"  free-space viol: {stats['free_space_viol_frac']:.2%}  "
          f"(mean −f={stats['free_space_viol_mean']:.4f})")
    print(f"  |∇f|:            mean={stats['grad_norm_mean']:.3f}  max={stats['grad_norm_max']:.3f}")
    print(f"  hit rate:        {stats['ray_hit_rate']:.1%}")
    print(f"  MVS |Δt|:        mean={stats['mvs_depth_err_mean']:.4f}  "
          f"p50={stats['mvs_depth_err_p50']:.4f}  p90={stats['mvs_depth_err_p90']:.4f}")
    return stats


@torch.no_grad()
def lego_stats(f: FTheta, down: int = 2, label: str = "lego") -> None:
    """Silhouette IoU + eikonal on the full Blender lego training set."""
    device = next(f.parameters()).device
    views  = load_blender_views(down=down)
    det    = make_deterministic_rays(views, down=1, device=str(device))
    total  = det["o"].shape[0]
    batch  = 4096
    tp = fp = fn = 0
    for i in range(0, total, batch):
        o  = det["o"][i:i+batch]; d = det["d"][i:i+batch]; fg = det["fg"][i:i+batch]
        _, _, hit = trace_nograd(f, o, d)
        tp += int((hit & fg).sum()); fp += int((hit & ~fg).sum()); fn += int((~hit & fg).sum())
    sil_iou = tp / max(tp + fp + fn, 1)

    pts = (2 * torch.rand(4096, 3, device=device) - 1) * 1.5
    pts.requires_grad_(True)
    with torch.enable_grad():
        g = torch.autograd.grad(f.sdf(pts).sum(), pts)[0]
    grad_norm = g.norm(dim=-1)
    cams_out  = (f(views["c2w"][:, :3, 3].to(device)) > 0).float().mean().item()

    print(f"\n=== {label} ===")
    print(f"  sil_IoU:   {sil_iou:.4f}  (TP={tp} FP={fp} FN={fn})")
    print(f"  recall:    {tp/(tp+fn+1e-6):.1%}    precision: {tp/(tp+fp+1e-6):.1%}")
    print(f"  |∇f|:      mean={grad_norm.mean():.3f}  max={grad_norm.max():.3f}")
    print(f"  cams_out:  {cams_out:.1%}")
