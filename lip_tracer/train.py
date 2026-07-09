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
from .data import (load_colmap_points, colmap_visibility_counts, colmap_visibility_matrix,
                   load_camera_centers, load_sfm_pairs,
                   load_views, load_blender_views, load_blender_gt_points,
                   make_deterministic_rays, precompute_alt_cameras,
                   precompute_alt_cameras_arccos,
                   alt_cameras_from_pairs, selected_pair_triangulation_angles)
from .loss import (photo_loss, idr_mask_loss, mask_loss_min_sdf, dvr_mask_loss, silhouette_loss, eikonal_loss, cam_free_loss,
                   sfm_sdf_loss, geo_neus_sdf_loss, free_space_loss, sfm_behind_loss, surface_loss, mvs_depth_loss,
                   mvsdf_carving_loss, behind_hit_loss, soft_argmin_photo_loss)
from .model import FTheta, ConvexPotentialLayer, NeuSMLP, RadianceNet, make_model
from .profile import StepProfiler, MemorySnapshot, dump_static_accounting
from .sphere_tracing import trace_unrolled, trace_idr, trace_nograd, get_last_trace_stats
from .bundle_adjustment import InLoopBundleAdjuster


import errno as _errno
import time as _time

# On the shared willow /scratch Lustre project quota, a full quota makes every
# write return ENOSPC (28) or EDQUOT (122). The condition is usually transient
# (other users' jobs finish and free space), so rather than crash a multi-hour
# run we wait for space and retry.
_DISK_FULL_ERRNOS = {_errno.ENOSPC, _errno.EDQUOT}


def _disk_full_retry(fn, *args, _desc="disk write", _wait=30, _max_wait=6 * 3600, **kwargs):
    """Call fn(*args, **kwargs), retrying on a full-quota OSError.

    Waits (blocking) for space to free instead of letting the run die. Other
    OSErrors propagate immediately. Gives up only after _max_wait seconds so a
    genuinely permanent full disk still eventually errors out.
    """
    waited = 0
    delay = _wait
    while True:
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if e.errno not in _DISK_FULL_ERRNOS:
                raise
            _safe_print(
                f"[disk] {_desc} failed: [{e.errno}] {e.strerror}; quota/disk full — "
                f"waiting {delay}s then retrying (waited {waited}s so far)")
            _time.sleep(delay)
            waited += delay
            if waited >= _max_wait:
                _safe_print(f"[disk] {_desc} still failing after {waited}s; giving up.")
                raise
            delay = min(delay * 2, 600)


def _safe_print(msg):
    """print() that never raises on a full disk (the log file lives on the same
    quota). Best-effort: a few short retries, then drop the message."""
    for _ in range(3):
        try:
            print(msg, flush=True)
            return
        except OSError as e:
            if e.errno not in _DISK_FULL_ERRNOS:
                raise
            _time.sleep(5)


class _DiskFullTolerantStream:
    """Wrap a text stream so writes don't crash the process when the log file's
    filesystem is full. Retries briefly, then drops the line and keeps training
    (we never want a failed *log* write to kill a run — only checkpoints are
    worth blocking for, via _disk_full_retry)."""

    def __init__(self, stream):
        self._s = stream

    def write(self, data):
        for _ in range(3):
            try:
                n = self._s.write(data)
                self._s.flush()
                return n
            except OSError as e:
                if e.errno not in _DISK_FULL_ERRNOS:
                    raise
                _time.sleep(5)
        return len(data)  # drop rather than crash

    def flush(self):
        try:
            self._s.flush()
        except OSError as e:
            if e.errno not in _DISK_FULL_ERRNOS:
                raise

    def __getattr__(self, name):
        return getattr(self._s, name)


def _install_disk_full_tolerant_stdio():
    """Make stdout/stderr survive a transient full-quota condition."""
    if not isinstance(sys.stdout, _DiskFullTolerantStream):
        sys.stdout = _DiskFullTolerantStream(sys.stdout)
    if not isinstance(sys.stderr, _DiskFullTolerantStream):
        sys.stderr = _DiskFullTolerantStream(sys.stderr)


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


def _try_compile_model(f):
    """torch.compile(f, dynamic=True) IFF inductor/Triton can actually build here.

    Inductor codegen gcc-compiles a CUDA helper that #include <Python.h>, so on
    nodes without matching python-dev headers (or a broken gcc/CUDA toolchain) the
    very first compiled forward raises InductorError and kills the run. Pre-flight
    a trivial compiled kernel on the GPU so a broken toolchain degrades to the
    eager module (with a warning) instead of crashing training.

    Returns (module, ok, msg): the compiled wrapper if the probe ran, else f.
    """
    if not torch.cuda.is_available():
        return f, False, "cuda unavailable"
    # Inductor gcc-builds a CUDA helper that #include <Python.h>. This cluster has
    # no python3.9-devel (sysconfig include dir lacks Python.h), so point gcc at a
    # copied 3.9 header set via CPATH (gcc searches CPATH after -I). Override the
    # location with LIPTRACER_PY_INCLUDE; falls through to the eager probe-fail
    # path if neither the system headers nor the copy are present.
    import os
    import sysconfig
    inc = sysconfig.get_path("include")
    if not os.path.exists(os.path.join(inc, "Python.h")):
        fallback = os.environ.get(
            "LIPTRACER_PY_INCLUDE",
            "/scratch/_projets_/willow/1-lip-tracer-new/py39_dev_include/python3.9",
        )
        if os.path.exists(os.path.join(fallback, "Python.h")):
            prev = os.environ.get("CPATH", "")
            os.environ["CPATH"] = fallback + (os.pathsep + prev if prev else "")
    try:
        probe = torch.compile(lambda z: z * 2.0 + 1.0, dynamic=True)
        probe(torch.zeros(8, device="cuda"))
        torch.cuda.synchronize()
    except Exception as e:  # InductorError, CalledProcessError, etc.
        first = (str(e).splitlines() or [""])[0][:160]
        return f, False, f"{type(e).__name__}: {first}"
    return torch.compile(f, dynamic=True), True, "ok"


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
    for key in ("feature_maps", "mvs_depth_dir", "mvsformer_depth_dir", "pairs_path"):
        if train_data.get(key) is not None:
            train_data[key] = Path(train_data[key])

    eval_data = dict(data.get("eval", {}))
    if eval_data.get("dtu_eval_dir") is not None:
        eval_data["dtu_eval_dir"] = Path(eval_data["dtu_eval_dir"])
    if eval_data.get("tnt_eval_dir") is not None:
        eval_data["tnt_eval_dir"] = Path(eval_data["tnt_eval_dir"])
    if eval_data.get("bmvs_eval_dir") is not None:
        eval_data["bmvs_eval_dir"] = Path(eval_data["bmvs_eval_dir"])
    if eval_data.get("bmvs_gt_mesh") is not None:
        eval_data["bmvs_gt_mesh"] = Path(eval_data["bmvs_gt_mesh"])

    return Config(
        model=_dataclass_from_dict(ModelConfig, data.get("model", {})),
        trace=_dataclass_from_dict(TraceConfig, data.get("trace", {})),
        init=_dataclass_from_dict(InitConfig, data.get("init", {})),
        train=_dataclass_from_dict(TrainConfig, train_data),
        eval=_dataclass_from_dict(EvalConfig, eval_data),
        scene=Path(data.get("scene", SCENE)),
        out_dir=Path(data.get("out_dir", OUT_DIR)),
    )


def _regen_det_rays(det: dict, c2w: torch.Tensor, K: torch.Tensor) -> None:
    """Rebuild det['o']/['d'] (primary rays) in place from refined poses.

    When in-loop BA moves the cameras, the precomputed primary rays (built once
    from the calibrated poses) go stale: the reprojection uses the live w2c, but
    the SOURCE rays that trace the surface would still sit at the old poses —
    an inconsistent half-update that stops BA from reshaping geometry. We recom-
    pute (o, d) from the stored pixel coords (det['px'], det['py'], det['vi'])
    and the refined c2w/K, preserving fg/gt labels and the fg/bg sampling tables
    (same pixels, so their fg membership is unchanged). Per-view to bound memory.
    """
    px, py, vi = det["px"], det["py"], det["vi"]      # CPU, length V*H_d*W_d
    c2w_c, K_c = c2w.detach().cpu().float(), K.detach().cpu().float()
    o_out = torch.empty_like(det["o"])
    d_out = torch.empty_like(det["d"])
    for v in vi.unique().tolist():
        m  = vi == v
        Kv, cw = K_c[v], c2w_c[v]
        x = (px[m] - Kv[0, 2]) / Kv[0, 0]
        y = (py[m] - Kv[1, 2]) / Kv[1, 1]
        d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        dw = d_cam @ cw[:3, :3].T
        dw = dw / dw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d_out[m] = dw
        o_out[m] = cw[:3, 3]
    det["o"], det["d"] = o_out, d_out


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
    if init_cfg.init_sdf_grid:
        sdf_path = Path(init_cfg.init_sdf_grid)
        print(f"  hull init: loading SDF grid target from {sdf_path}")
        sdf_grid = np.load(sdf_path)
        if sdf_grid.ndim != 3 or len(set(sdf_grid.shape)) != 1:
            raise ValueError(f"init_sdf_grid must be a cubic 3D grid, got shape {sdf_grid.shape}")
        if sdf_grid.shape[0] != init_cfg.hull_res:
            print(f"  hull init: overriding hull_res {init_cfg.hull_res} -> {sdf_grid.shape[0]}")
            init_cfg.hull_res = int(sdf_grid.shape[0])
        if not np.isfinite(sdf_grid).all():
            raise ValueError(f"init_sdf_grid contains non-finite values: {sdf_path}")
        occ = sdf_grid < 0.0
        if not occ.any() or occ.all():
            raise ValueError(f"init_sdf_grid has degenerate occupancy: {sdf_path}")
    else:
        print(f"  hull init: carving at res={init_cfg.hull_res} …")
        roi_bounds = None
        if init_cfg.hull_sfm_roi:
            try:
                sfm_pts = load_colmap_points(scene).numpy()
                sfm_lo = sfm_pts.min(axis=0)
                sfm_hi = sfm_pts.max(axis=0)
                pad = np.maximum(0.15, 0.15 * (sfm_hi - sfm_lo))
                roi_bounds = (np.maximum(sfm_lo - pad, -bound),
                              np.minimum(sfm_hi + pad, bound))
            except (FileNotFoundError, ValueError) as e:
                print(f"  hull init: SFM ROI unavailable ({e})")
        occ = carve(scene=scene, res=init_cfg.hull_res, bound=bound,
                    roi_bounds=roi_bounds, min_views=init_cfg.hull_min_views,
                    border_aware=init_cfg.hull_border_aware)
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
    sfm_pairs = None
    if init_cfg.w_sfm_free > 0 and not is_blender:
        try:
            from .data import load_sfm_pairs
            sfm_pairs = load_sfm_pairs(scene)
            print(f"  hull init: loaded {sfm_pairs[0].shape[0]} SFM sight-line pairs "
                  f"for free-space (w={init_cfg.w_sfm_free})")
        except (FileNotFoundError, ValueError) as e:
            print(f"  hull init: SFM pairs unavailable for free-space ({e})")
    # MLP has no Lipschitz bound → large initial gradients with high-freq PE → need lower lr
    hull_lr = init_cfg.lr if getattr(model_cfg, "architecture", "cpl") == "cpl" else min(init_cfg.lr, 5e-4)
    f = fit_to_hull(occ, bound=bound, steps=init_cfg.steps, batch=init_cfg.batch,
                    lr=hull_lr, cfg=model_cfg,
                    depth_points=depth_pts, w_depth_surface=w_depth_surface,
                    cam_origins=cam_origins_np, w_cam_free=0.0 if is_blender else 1.0,
                    sfm_pairs=sfm_pairs, w_sfm_free=init_cfg.w_sfm_free,
                    sfm_free_eps=init_cfg.sfm_free_eps)
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

    Regresses f_theta directly to the mesh's signed distance field (computed
    by Open3D RaycastingScene) on a near-surface ± uniform-volume mixture —
    same recipe as fit_gt_sdf.py. This gives mm-accurate surface localisation
    at hand-off instead of the binary occupancy that fit_to_hull would produce.

    An occupancy grid is still built from the same SDF (`sd < 0`) for downstream
    AABB sampling (_visual_hull_sample_bounds); the sibling fused.ply union is
    kept solely to tighten that AABB, not as a fit target.

    Mesh is expected in NSVF-COLMAP (un-normalised) frame; we re-normalise it
    into the unit cube the model lives in via scene/bbox.txt.
    """
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

    # --- SDF regression on the same voxel grid hull-init uses ---
    # We already evaluated true SDF (`sd`) at every voxel of the
    # `init_cfg.hull_res^3` grid via RaycastingScene above. Reuse those
    # points + values as the regression dataset (continuous-mesh signed
    # distance, sub-voxel accurate) via the shared regressor.
    f = _fit_sdf_to_grid(pts, sd, model_cfg, init_cfg, bound, tag="colmap init")
    return f, occ


def _fit_sdf_to_grid(
    pts: np.ndarray,
    sd: np.ndarray,
    model_cfg: ModelConfig,
    init_cfg: InitConfig,
    bound: float,
    tag: str = "colmap init",
) -> FTheta:
    """Regress f_θ to a precomputed signed-distance field sampled on a
    [-bound,bound]^3 grid (``pts``, ``sd``).

    Mirrors fit_to_hull's recipe (MSE loss, half-batch biased to the narrow band
    |sdf| < 4·voxel_size, lr=1e-1 — which empirically lands ‖∇f‖≈1) but with the
    continuous-mesh signed distance instead of a discrete EDT. Shared by the
    COLMAP-mesh (fit_colmap_init) and sparse-points (fit_points_init) warm-starts.
    """
    res = init_cfg.hull_res
    voxel_size = 2.0 * bound / max(res - 1, 1)
    band_width = 4.0 * voxel_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_pts_t = torch.from_numpy(pts.astype(np.float32)).to(device)
    grid_sdf_t = torch.from_numpy(sd.astype(np.float32)).to(device)
    N_grid = grid_pts_t.shape[0]
    narrow_idx = torch.nonzero(grid_sdf_t.abs() <= band_width,
                               as_tuple=False).squeeze(-1)
    print(f"  {tag}: SDF target on {N_grid:,} grid pts  "
          f"(narrow band |sdf|<{band_width:.4f}: {len(narrow_idx):,} pts)")

    f = make_model(hidden=model_cfg.hidden, depth=model_cfg.depth,
                   group_size=model_cfg.group_size, activation=model_cfg.activation,
                   input_encoding=model_cfg.input_encoding, multires=model_cfg.multires,
                   architecture=model_cfg.architecture).to(device)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))  # instantiate any lazy buffers

    steps = init_cfg.steps
    batch = init_cfg.batch
    lr    = init_cfg.lr if getattr(model_cfg, "architecture", "cpl") == "cpl" \
            else min(init_cfg.lr, 5e-4)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    n_narrow_b = min(batch // 2, int(narrow_idx.numel())) if narrow_idx.numel() > 0 else 0
    n_uniform_b = batch - n_narrow_b
    print(f"  {tag}: SDF regression  steps={steps}  batch={batch}  "
          f"narrow={n_narrow_b}/{n_narrow_b + n_uniform_b}  lr={lr}")

    for step in range(steps + 1):
        idx_u = torch.randint(0, N_grid, (n_uniform_b,), device=device)
        if n_narrow_b > 0:
            idx_n = narrow_idx[torch.randint(0, narrow_idx.numel(),
                                             (n_narrow_b,), device=device)]
            idx = torch.cat([idx_u, idx_n], dim=0)
        else:
            idx = idx_u
        x = grid_pts_t[idx]
        y = grid_sdf_t[idx]
        pred = f(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                idx_e_n = narrow_idx[torch.randint(0, narrow_idx.numel(),
                                                   (min(8192, narrow_idx.numel()),),
                                                   device=device)] \
                    if narrow_idx.numel() > 0 else None
                idx_e_v = torch.randint(0, N_grid,
                                        (min(8192, N_grid),), device=device)
                if idx_e_n is not None:
                    near_l1 = (f(grid_pts_t[idx_e_n])
                               - grid_sdf_t[idx_e_n]).abs().mean().item()
                else:
                    near_l1 = float("nan")
                vol_l1  = (f(grid_pts_t[idx_e_v])
                           - grid_sdf_t[idx_e_v]).abs().mean().item()
            with torch.enable_grad():
                if idx_e_n is not None:
                    xn = grid_pts_t[idx_e_n].detach().clone().requires_grad_(True)
                    gn = torch.autograd.grad(f(xn).sum(), xn)[0]
                    gn_norm = gn.norm(dim=-1).mean().item()
                else:
                    gn_norm = float("nan")
                xv = grid_pts_t[idx_e_v].detach().clone().requires_grad_(True)
                gv = torch.autograd.grad(f(xv).sum(), xv)[0]
                gv_norm = gv.norm(dim=-1).mean().item()
            print(f"  [{tag}@{step:6d}] loss={loss.item():.6f}  "
                  f"near_l1={near_l1:.6f}  vol_l1={vol_l1:.6f}  "
                  f"|∇f|near={gn_norm:.3f}  |∇f|vol={gv_norm:.3f}", flush=True)
    print(f"  {tag} done")
    return f


def fit_points_init(
    model_cfg: ModelConfig = None,
    init_cfg:  InitConfig  = None,
    scene:     Path        = SCENE,
    bound:     float       = 1.5,
) -> tuple[FTheta, np.ndarray]:
    """Minimal warm-start straight from the COLMAP sparse cloud.

    Reconstructs an initial surface directly from ``sparse_sfm_points.txt``
    (already in the normalized training frame) — no masks, no dense MVS. The
    recipe is the textbook oriented-point → implicit-surface pipeline:

      1. per-point normals via kNN-PCA;
      2. orient each normal toward its nearest camera centre (resolves the
         inside/outside sign), then propagate consistency with open3d's
         tangent-plane MST (and globally flip back if the MST inverted us);
      3. screened-Poisson surface, density-trimmed to drop the balloon
         extrapolation Poisson invents in unobserved regions;
      4. keep the largest connected component;
      5. regress f_θ to that mesh's signed distance (shared _fit_sdf_to_grid).

    Returns (f, occ) like fit_colmap_init, so downstream AABB sampling works.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree
    from .data import load_views

    model_cfg = model_cfg or ModelConfig()
    init_cfg  = init_cfg  or InitConfig()

    pts_cloud = load_colmap_points(scene).numpy().astype(np.float64)
    print(f"  points init: {len(pts_cloud):,} COLMAP points  "
          f"r_mean={np.linalg.norm(pts_cloud, axis=1).mean():.3f}")
    cams = load_views(scene)["c2w"][:, :3, 3].numpy().astype(np.float64)

    # --- oriented normals (kNN-PCA → nearest-camera flip → consistent MST) ---
    knn = init_cfg.points_normal_knn
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cloud))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    n = np.asarray(pcd.normals)
    ci = cKDTree(cams).query(pts_cloud, k=1)[1]
    view = cams[ci] - pts_cloud
    view /= np.linalg.norm(view, axis=1, keepdims=True).clip(1e-9)
    n[(n * view).sum(1) < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(n)
    pcd.orient_normals_consistent_tangent_plane(k=knn)
    n = np.asarray(pcd.normals)
    if ((n * view).sum(1) < 0).mean() > 0.5:   # MST flipped the global sign
        n *= -1
        pcd.normals = o3d.utility.Vector3dVector(n)
    print(f"  points init: normals oriented  mean·camera-dir={(n * view).sum(1).mean():.3f}")

    # --- screened Poisson + density trim + largest component ---
    depth = init_cfg.points_poisson_depth
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1)
    dens = np.asarray(dens)
    q = init_cfg.points_trim_quantile
    if q > 0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, q))
        mesh.remove_unreferenced_vertices()
    tri_ids, tri_counts, _ = mesh.cluster_connected_triangles()
    tri_counts = np.asarray(tri_counts)
    if len(tri_counts) > 1:
        keep = np.asarray(tri_ids) == int(tri_counts.argmax())
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices)
    if len(V) == 0 or len(mesh.triangles) == 0:
        raise ValueError("points init: Poisson reconstruction produced an empty mesh "
                         "(try a lower --points-trim-quantile or smaller --points-poisson-depth)")
    print(f"  points init: Poisson depth={depth} trim_q={q} -> "
          f"V={len(V):,} F={len(mesh.triangles):,}  "
          f"bbox=[{V.min(0).round(3)} .. {V.max(0).round(3)}]")

    # --- signed distance on the regression grid ---
    res = init_cfg.hull_res
    grid = np.linspace(-bound, bound, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    scene_rc = o3d.t.geometry.RaycastingScene()
    scene_rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    sd = scene_rc.compute_signed_distance(o3d.core.Tensor(pts)).numpy()
    occ = (sd < 0).reshape(res, res, res)
    print(f"  points init: occupied {occ.sum()} / {occ.size} ({100 * occ.mean():.2f}%)")

    f = _fit_sdf_to_grid(pts, sd, model_cfg, init_cfg, bound, tag="points init")
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
                  res: int = 400, trace_cfg: TraceConfig | None = None,
                  crop_fg: bool = False, radiance=None) -> None:
    """Sphere-trace 4 training views with Phong shading → PNG strip.

    crop_fg=True (e.g. MVMannequin, where the object fills ~9% of the frame)
    crops every panel to its view's foreground-mask bbox to drop empty margins.

    radiance!=None adds a 4th row: the IDR-style colour MLP's predicted RGB at
    each surface hit (view-dependent, using the per-ray direction as view dir).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sphere_tracing import trace_nograd

    V      = views["c2w"].shape[0]
    H_full = views["H"]; W_full = views["W"]
    ids    = [int(round(i * (V - 1) / 2)) for i in range(3)]
    # 4th view: camera most opposite to view 0 (back of the object)
    cam_positions = views["c2w"][:, :3, 3].cpu().numpy()  # (V, 3)
    dir0 = cam_positions[ids[0]] / (np.linalg.norm(cam_positions[ids[0]]) + 1e-6)
    dots = (cam_positions / (np.linalg.norm(cam_positions, axis=-1, keepdims=True) + 1e-6)) @ dir0
    back_id = int(np.argmin(dots))
    ids.append(back_id)
    down   = max(1, H_full // res)
    H, W   = H_full // down, W_full // down

    light = np.array([0.577, 0.577, 0.577], dtype=np.float32)
    base  = np.array([0.72, 0.72, 0.85],    dtype=np.float32)

    imgs_phong, imgs_color, imgs_hit, imgs_pred, crop_boxes = [], [], [], [], []
    for vi in ids:
        K   = views["K"][vi].cpu().numpy()
        c2w = views["c2w"][vi].cpu().numpy()
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
        n_t = torch.cat(grads, dim=0)
        n_t = n_t / n_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        n = n_t.cpu().numpy()
        hit_np  = hit.cpu().numpy().reshape(H, W, 1)

        # predicted view-dependent colour from the IDR colour MLP (real hits only)
        if radiance is not None:
            pred_parts = []
            with torch.no_grad():
                for i in range(0, xr_all.shape[0], 65536):
                    pred_parts.append(
                        radiance(xr_all[i:i + 65536], n_t[i:i + 65536],
                                 d_t[i:i + 65536]).cpu())
            pred = torch.cat(pred_parts, dim=0).numpy().reshape(H, W, 3)
            imgs_pred.append(np.where(hit_np, pred, 1.0))

        diffuse = np.clip((n * light).sum(-1, keepdims=True), 0, 1)
        shaded  = (0.35 + 0.65 * diffuse) * base
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

        # zoom-to-object: bbox of the foreground mask (12% pad), full frame if empty
        box = (0, H, 0, W)
        if crop_fg:
            ys_fg, xs_fg = np.where(fg)
            if ys_fg.size:
                pad = 0.12
                r0, r1, c0, c1 = ys_fg.min(), ys_fg.max(), xs_fg.min(), xs_fg.max()
                dr, dc = int((r1 - r0) * pad) + 1, int((c1 - c0) * pad) + 1
                box = (max(r0 - dr, 0), min(r1 + dr, H - 1) + 1,
                       max(c0 - dc, 0), min(c1 + dc, W - 1) + 1)
        crop_boxes.append(box)

    if crop_fg:
        def _cb(im, b):
            return im[b[0]:b[1], b[2]:b[3]]
        imgs_phong = [_cb(im, b) for im, b in zip(imgs_phong, crop_boxes)]
        imgs_color = [_cb(im, b) for im, b in zip(imgs_color, crop_boxes)]
        imgs_hit   = [_cb(im, b) for im, b in zip(imgs_hit,   crop_boxes)]
        imgs_pred  = [_cb(im, b) for im, b in zip(imgs_pred,  crop_boxes)]

    n_rows = 4 if imgs_pred else 3
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
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
    if imgs_pred:
        for ax, img in zip(axes[3], imgs_pred):
            ax.imshow(img.clip(0, 1)); ax.axis("off")
        axes[3][1].set_title("Predicted colour (RadianceNet, view-dep)", fontsize=10)
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

        # --- zoom-to-object: the mannequin fills only ~9% of frame height, so
        # crop every panel to the mesh's projected bbox (per view) to drop the
        # large empty margins. Fractional bbox so the normals raster (H_d×W_d)
        # and coverage maps (Hc×Wc) crop consistently.
        mesh_v = np.asarray(mesh.vertices, dtype=np.float64)

        def _obj_crop_frac(vi, pad=0.12):
            """Fractional (fu0, fu1, fv0, fv1) bbox of the mesh in view vi."""
            try:
                K0 = views["K"][vi].numpy().astype(np.float64)
                w2c = np.linalg.inv(views["c2w"][vi].numpy().astype(np.float64))
                Xc = mesh_v @ w2c[:3, :3].T + w2c[:3, 3]
                front = Xc[:, 2] > 1e-6
                if int(front.sum()) < 3:
                    return 0.0, 1.0, 0.0, 1.0
                uv = Xc[front] @ K0.T
                u = uv[:, 0] / uv[:, 2]
                v = uv[:, 1] / uv[:, 2]
                u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
                du, dv = (u1 - u0) * pad, (v1 - v0) * pad
                fu0 = min(max((u0 - du) / W_full, 0.0), 1.0)
                fu1 = min(max((u1 + du) / W_full, 0.0), 1.0)
                fv0 = min(max((v0 - dv) / H_full, 0.0), 1.0)
                fv1 = min(max((v1 + dv) / H_full, 0.0), 1.0)
                if fu1 - fu0 < 1e-3 or fv1 - fv0 < 1e-3:
                    return 0.0, 1.0, 0.0, 1.0
                return fu0, fu1, fv0, fv1
            except Exception:
                return 0.0, 1.0, 0.0, 1.0

        def _crop2obj(arr, frac):
            """Crop arr (H, W[, C]) to fractional bbox frac=(fu0,fu1,fv0,fv1)."""
            fu0, fu1, fv0, fv1 = frac
            H, W = arr.shape[:2]
            c0, c1 = int(np.floor(fu0 * W)), int(np.ceil(fu1 * W))
            r0, r1 = int(np.floor(fv0 * H)), int(np.ceil(fv1 * H))
            c1, r1 = max(c1, c0 + 1), max(r1, r0 + 1)
            return arr[r0:r1, c0:c1]

        show_coverage = train_cfg is not None and trace_cfg is not None
        n_rows = 2 if show_coverage else 1
        fig, axes = plt.subplots(n_rows, len(ids), figsize=(5 * len(ids), 5 * n_rows),
                                 squeeze=False)
        for col, vi in enumerate(ids):
            ax = axes[0, col]
            frac = _obj_crop_frac(vi)
            K = views["K"][vi].numpy().copy()
            K[0, 0] /= d; K[1, 1] /= d
            K[0, 2] = (K[0, 2] + 0.5) / d - 0.5
            K[1, 2] = (K[1, 2] + 0.5) / d - 0.5
            img = render_normals_only(mesh, intersector,
                                      views["c2w"][vi].numpy(), K,
                                      H_d, W_d, 1)
            ax.imshow(np.clip(_crop2obj(img, frac), 0, 1)); ax.axis("off")
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
                for p in _panels:
                    p["arr"] = _crop2obj(p["arr"], frac)
                _save_view_diag_png(_panels, vi, step, run_dir)

                ax_cov = axes[1, col]
                im_cov = ax_cov.imshow(_crop2obj(cov, frac), cmap="viridis",
                                       vmin=0, vmax=max(len(alts), 1))
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
                _occ_slack = getattr(trace_cfg, "occ_depth_slack", 1e-2)
                if occ_md == "from_hit":
                    _, tp, hp = trace_nograd(f, xs + 1e-2 * dp, dp, trace_cfg)
                    not_occl = (~hp) | (tp > dist - _occ_slack)
                else:                                            # pinhole
                    _, tp, hp = trace_nograd(f, op.unsqueeze(0).expand_as(xs),
                                             -dp, trace_cfg)
                    not_occl = hp & (dist <= tp + _occ_slack)
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
                bound: float = 1.5, res: int = 128, mc_level: float = 0.0) -> dict[str, float] | None:
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
    verts, faces, *_ = marching_cubes(vol, level=mc_level, spacing=(spacing,) * 3)
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
    mc_level: float = 0.0,
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
    verts, *_ = marching_cubes(vol, level=mc_level, spacing=(spacing,) * 3)
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


def _extract_world_mesh_for_dtu(f, scale_mat: np.ndarray, device: str, out_ply: Path,
                                bound: float = 1.0, res: int = 384, mc_level: float = 0.0) -> Path | None:
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
    verts_norm, faces, *_ = marching_cubes(vol, level=mc_level, spacing=(spacing,) * 3)
    verts_norm = (verts_norm - bound).astype(np.float32)
    if len(verts_norm) == 0:
        return None

    # largest connected component (by surface area) — drop MC/PE floaters that
    # form in under-observed pockets (in-FG but off-frame in most views), which
    # neither silhouette nor sparse-SFM carving can remove (see _carve_sightlines).
    mesh = trimesh.Trimesh(vertices=verts_norm, faces=faces, process=False)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.area)
        print(f"  [dtu_official] kept largest of {len(parts)} components "
              f"({len(mesh.faces)}/{len(faces)} faces)", flush=True)
    verts_norm = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    if len(verts_norm) == 0:
        return None

    v_h = np.concatenate([verts_norm, np.ones((len(verts_norm), 1), dtype=np.float32)], axis=1)
    verts_world = (scale_mat @ v_h.T).T[:, :3].astype(np.float32)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=verts_world, faces=faces, process=False).export(str(out_ply))
    return out_ply


def _run_blender_official_eval(f, scene: Path, out_dir: Path, device: str,
                               bound: float = 1.5, res: int = 512,
                               n_samples: int = 100_000,
                               mask_crop: bool = True, mask_dilate_px: int = 12,
                               mask_crop_min_ratio: float = 1.0,
                               mask_crop_min_views: int = 1,
                               gt_mesh: Path | None = None,
                               mc_level: float = 0.0) -> dict[str, float] | None:
    """In-training HF-NeuS-style Blender Chamfer with DTU-style fg-mask crop.

    Mirrors evaluation/eval_Blender_official.py (--mask-crop): extract the
    normalized-frame MC mesh (largest connected component), crop faces by
    dilated Blender alpha masks, then compute symmetric Chamfer-L1 against the
    resolved exact GT mesh (data/blender_gt/<scene>.ply).
    """
    repo_root = Path(__file__).resolve().parent.parent
    eval_dir = repo_root / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    try:
        from eval_Blender_official import (
            eval_Blender_official, crop_blender_mesh_by_foreground_masks,
            _resolve_gt_mesh, _scene_from_mesh, NERF_SYNTHETIC_ROOT,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [blender_official] skipped — cannot import evaluator ({e})", flush=True)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    scene_name = _scene_from_mesh(scene) or scene.name
    try:
        gt_path = _resolve_gt_mesh(scene_name, scene, gt_mesh,
                                   repo_root / "data" / "blender_gt", NERF_SYNTHETIC_ROOT)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [blender_official] skipped — no GT mesh ({e})", flush=True)
        return None

    # MC mesh in the normalized Blender frame (identity scale_mat → no world xform)
    pred_ply = _extract_world_mesh_for_dtu(
        f, np.eye(4, dtype=np.float32), device, out_dir / "pred_norm_mesh.ply",
        bound=bound, res=res, mc_level=mc_level,
    )
    if pred_ply is None:
        return None

    crop_stats = {"enabled": False}
    eval_ply = pred_ply
    if mask_crop:
        try:
            eval_ply, crop_stats = crop_blender_mesh_by_foreground_masks(
                pred_ply, scene, out_dir / "pred_norm_mesh_fgcrop.ply",
                dilate_px=mask_dilate_px, min_ratio=mask_crop_min_ratio,
                min_views=mask_crop_min_views,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [blender_official] mask crop failed: {e}", flush=True)
            return None

    try:
        metrics = eval_Blender_official(eval_ply, gt_path, n_points=n_samples)
    except Exception as e:  # noqa: BLE001
        print(f"  [blender_official] chamfer failed: {e}", flush=True)
        return None
    metrics["foreground_mask_crop"] = crop_stats
    (out_dir / "blender_official.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _dilate_masks_disk_np(masks: np.ndarray, radius: int) -> np.ndarray:
    masks = masks.astype(bool)
    if radius <= 0:
        return masks
    from skimage import morphology as morph

    elem = morph.disk(int(radius))
    return np.stack([morph.binary_dilation(m, elem) for m in masks]).astype(bool)


def _crop_dtu_mesh_by_foreground_masks(
    mesh_ply: Path,
    scene: Path,
    out_dir: Path,
    dilate_px: int = 12,
    min_ratio: float = 1.0,
    min_views: int = 1,
    chunk: int = 32768,
) -> tuple[Path, dict]:
    """NeuralWarp-style eval crop: 12px-dilated DTU masks, projected face centroids."""
    import trimesh

    views = load_views(scene, down=1)
    masks = views["masks"].detach().cpu().numpy().astype(bool)
    masks = _dilate_masks_disk_np(masks, dilate_px)
    K = views["K"].detach().cpu().numpy().astype(np.float64)
    c2w = views["c2w"].detach().cpu().numpy().astype(np.float64)
    w2c = np.linalg.inv(c2w)
    R = w2c[:, :3, :3]
    tcw = w2c[:, :3, 3]
    H, W = int(views["H"]), int(views["W"])

    scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
    s = float(scale_mat[0, 0])
    t = scale_mat[:3, 3].astype(np.float64)

    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms)
    mesh.remove_unreferenced_vertices()
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(faces) == 0:
        raise ValueError(f"mesh has no faces: {mesh_ply}")
    cent_norm = (verts[faces].mean(axis=1) - t) / s

    V = masks.shape[0]
    keep = np.zeros(len(faces), dtype=bool)
    min_views = max(1, int(min_views))
    min_ratio = float(min_ratio)
    seen_mean_acc = 0.0
    fg_mean_acc = 0.0

    print(f"  [dtu_official] mask crop: {len(faces):,} faces, {V} masks, "
          f"dilate={dilate_px}px ratio={min_ratio:g}", flush=True)
    for start in range(0, len(cent_norm), chunk):
        end = min(start + chunk, len(cent_norm))
        p = cent_norm[start:end]
        xc = np.einsum("vij,nj->vni", R, p) + tcw[:, None, :]
        uvh = np.einsum("vij,vnj->vni", K, xc)
        z = uvh[..., 2]
        denom = np.where(z > 1e-6, z, 1.0)
        u = uvh[..., 0] / denom
        v = uvh[..., 1] / denom
        in_b = (z > 1e-6) & (xc[..., 2] > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = np.rint(u).astype(np.int64).clip(0, W - 1)
        vi = np.rint(v).astype(np.int64).clip(0, H - 1)

        fg = np.zeros_like(in_b, dtype=bool)
        for view_idx in range(V):
            fg[view_idx] = masks[view_idx, vi[view_idx], ui[view_idx]]
        seen = in_b.sum(axis=0)
        fg_seen = (fg & in_b).sum(axis=0)
        ratio = np.divide(
            fg_seen, np.maximum(seen, 1),
            out=np.zeros(len(seen), dtype=np.float32),
            where=seen > 0,
        )
        keep[start:end] = (seen >= min_views) & (ratio >= min_ratio)
        seen_mean_acc += float(seen.sum())
        fg_mean_acc += float(fg_seen.sum())

    if not keep.any():
        raise RuntimeError("foreground-mask crop removed every DTU mesh face")

    cropped = trimesh.Trimesh(vertices=verts, faces=faces[keep], process=False)
    cropped.remove_unreferenced_vertices()
    out_ply = out_dir / f"{mesh_ply.stem}_fgmask_dilate{int(dilate_px)}.ply"
    cropped.export(str(out_ply))
    stats = {
        "enabled": True,
        "dilate_px": int(dilate_px),
        "min_ratio": min_ratio,
        "min_views": min_views,
        "faces_before": int(len(faces)),
        "faces_after": int(keep.sum()),
        "face_keep_fraction": float(keep.mean()),
        "vertices_before": int(len(verts)),
        "vertices_after": int(len(cropped.vertices)),
        "seen_views_mean": float(seen_mean_acc / max(len(faces), 1)),
        "fg_views_mean": float(fg_mean_acc / max(len(faces), 1)),
        "mesh": str(out_ply),
    }
    print(f"  [dtu_official] mask crop faces {stats['faces_before']:,} → "
          f"{stats['faces_after']:,} ({100.0 * stats['face_keep_fraction']:.1f}%)",
          flush=True)
    return out_ply, stats


def _run_mvmannequin_official_eval(f, scene: Path, out_dir: Path,
                                   bound: float, res: int, device: str,
                                   mc_level: float = 0.0) -> dict[str, float] | None:
    """In-training MVMannequin Chamfer eval. Reproduces Inria-Morpheo's official
    protocol exactly: z>0.05 m slice + largest CC + ICP-p2l (Tukey, 25 mm)
    + pysdf point-to-mesh distance + clamp@100 mm.

    Two-stage: (1) extract pred mesh in WORLD frame from this venv, (2) subprocess
    to the pixi env which has pysdf + open3d.
    """
    eval_script = Path(__file__).resolve().parent.parent / "analysis/eval_mvmannequin_official.py"
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
        v_norm, faces, *_ = marching_cubes(vol, level=mc_level, spacing=(spacing,) * 3)
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
                           out_dir: Path, scene: Path | None = None,
                           mask_crop: bool = True, mask_dilate_px: int = 12,
                           mask_crop_min_ratio: float = 1.0,
                           mask_crop_min_views: int = 1) -> dict[str, float] | None:
    eval_script = Path(__file__).resolve().parent.parent / "DTUeval-python" / "eval.py"
    if not eval_script.exists():
        print(f"  [dtu_official] skipped — missing {eval_script}", flush=True)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    crop_stats = {"enabled": False}
    if mask_crop:
        if scene is None:
            print("  [dtu_official] mask crop skipped — scene unavailable", flush=True)
        else:
            try:
                mesh_ply, crop_stats = _crop_dtu_mesh_by_foreground_masks(
                    mesh_ply, scene, out_dir,
                    dilate_px=mask_dilate_px,
                    min_ratio=mask_crop_min_ratio,
                    min_views=mask_crop_min_views,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [dtu_official] mask crop failed: {e}", flush=True)
                return None

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
    payload = {"accuracy": acc, "completeness": comp, "chamfer": chamfer,
               "mesh": str(mesh_ply), "foreground_mask_crop": crop_stats}
    (out_dir / "dtu_official.json").write_text(json.dumps(payload, indent=2))
    return payload


def _infer_tnt_scene_name(scene: Path, eval_dir: Path | None,
                          explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    candidates: set[str] = {scene.name}
    for part in scene.parts:
        candidates.add(part)
        candidates.add(part.capitalize())
    if eval_dir is not None and eval_dir.exists():
        candidates.add(eval_dir.name)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tnt_eval"))
        from config import scenes_tau_dict
        for name in scenes_tau_dict:
            lowered = name.lower()
            if name in candidates or any(lowered in c.lower() for c in candidates):
                return name
    except Exception:  # noqa: BLE001
        pass
    return None


def _resolve_tnt_gt_dir(eval_dir: Path, scene_name: str) -> Path | None:
    direct = eval_dir
    nested = eval_dir / scene_name
    for cand in (nested, direct):
        if all((cand / f).exists() for f in (
            f"{scene_name}.ply",
            f"{scene_name}_trans.txt",
            f"{scene_name}.json",
            f"{scene_name}_COLMAP_SfM.log",
        )):
            return cand
    return None


def _extract_tnt_eval_points(f, scene: Path, out_dir: Path, device: str,
                             bound: float = 1.5, res: int = 512,
                             n_samples: int = 2_000_000,
                             mc_level: float = 0.0) -> Path | None:
    """Extract current SDF surface in TnT world frame and sample eval points."""
    import open3d as o3d
    from skimage.measure import marching_cubes

    bbox_p = scene / "bbox.txt"
    if not bbox_p.exists():
        print(f"  [tnt_official] skipped — missing {bbox_p}", flush=True)
        return None
    bbox = np.loadtxt(bbox_p, dtype=np.float32)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    lo_n = (bbox[:3] - center) / scale
    hi_n = (bbox[3:6] - center) / scale
    pad = 0.10 * (hi_n - lo_n)
    lo_n = np.clip(lo_n - pad, -bound, bound)
    hi_n = np.clip(hi_n + pad, -bound, bound)

    gx = torch.linspace(float(lo_n[0]), float(hi_n[0]), res, device=device)
    gy = torch.linspace(float(lo_n[1]), float(hi_n[1]), res, device=device)
    gz = torch.linspace(float(lo_n[2]), float(hi_n[2]), res, device=device)
    grid = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), dim=-1).reshape(-1, 3)
    print(f"  [tnt_official] MC eval {grid.shape[0]:,} pts at res={res} "
          f"scene-box(norm) lo={lo_n} hi={hi_n}", flush=True)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i + 65536]) for i in range(0, len(grid), 65536)])
    vol = vals.reshape(res, res, res).detach().cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        print(f"  [tnt_official] n/a (surface not in bounds; "
              f"f in [{vol.min():.4f},{vol.max():.4f}])", flush=True)
        return None

    verts, faces, *_ = marching_cubes(vol, level=mc_level)
    span = (hi_n - lo_n).astype(np.float32)
    verts = lo_n.astype(np.float32) + verts.astype(np.float32) / (res - 1) * span
    verts_w = verts * scale + center

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_w.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_p = out_dir / "mesh.ply"
    points_p = out_dir / "points.ply"
    o3d.io.write_triangle_mesh(str(mesh_p), mesh)
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_samples))
    o3d.io.write_point_cloud(str(points_p), pcd)
    print(f"  [tnt_official] wrote {mesh_p.name} and {points_p.name} "
          f"({int(n_samples):,} pts)", flush=True)
    return points_p


def _run_tnt_official_eval(points_ply: Path, scene: Path, eval_dir: Path,
                           scene_name: str, out_dir: Path,
                           frame: str = "colmap-pose") -> dict[str, float] | None:
    eval_script = Path(__file__).resolve().parent.parent / "analysis" / "eval_tnt_official.py"
    if not eval_script.exists():
        print(f"  [tnt_official] skipped — missing {eval_script}", flush=True)
        return None
    gt_dir = _resolve_tnt_gt_dir(eval_dir, scene_name)
    if gt_dir is None:
        print(f"  [tnt_official] skipped — missing GT assets for {scene_name} under {eval_dir}",
              flush=True)
        return None

    try:
        sys.path.insert(0, str(eval_script.parent))
        from eval_tnt_official import run_tnt_eval
        run_tnt_eval(points_ply, scene, gt_dir, scene_name, out_dir, frame=frame)
        payload = json.loads((out_dir / "fscore.json").read_text())
        return {
            "precision": float(payload["precision"]),
            "recall": float(payload["recall"]),
            "fscore": float(payload["fscore"]),
            "tau": float(payload["tau"]),
        }
    except Exception as e:  # noqa: BLE001
        print(f"  [tnt_official] failed: {e}; see {out_dir}", flush=True)
        return None


def _run_bmvs_official_eval(f, gt_mesh_path: Path, scale_mat: np.ndarray, out_dir: Path,
                            device: str, bound: float, res: int, n_samples: int,
                            protocol: str, ground_axis: int, ground_value: float | None,
                            scene_dir: Path | None = None, gt_space: str = "normalized",
                            metric: str = "auto", mask_crop: bool = False,
                            mask_dilate: int = 12, mc_level: float = 0.0) -> dict[str, float] | None:
    """In-training BlendedMVS Chamfer eval. Mirrors analysis/eval_bmvs_chamfer.py.

    protocol="probesdf" reproduces ProbeSDF's BMVS eval: pysdf distance from
    each mesh's vertices to the other mesh, ignoring distances >= 0.025 in the
    normalized frame. protocol="volsdf" keeps the sampled B.2-style recipe.
    """
    from skimage.measure import marching_cubes
    import trimesh

    eval_dir = Path(__file__).resolve().parent.parent / "analysis"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    try:
        from eval_bmvs_chamfer import (_load_mesh, _transform_mesh, _largest_component,
                                       _drop_below_plane, _sample, _nn_metrics,
                                       _dists_to_mesh_pysdf, _dists_to_mesh,
                                       _dists_to_points, _mask_hull_keep)
    except Exception as e:  # noqa: BLE001
        print(f"  [bmvs_official] skipped — cannot import evaluator ({e})", flush=True)
        return None

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i + 4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).detach().cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None
    spacing = 2 * bound / (res - 1)
    verts, faces, *_ = marching_cubes(vol, level=mc_level, spacing=(spacing,) * 3)
    verts = (verts - bound).astype(np.float32)
    if len(verts) == 0 or len(faces) == 0:
        return None
    pred_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    gt_mesh = _load_mesh(gt_mesh_path)
    if gt_space == "world":
        gt_mesh = _transform_mesh(gt_mesh, np.linalg.inv(scale_mat))

    if protocol == "volsdf":
        if ground_value is not None:
            pred_mesh = _drop_below_plane(pred_mesh, ground_axis, ground_value)
        pred_mesh = _largest_component(pred_mesh)
        max_dist = None
    else:
        max_dist = 0.025

    # NB: `probesdf` would use pysdf's surface-distance primitive, but pysdf is
    # not in the training venv (3.9, no build headers). The earlier BMVS runs all
    # used open3d point-to-mesh, so default `auto`->point-to-mesh keeps numbers
    # comparable. Pass metric="probesdf" explicitly only where pysdf is available.
    metric = "point-to-mesh" if metric == "auto" else metric

    if metric == "probesdf":
        pred_pts = np.asarray(pred_mesh.vertices, dtype=np.float32)
        gt_pts = np.asarray(gt_mesh.vertices, dtype=np.float32)
    else:
        pred_pts = _sample(pred_mesh, n_samples, 0)
        gt_pts = _sample(gt_mesh, n_samples, 1)
    mask_stats = None
    if mask_crop and scene_dir is not None:
        keep_p, sp = _mask_hull_keep(pred_pts, scene_dir, mask_dilate, 1, "all", 0.95)
        keep_g, sg = _mask_hull_keep(gt_pts, scene_dir, mask_dilate, 1, "all", 0.95)
        pred_pts, gt_pts = pred_pts[keep_p], gt_pts[keep_g]
        mask_stats = {"pred": sp, "gt": sg}
    if metric == "probesdf":
        try:
            acc = _dists_to_mesh_pysdf(pred_pts, gt_mesh)
            comp = _dists_to_mesh_pysdf(gt_pts, pred_mesh)
        except Exception as e:  # noqa: BLE001
            print(f"  [bmvs_official] skipped — ProbeSDF pysdf metric unavailable ({e})", flush=True)
            return None
    elif metric == "point-to-mesh":
        acc = _dists_to_mesh(pred_pts, gt_mesh)
        comp = _dists_to_mesh(gt_pts, pred_mesh)
    else:
        acc = _dists_to_points(pred_pts, gt_pts)
        comp = _dists_to_points(gt_pts, pred_pts)
    metrics = _nn_metrics(acc, comp, max_dist, ignore=True)
    raw_units_per_norm = float(np.linalg.norm(scale_mat[:3, :3], axis=0)[0])
    metrics.update({
        "protocol": protocol,
        "metric": metric,
        "gt_space": gt_space,
        "mask_crop": mask_stats,
        "gt_mesh": str(gt_mesh_path),
        "n_pred_points": int(len(pred_pts)),
        "n_gt_points": int(len(gt_pts)),
        "raw_units_per_normalized_unit": raw_units_per_norm,
        "chamfer_raw_units": metrics["chamfer"] * raw_units_per_norm,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bmvs_chamfer.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


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
          run_dir: Path | None = None, view_keep=None, lr_warm_restart: bool = False) -> Path:
    """Train the 1-Lip SDF and save checkpoints to a timestamped run directory.

    Returns the path to the final checkpoint.
    """
    import os
    # Survive a transient full shared-quota (see _disk_full_retry): don't let a
    # failed log/checkpoint write kill a multi-hour run.
    _install_disk_full_tolerant_stdio()
    cfg = cfg or Config()
    # unpack for convenience
    model_cfg = cfg.model
    trace_cfg = cfg.trace
    init_cfg  = cfg.init
    train_cfg = cfg.train
    eval_cfg  = cfg.eval
    scene     = cfg.scene
    out_dir   = cfg.out_dir

    if not train_cfg.use_masks:
        mask_loss_weights = {
            "w_mask_fg": train_cfg.w_mask_fg,
            "w_mask_bg": train_cfg.w_mask_bg,
            "w_idr_mask": train_cfg.w_idr_mask,
            "w_sil": train_cfg.w_sil,
        }
        active_mask_losses = [k for k, v in mask_loss_weights.items() if v > 0]
        if active_mask_losses:
            raise ValueError(
                "train.use_masks=False is incompatible with mask-supervised losses: "
                + ", ".join(active_mask_losses)
                + ". Set those weights to 0 or enable masks."
            )

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
        # Preemption recovery for explicit-resume jobs: an explicitly passed
        # --resume points at a FIXED checkpoint (e.g. checkpoint_final.pt). On a
        # SLURM requeue we must not rewind to it — that throws away every step
        # this run has logged since. Prefer the run's own checkpoint_latest.pt
        # when it has advanced past the explicit target.
        if slurm_restart_cnt > 0:
            for cand in (run_dir / "ckpt" / "checkpoint_latest.pt",
                         run_dir / "checkpoint_latest.pt"):
                if cand.exists() and cand.resolve() != resume.resolve():
                    try:
                        cand_step = int(torch.load(cand, map_location="cpu").get("step", -1))
                        res_step  = int(torch.load(resume, map_location="cpu").get("step", -1))
                    except Exception:
                        cand_step = res_step = -1
                    if cand_step > res_step:
                        print(f"  [preemption] SLURM_RESTART_COUNT={slurm_restart_cnt}: "
                              f"explicit --resume {resume.name} (step {res_step}) is stale; "
                              f"using {cand.name} (step {cand_step}) instead")
                        resume = cand
                    break
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
    # Atomic write: a failed (ENOSPC) write to a tmp file then replace() can never
    # truncate an existing-good config.json to 0 bytes (that corruption broke the
    # hotdog resume — write_text opens with truncate before the failing write).
    _cfg_tmp = run_dir / "config.json.tmp"
    _disk_full_retry(_cfg_tmp.write_text, json.dumps(full_cfg, indent=2), _desc="config.json")
    _cfg_tmp.replace(run_dir / "config.json")
    print(f"  run dir → {run_dir}")

    if use_wandb:
        import wandb
        wandb.init(project="1lip-tracer", name=run_dir.name, config=full_cfg, dir=str(run_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    # Controls whether load_views() zeroes the RGB background via the dataset
    # mask. Set before any load so a truly mask-free run keeps real backgrounds.
    from . import data as _data
    _data.BAKE_BACKGROUND = train_cfg.bake_background
    if not train_cfg.bake_background:
        print("  [mask-free] keeping real photographed background (no img[~msk]=0 bake)")
    if train_cfg.use_blender:
        if view_keep is not None:
            raise NotImplementedError("view_keep not supported for Blender scenes")
        views = load_blender_views(scene=scene, down=train_cfg.down)
    else:
        views = load_views(scene, view_keep=view_keep)
    if not train_cfg.use_masks and "masks" in views:
        views = dict(views)
        views["masks"] = torch.ones_like(views["masks"], dtype=torch.bool)
        print("  [mask-free] ignoring dataset masks for training, ray labels, and SFM visibility")
    images      = views["images"].to(device).half()
    masks       = views["masks"].to(device) if train_cfg.use_masks and "masks" in views else None
    c2w_all     = views["c2w"].to(device)
    K_all       = views["K"].to(device)
    H, W        = views["H"], views["W"]
    V           = images.shape[0]
    print(f"  views: {V} cameras  {H}×{W}")
    origins_all = c2w_all[:, :3, 3]
    w2c_all     = torch.linalg.inv(c2w_all)

    # Bundle adjustment is a separate post-training pass — block-coordinate
    # refinement of (θ, φ) from a converged checkpoint. It lives entirely in
    # bundle_adjustment.run_bundle_adjustment; this training loop only ever
    # optimises θ with the cameras held at their calibrated poses.

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

    sfm_vis = None   # (V, P) per-view COLMAP visibility for Geo-Neus per-view SDF loss
    is_mvm = False
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
        # scale_mat + scan id for periodic official DTU eval
        dtu_scale_mat  = None
        dtu_scan_id    = None
        if eval_cfg.dtu_eval_dir is not None and eval_cfg.dtu_official_freq > 0:
            import re
            m = re.search(r"scan(\d+)", str(scene))
            if m:
                dtu_scan_id = int(m.group(1))
                dtu_scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
        # MVMannequin scene auto-detect: presence of GT mesh next to cameras.npz.
        # Reuses dtu_official_freq for cadence; gt_mesh.ply lives in normalized frame.
        is_mvm = (scene / "gt_mesh.ply").exists() and (scene / "cameras.npz").exists() and not dtu_scan_id
        tnt_scene_name = None
        if eval_cfg.tnt_eval_dir is not None and eval_cfg.tnt_official_freq > 0:
            tnt_scene_name = _infer_tnt_scene_name(
                scene, eval_cfg.tnt_eval_dir, eval_cfg.tnt_official_scene)
            if tnt_scene_name is None:
                print(f"  [tnt_official] disabled — cannot infer official scene name from {scene}")
            elif _resolve_tnt_gt_dir(eval_cfg.tnt_eval_dir, tnt_scene_name) is None:
                print(f"  [tnt_official] disabled — missing GT assets for {tnt_scene_name} "
                      f"under {eval_cfg.tnt_eval_dir}")
                tnt_scene_name = None
            else:
                print(f"  [tnt_official] {tnt_scene_name}: every "
                      f"{eval_cfg.tnt_official_freq} steps  res={eval_cfg.tnt_official_res}  "
                      f"n={eval_cfg.tnt_official_n_samples:,}  frame={eval_cfg.tnt_official_frame}")
        # BlendedMVS scene auto-detect: cameras_sphere.npz frame + a resolvable GT mesh.
        bmvs_scale_mat = None
        bmvs_gt_mesh_path = None
        if eval_cfg.bmvs_official_freq > 0:
            cam = scene / "cameras_sphere.npz"
            if not cam.exists():
                print(f"  [bmvs_official] disabled — missing {cam}")
            else:
                gtp = eval_cfg.bmvs_gt_mesh
                if gtp is None and eval_cfg.bmvs_eval_dir is not None:
                    _adir = Path(__file__).resolve().parent.parent / "analysis"
                    if str(_adir) not in sys.path:
                        sys.path.insert(0, str(_adir))
                    try:
                        from eval_bmvs_chamfer import GT_REL
                    except Exception:  # noqa: BLE001
                        GT_REL = {}
                    rel = GT_REL.get(scene.name)
                    if rel is not None:
                        cand = eval_cfg.bmvs_eval_dir / rel
                        if not cand.exists():
                            alt = eval_cfg.bmvs_eval_dir / "GT_meshes" / rel
                            cand = alt if alt.exists() else cand
                        gtp = cand
                if gtp is None or not Path(gtp).exists():
                    print("  [bmvs_official] disabled — GT mesh not found "
                          "(set eval.bmvs_gt_mesh, or eval.bmvs_eval_dir for a known bmvs_* scene)")
                else:
                    bmvs_gt_mesh_path = Path(gtp)
                    bmvs_scale_mat = np.load(cam)["scale_mat_0"].astype(np.float64)
                    print(f"  [bmvs_official] {scene.name}: every {eval_cfg.bmvs_official_freq} steps  "
                          f"res={eval_cfg.bmvs_official_res}  n={eval_cfg.bmvs_official_n_samples:,}  "
                          f"protocol={eval_cfg.bmvs_official_protocol}  gt={bmvs_gt_mesh_path.name}")
        if train_cfg.w_sfm > 0 or train_cfg.w_geo_sdf > 0:
            try:
                sfm_pts_all = load_colmap_points(scene)
                if train_cfg.sfm_min_views > 0:
                    sfm_counts = colmap_visibility_counts(sfm_pts_all, views)
                    sfm_pts = sfm_pts_all[sfm_counts >= train_cfg.sfm_min_views].to(device)
                    print(f"  loaded {sfm_pts.shape[0]}/{sfm_pts_all.shape[0]} SFM points "
                          f"(>={train_cfg.sfm_min_views} views)")
                else:
                    sfm_pts = sfm_pts_all.to(device)
                    print(f"  loaded {sfm_pts.shape[0]} SFM points")
                # Geo-Neus per-view SDF: (V, P) visibility = their view_id.npy.
                sfm_vis = colmap_visibility_matrix(sfm_pts.cpu(), views).to(device)
                _ppv = sfm_vis.sum(1)
                print(f"  per-view SDF supervision: {sfm_vis.shape[0]} views, "
                      f"{_ppv.float().mean():.0f} pts/view (min {_ppv.min()}, max {_ppv.max()})")
            except FileNotFoundError as e:
                print(f"  sparse_sfm_points.txt not found — w_sfm/w_geo_sdf disabled ({e})")
                train_cfg = dataclasses.replace(train_cfg, w_sfm=0.0, w_geo_sdf=0.0)
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
    elif init_cfg.init in ("colmap", "points"):
        _init_fn = fit_points_init if init_cfg.init == "points" else fit_colmap_init
        f, hull_occ = _init_fn(
            model_cfg=model_cfg, init_cfg=init_cfg, scene=scene, bound=bound,
        )
        mvs_sdf_bounds_np = _visual_hull_sample_bounds(hull_occ, bound)
        if mvs_sdf_bounds_np is not None:
            lo_np, hi_np = mvs_sdf_bounds_np
            print(f"  mvs-sdf samples: {init_cfg.init}-hull AABB "
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
    # --- optional IDR-style view-dependent colour MLP ---
    radiance = None
    if train_cfg.w_rgb > 0:
        radiance = RadianceNet(hidden=train_cfg.rgb_hidden, depth=train_cfg.rgb_depth,
                               view_dep=train_cfg.rgb_view_dep,
                               input_encoding=train_cfg.rgb_input_encoding,
                               multires=train_cfg.rgb_multires).to(device)
        if _resume_ckpt is not None and "radiance" in _resume_ckpt:
            radiance.load_state_dict(_resume_ckpt["radiance"])
        print(f"  radiance MLP: hidden={train_cfg.rgb_hidden} depth={train_cfg.rgb_depth} "
              f"view_dep={train_cfg.rgb_view_dep} "
              f"enc={train_cfg.rgb_input_encoding}"
              f"{f'(L={train_cfg.rgb_multires})' if train_cfg.rgb_input_encoding == 'pe' else ''} "
              f"params={sum(p.numel() for p in radiance.parameters()):,}")
    total_params = sum(p.numel() for p in f.parameters())
    n_cpl = sum(1 for m in f.net if isinstance(m, ConvexPotentialLayer)) if hasattr(f, "net") else 0
    arch_tag = getattr(f, "architecture", "mlp")
    act_tag = getattr(f, "activation", "-")
    print(f"  model: hidden={f.hidden}  depth={n_cpl or f.depth}  params={total_params:,}  "
          f"arch={arch_tag}  act={act_tag}  enc={f.input_encoding}  multires={f.multires}")
    # f_fwd: compiled view of f for the hot path (trace + photo loss). Shares
    # parameters/buffers with f, so the optimiser (over f.parameters()) and all
    # checkpointing/introspection keep using the raw `f` — only forward calls in
    # the inner loop route through the fused graph. dynamic=True: the compacted
    # trace passes variable-size batches each iteration.
    # Differentiable normals (∇f traced with create_graph=True, used when the NCC
    # loss optimizes orientation — w_ncc_normal>0 or --no-ncc-detach-normals) are
    # incompatible with torch.compile's donated-buffer optimization, whose compiled
    # backward asserts create_graph=False. Without this the first such backward
    # aborts: "non-empty donated buffers requires create_graph=False".
    _need_diff_normal_compile = (
        train_cfg.w_ncc_normal > 0
        or (train_cfg.w_ncc > 0 and not train_cfg.ncc_detach_normals)
    )
    if getattr(train_cfg, "ncc_attach_normal_point", False):
        if _need_diff_normal_compile:
            print("  [ncc] attach_normal_point ON — normal evaluated at differentiable "
                  "x_theta (position→normal gradient flows; 'detach nothing')")
        else:
            print("  [warn] --ncc-attach-normal-point set but no differentiable normal "
                  "path active (needs --no-ncc-detach-normals or w_ncc_normal>0) — ignored")
    if getattr(train_cfg, "compile", True):
        if _need_diff_normal_compile:
            # Differentiable normals compute ∇(NCC) through x_theta with
            # torch.autograd.grad(create_graph=True), i.e. a double backward
            # through f. torch.compile's aot_autograd raises "does not currently
            # support double backward", so force eager for these configs.
            f_fwd = f
            print("  torch.compile: disabled (differentiable normals need double "
                  "backward, unsupported by aot_autograd) — running eager")
        else:
            f_fwd, _ok, _msg = _try_compile_model(f)
            if _ok:
                print("  torch.compile(dynamic=True) enabled for hot-path f")
            else:
                print(f"  [warn] torch.compile unavailable ({_msg}) — running eager")
    else:
        f_fwd = f
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
    # mask-free object-focused sampling: trace det rays vs the init SDF once and
    # set fg = hit. The carved init hull is a conservative superset of the object,
    # so true-surface rays are covered; pair with fg_fraction<1 + force_fg_bg_split
    # for a full-frame escape valve. No segmentation mask involved.
    if train_cfg.init_hit_sampling:
        # cheap occ-style trace — fg is a boolean, no need for primary-trace precision
        fg_trace_cfg = dataclasses.replace(
            trace_cfg, iters=trace_cfg.occ_iters,
            newton_steps=trace_cfg.occ_newton_steps,
            eps=max(trace_cfg.occ_eps, trace_cfg.eps))
        hit_fg = torch.empty(total_rays, dtype=torch.bool)
        chunk  = 1 << 21
        with torch.no_grad():
            for s in range(0, total_rays, chunk):
                e = min(s + chunk, total_rays)
                _, _, hit_c = trace_nograd(f_fwd, det["o"][s:e].to(device),
                                           det["d"][s:e].to(device), fg_trace_cfg)
                hit_fg[s:e] = hit_c.cpu()
                print(f"    init-hit trace {e}/{total_rays}", flush=True)
        n_hit = int(hit_fg.sum())
        print(f"  init-hit sampling: {n_hit}/{total_rays} rays hit init SDF "
              f"({100 * n_hit / max(total_rays, 1):.1f}%) → fg")
        if n_hit == 0:
            print("  [init-hit] WARNING: no rays hit init SDF — keeping original fg")
        else:
            det["fg"] = hit_fg
    fg_frac = det["fg"].float().mean().item()
    print(f"  fg rays: {det['fg'].sum():.0f}/{total_rays} ({fg_frac:.1%})"
          f"{'  [NO MASKS]' if masks is None else ''}")
    fg_idx = det["fg"].nonzero(as_tuple=True)[0]
    bg_idx = (~det["fg"]).nonzero(as_tuple=True)[0]
    _bg_loss_active = any(w > 0 for w in (
        train_cfg.w_mask_bg, train_cfg.w_idr_mask, train_cfg.w_sil,
        train_cfg.w_behind_hit, train_cfg.w_ray_free,
    ))
    if not 0.0 <= train_cfg.fg_fraction <= 1.0:
        raise ValueError(f"fg_fraction must be in [0, 1], got {train_cfg.fg_fraction}")
    if len(bg_idx) == 0 or (not _bg_loss_active and not train_cfg.force_fg_bg_split):
        n_fg = train_cfg.batch
        n_bg = 0
    elif len(fg_idx) == 0:
        n_fg = 0
        n_bg = train_cfg.batch
    else:
        n_fg = int(train_cfg.batch * train_cfg.fg_fraction)
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

    _vsel = getattr(train_cfg, "view_selection", "nearest")
    if _vsel == "pairs_file":
        _pp = getattr(train_cfg, "pairs_path", None)
        if _pp is None:
            raise ValueError("view_selection='pairs_file' requires train.pairs_path")
        alt_nn = alt_cameras_from_pairs(_pp, views["c2w"].shape[0],
                                        train_cfg.n_alt).to(device)
        print(f"  alt cameras: pairs_file '{_pp}' — first {train_cfg.n_alt} "
              f"ranked srcs per view")
    elif _vsel in ("arccos", "arccos_nn"):
        alt_nn = precompute_alt_cameras_arccos(views, train_cfg.n_alt).to(device)
        _mode = "minimal (flat sampling)" if _vsel == "arccos_nn" else "2-level uniform-cam"
        print(f"  alt cameras: {train_cfg.n_alt} arccos-nearest (angular distance) per view "
              f"[{_mode}]")
    else:
        alt_nn = precompute_alt_cameras(views, train_cfg.n_alt).to(device)
        print(f"  alt cameras: {train_cfg.n_alt} nearest NN per view")

    # arccos mode: precompute sorted fg/bg index tables for vectorised 2-level
    # sampling (camera ~ Uniform(V) → ray ~ Uniform within that camera's pool).
    if _vsel == "arccos":
        _vi_all = det["vi"]   # (total_rays,) CPU
        def _build_view_table(pool):
            """Returns (sorted_pool, view_offsets[V+1], view_counts[V]) on CPU."""
            vi_pool = _vi_all[pool]
            order   = vi_pool.argsort()
            sorted_pool = pool[order]
            counts  = torch.bincount(vi_pool, minlength=V)
            offsets = torch.zeros(V + 1, dtype=torch.long)
            offsets[1:] = counts.cumsum(0)
            return sorted_pool, offsets, counts
        _fg_sorted, _fg_offsets, _fg_counts = _build_view_table(fg_idx)
        _bg_sorted, _bg_offsets, _bg_counts = _build_view_table(bg_idx)
        print(f"  [arccos] one-ref-view-per-step: ref cam ~ Uniform(V), all rays from that cam")

    # one-time comparison log: selected source ids + triangulation angle per ref
    _ang = selected_pair_triangulation_angles(views, alt_nn, scene)
    _alt_cpu = alt_nn.cpu().tolist()
    print(f"  [view-select:{_vsel}] per-view selected src ids "
          f"(triangulation angle deg @object-centre):")
    for r in range(alt_nn.shape[0]):
        _pairs = "  ".join(f"{s}({a:.1f})" for s, a in zip(_alt_cpu[r], _ang[r]))
        print(f"    ref {r:3d}: {_pairs}")

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
    _opt_params = list(f.parameters())
    if radiance is not None:
        _opt_params += list(radiance.parameters())
    opt       = torch.optim.Adam(_opt_params, lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.steps,
                                                             eta_min=train_cfg.lr / 10)
    start_step = 0
    if _resume_ckpt is not None:
        start_step = int(_resume_ckpt.get("step", -1)) + 1
        if lr_warm_restart:
            # Warm restart: keep the trained weights but give a FRESH cosine that
            # anneals lr -> eta_min over the remaining [start_step, steps) segment,
            # and reset the optimiser moments. Without this, a plain resume restores
            # the floored scheduler (lr ~= eta_min) and the continuation barely moves.
            remaining = max(train_cfg.steps - start_step, 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=remaining, eta_min=train_cfg.lr / 10)
            print(f"  [resume] LR warm-restart: fresh cosine "
                  f"lr={train_cfg.lr:g} -> {train_cfg.lr / 10:g} over {remaining} steps "
                  f"(optimiser + scheduler reset)")
        else:
            if "opt" in _resume_ckpt:
                opt.load_state_dict(_resume_ckpt["opt"])
            if "scheduler" in _resume_ckpt:
                scheduler.load_state_dict(_resume_ckpt["scheduler"])
            else:
                for _ in range(start_step):
                    scheduler.step()
        print(f"  [resume] continuing from step {start_step} / {train_cfg.steps}")

    # --- in-loop bundle adjustment (interleaved φ-blocks) --------------------
    # When enabled, refine poses DURING training: the main loop is the θ phase
    # and every bundle.interval steps (after bundle.warmup_steps) we inject one
    # φ-block, then sync the refined poses into the training tensors below.
    inloop_ba = None
    if train_cfg.bundle.in_loop:
        if train_cfg.use_blender:
            print("  [inloop-BA] disabled — Blender scenes have GT poses (nothing to refine)")
        else:
            inloop_ba = InLoopBundleAdjuster(f, views, train_cfg, trace_cfg, device,
                                             images_dev=images)
            if _resume_ckpt is not None and "inloop_ba" in _resume_ckpt:
                inloop_ba.load_state_dict(_resume_ckpt["inloop_ba"])
                # re-apply the refined poses the run was training against
                c2w_all     = inloop_ba.current_c2w().to(c2w_all.dtype)
                origins_all = c2w_all[:, :3, 3]
                w2c_all     = torch.linalg.inv(c2w_all)
                z_cams_all  = c2w_all[:, :3, 2]
                views["c2w"] = c2w_all
                _regen_det_rays(det, c2w_all, K_all)

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
    prof = StepProfiler(_prof_dir, flush_every=1000, rays_per_step=train_cfg.batch,
                        enabled=train_cfg.profile)
    mem_snap = MemorySnapshot(_prof_dir, at_step=max(start_step + 50, 200))

    # ------------------------------------------------------------------ loop --
    for step in range(start_step, train_cfg.steps):
        prof.step_begin()
        # --- step-0 checkpoint: post-init weights BEFORE any optimization, so the
        #     pre-erosion ("before") state is recoverable for hole diagnostics ---
        if step == 0:
            step0_out = ckpt_dir / "checkpoint_step_000000.pt"
            torch.save({"f": f.state_dict(), "step": 0,
                        "architecture": f.architecture, "group_size": f.group_size,
                        "depth": f.depth, "activation": f.activation,
                        "input_encoding": f.input_encoding,
                        "multires": f.multires}, step0_out)
            print(f"  [ckpt] saved step-0 (post-init) checkpoint → {step0_out.name}", flush=True)
        parts = []
        if _vsel == "arccos":
            # One reference view per step (IDR/NeuS-style): pick a single camera
            # uniformly, then draw all this step's rays from THAT camera's pools.
            ref_cam = int(torch.randint(0, V, (1,)).item())
            def _draw_from_cam(n, sorted_pool, offsets, counts, fallback_pool):
                cnt = int(counts[ref_cam])
                if cnt == 0:                      # camera empty in this stratum → global fallback
                    return fallback_pool[torch.randint(0, len(fallback_pool), (n,))]
                local = torch.randint(0, cnt, (n,))
                return sorted_pool[offsets[ref_cam] + local]
            if n_fg > 0 and len(fg_idx) > 0:
                parts.append(_draw_from_cam(n_fg, _fg_sorted, _fg_offsets, _fg_counts, fg_idx))
            if n_bg > 0 and len(bg_idx) > 0:
                parts.append(_draw_from_cam(n_bg, _bg_sorted, _bg_offsets, _bg_counts, bg_idx))
        else:
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
        o = det["o"][idx].to(device);   u  = det["d"][idx].to(device)

        _need_eik = train_cfg.w_eikonal > 0 or train_cfg.w_mvs_sdf > 0 or train_cfg.mvsdf_schedule.enabled
        _trace_fn = trace_idr if trace_cfg.grad_mode == "idr" else trace_unrolled
        _need_diff_normal = (
            train_cfg.w_ncc_normal > 0
            or (train_cfg.w_ncc > 0 and not train_cfg.ncc_detach_normals)
        )
        with prof.timed("trace"):
            x_theta, t, hit, eik_pts, n_raw, sdf_min, hit_bg = _trace_fn(
                f_fwd, o, u, trace_cfg,
                collect_eik=_need_eik,
                diff_normal=_need_diff_normal,
                attach_normal_point=train_cfg.ncc_attach_normal_point)
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

        # Gipuma bilateral γ schedule (exp anneal): fixed large patch, γ↓ shrinks
        # the effective support window large→small (coarse→fine) over training.
        _bg_start = getattr(train_cfg, "ncc_bilateral_gamma", 0.0)
        _bg_end   = getattr(train_cfg, "ncc_bilateral_gamma_end", 0.0)
        if _bg_start > 0 and _bg_end > 0 and _bg_end != _bg_start:
            _t = step / max(train_cfg.steps - 1, 1)
            _eff_bgamma = _bg_start * (_bg_end / _bg_start) ** _t
        else:
            _eff_bgamma = _bg_start

        # --- losses ---
        with prof.timed("photo_ncc"):
            ph, ph_stats = photo_loss(
                f_fwd, x_theta, hit_for_photo, n,
                vi, alt_nn, origins_all,
                images, K_all, w2c_all, feature_maps, masks, fg_self,
                H, W, uv_self,
                train_cfg.n_alt, train_cfg.cos_thresh,
                _eff_w_photo, _eff_w_feat, train_cfg.w_ncc, train_cfg.ncc_patch, train_cfg.ncc_half_pix,
                train_cfg.sample_mode, _eff_sigma, _eff_radius,
                step, train_cfg.ncc_min, train_cfg.occ_mode,
                hit_bg=hit_bg if trace_cfg.bsphere_radius > 0 else None,
                ncc_sat_tau=getattr(train_cfg, "ncc_sat_tau", -1.0),
                w_ncc_normal=train_cfg.w_ncc_normal,
                ncc_detach_normals=train_cfg.ncc_detach_normals,
                ncc_topk=train_cfg.ncc_topk,
                ncc_abs_tau=getattr(train_cfg, "ncc_abs_tau", -1.0),
                ncc_color=train_cfg.ncc_color,
                ncc_grad_alpha=train_cfg.ncc_grad_alpha,
                ncc_normal_patch=train_cfg.ncc_normal_patch,
                ncc_normal_half_pix=train_cfg.ncc_normal_half_pix,
                ncc_patch_wsigma=_eff_wsigma,
                ncc_patch_bilateral_gamma=_eff_bgamma,
                ncc_world_patch=getattr(train_cfg, "ncc_world_patch", -1.0),
                trace_cfg=trace_cfg,
                prof=prof,
            )

        # --- soft-argmin photo-coherence (method.pdf §2) ---
        _sa_active = train_cfg.w_soft_argmin > 0 and step >= train_cfg.sa_start_step
        if _sa_active:
            # Temperature annealed from tau_start (soft global pull) to tau_end (sharp local pick)
            _prog_sa = step / max(train_cfg.steps - 1, 1)
            _sa_tau  = train_cfg.sa_tau_start * (
                train_cfg.sa_tau_end / max(train_cfg.sa_tau_start, 1e-9)
            ) ** _prog_sa
            _sa_t_far = train_cfg.sa_t_far if train_cfg.sa_t_far > 0 else trace_cfg.t_far
            sa_pull, sa_stats = soft_argmin_photo_loss(
                f,
                x_theta, hit, o, u, vi,
                alt_nn, origins_all,
                images, K_all, w2c_all,
                H, W,
                train_cfg.sa_n_candidates,
                _sa_tau,
                train_cfg.sa_t_near,
                _sa_t_far,
                train_cfg.sa_use_bg,
                train_cfg.sa_bg_color,
                trace_cfg=trace_cfg,
                prof=prof,
            )
        else:
            sa_pull = torch.zeros(1, device=device).squeeze()
            sa_stats = {}

        if train_cfg.sil_s_interval > 0:
            n_doublings = min(step // train_cfg.sil_s_interval, train_cfg.sil_s_max_mults)
            alpha = train_cfg.sil_s * (2.0 ** n_doublings)
        else:
            alpha = train_cfg.sil_s

        with prof.timed("idr_mask"):
            idr_mask, idr_stats = (idr_mask_loss(f, o, u, hit, fg_self, alpha,
                                                 train_cfg.idr_n_samples,
                                                 train_cfg.sil_t_near, train_cfg.sil_t_far)
                                   if train_cfg.w_idr_mask > 0
                                   else (torch.zeros(1, device=device).squeeze(), {}))
        sil  = (mask_loss_min_sdf(sdf_min, fg_self, alpha,
                                  fg_offset=train_cfg.sil_fg_offset,
                                  bg_offset=train_cfg.sil_bg_offset,
                                  focal_gamma=train_cfg.sil_focal_gamma,
                                  balance_classes=train_cfg.sil_balance,
                                  normalize_by_alpha=train_cfg.sil_norm_alpha)
                if train_cfg.w_sil > 0 else torch.zeros(1, device=device).squeeze())
        mask_fg, mask_bg = (dvr_mask_loss(f, o, u, fg_self, trace_cfg.t_far,
                                          train_cfg.mask_fg_margin, train_cfg.mask_bg_margin,
                                          train_cfg.n_mask_fg, train_cfg.n_mask_bg)
                            if (train_cfg.w_mask_fg > 0 or train_cfg.w_mask_bg > 0)
                            else (torch.zeros(1, device=device).squeeze(),
                                  torch.zeros(1, device=device).squeeze()))
        with prof.timed("eikonal"):
            eik  = (eikonal_loss(f, eik_pts, train_cfg.n_eik_vol, device)
                    if train_cfg.w_eikonal > 0 else torch.zeros(1, device=device).squeeze())
        cfr  = (cam_free_loss(f, o)
                if train_cfg.w_cam_free > 0 else torch.zeros(1, device=device).squeeze())
        if train_cfg.w_sfm > 0:
            # Geo-Neus exact SDF loss: L1, one view's visible COLMAP points per
            # step (cycling iter % V, as in exp_runner). Falls back to global L1
            # when no per-view visibility is available (e.g. blender GT points).
            _view_sel = (step % sfm_vis.shape[0]) if sfm_vis is not None else None
            sfm_surface = geo_neus_sdf_loss(f, sfm_pts, vis=sfm_vis,
                                            view_sel=_view_sel, batch=train_cfg.batch)
            sfm_clear = (free_space_loss(f, sfm_origins, sfm_targets, n_sfm_pairs,
                                         train_cfg.batch, train_cfg.n_free)
                         if n_sfm_pairs > 0 else torch.zeros(1, device=device).squeeze())
            sfm_behind = (sfm_behind_loss(f, sfm_origins, sfm_targets, n_sfm_pairs,
                                          train_cfg.batch, train_cfg.sfm_behind_eps)
                          if n_sfm_pairs > 0 and train_cfg.sfm_behind_eps > 0
                          else torch.zeros(1, device=device).squeeze())
            sfm = sfm_surface + sfm_clear + sfm_behind
        else:
            sfm = torch.zeros(1, device=device).squeeze()
        # Pure Geo-Neus SDF loss (surface term ONLY), independent of the w_sfm
        # bundle above. Exactly exp_runner's L1, per-view. No free-space/behind.
        if train_cfg.w_geo_sdf > 0:
            _gv = (step % sfm_vis.shape[0]) if sfm_vis is not None else None
            sfm_geo = geo_neus_sdf_loss(f, sfm_pts, vis=sfm_vis,
                                        view_sel=_gv, batch=train_cfg.batch)
        else:
            sfm_geo = torch.zeros(1, device=device).squeeze()
        fs   = (free_space_loss(f, sfm_origins, sfm_targets, n_sfm_pairs,
                                train_cfg.batch, train_cfg.n_free)
                if train_cfg.w_free > 0 and n_sfm_pairs > 0
                else torch.zeros(1, device=device).squeeze())
        surf = (surface_loss(f, o, u, vi, c2w_all, mvs_depth_flat, mvs_valid_flat, idx_dev)
                if train_cfg.w_surf > 0 else torch.zeros(1, device=device).squeeze())
        mvs  = (mvs_depth_loss(x_theta, o, u, vi, hit, c2w_all,
                               mvs_depth_flat, mvs_valid_flat, idx_dev, step)
                if train_cfg.w_mvs > 0 else torch.zeros(1, device=device).squeeze())
        with prof.timed("mvs_sdf"):
            if _eff_w_msdf > 0:
                mvs_sdf_pts = mvs_sdf_lo + torch.rand(
                    train_cfg.n_mvs_sdf, 3, device=device,
                ) * (mvs_sdf_hi - mvs_sdf_lo)
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

        # learned view-dependent colour: L1 between predicted RGB at the surface
        # hit and the observed pixel colour `gt` (real hits only). Gradient flows
        # through x_theta → SDF (à la IDR).
        if radiance is not None and hit.any():
            rgb_pred = radiance(x_theta[hit], n[hit], u[hit])
            rgb = (rgb_pred - gt[hit]).abs().mean()
        elif radiance is not None:
            # no hits this step — keep a valid grad_fn so backward doesn't error
            rgb = radiance(x_theta[:1], n[:1], u[:1]).sum() * 0.0
        else:
            rgb = torch.zeros(1, device=device).squeeze()

        loss = (ph + train_cfg.w_rgb * rgb
                + train_cfg.w_idr_mask * idr_mask + train_cfg.w_sil * sil + train_cfg.w_eikonal * eik
                + train_cfg.w_mask_fg * mask_fg + train_cfg.w_mask_bg * mask_bg
                + train_cfg.w_cam_free * cfr + train_cfg.w_sfm * sfm + train_cfg.w_geo_sdf * sfm_geo + train_cfg.w_free * fs
                + train_cfg.w_surf * surf + train_cfg.w_mvs * mvs + _eff_w_msdf * msdf
                + train_cfg.w_behind_hit * beh + train_cfg.w_ray_free * rf
                + train_cfg.w_normal * nrm
                + train_cfg.w_soft_argmin * sa_pull)
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

        # --- in-loop BA: inject a φ-block, then sync refined poses --------------
        # The φ-block refines the cameras against the *current* geometry (θ frozen
        # inside it); we then overwrite the pose tensors the θ-loop reads so the
        # next steps train against the corrected poses. alt_nn (neighbour topology)
        # is left unchanged — BA pose deltas are sub-degree, so the nearest-view
        # sets are stable; the reprojection itself uses the refreshed w2c_all.
        if inloop_ba is not None:
            _ba = inloop_ba.step(f, step)
            if _ba is not None:
                c2w_all     = inloop_ba.current_c2w().to(c2w_all.dtype)
                origins_all = c2w_all[:, :3, 3]
                w2c_all     = torch.linalg.inv(c2w_all)
                z_cams_all  = c2w_all[:, :3, 2]
                views["c2w"] = c2w_all
                # rebuild the precomputed PRIMARY rays so source rays move with φ
                _regen_det_rays(det, c2w_all, K_all)
                print(f"  [inloop-BA@{step}] block {_ba['block']} ({train_cfg.bundle.block_phi} φ-steps)  "
                      f"E={_ba['E']:.4f} ncc={_ba['ncc']:.3f} kept={_ba['kept']:.2f}  "
                      f"|Δφ|rot={_ba['rot_deg']:.4f}° trans={_ba['trans']:.3e}  "
                      f"lr_scale={_ba['lr_scale']:.2f}", flush=True)
                if use_wandb:
                    wandb.log({"inloop_ba/E": _ba["E"], "inloop_ba/ncc": _ba["ncc"],
                               "inloop_ba/rot_deg": _ba["rot_deg"],
                               "inloop_ba/trans": _ba["trans"],
                               "inloop_ba/lr_scale": _ba["lr_scale"]}, step=step)

        # --- periodic latest checkpoint (atomic, includes opt/scheduler for preemption) ---
        if step > 0 and step % LATEST_FREQ == 0:
            _ckpt_payload = {"f": f.state_dict(), "step": step,
                             "architecture": f.architecture, "group_size": f.group_size,
                             "depth": f.depth, "activation": f.activation,
                             "input_encoding": f.input_encoding,
                             "multires": f.multires,
                             "opt": opt.state_dict(),
                             "scheduler": scheduler.state_dict()}
            if radiance is not None:
                _ckpt_payload["radiance"] = radiance.state_dict()
            if inloop_ba is not None:
                _ckpt_payload["inloop_ba"] = inloop_ba.state_dict()
            tmp = LATEST_OUT.with_suffix(".pt.tmp")
            _disk_full_retry(torch.save, _ckpt_payload, tmp, _desc="latest checkpoint")
            tmp.replace(LATEST_OUT)

        # --- step checkpoint every STEP_CKPT_FREQ steps ---
        if step > 0 and step % STEP_CKPT_FREQ == 0:
            step_out = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
            _step_payload = {"f": f.state_dict(), "step": step,
                        "architecture": f.architecture, "group_size": f.group_size,
                        "depth": f.depth, "activation": f.activation,
                        "input_encoding": f.input_encoding,
                        "multires": f.multires,
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict()}
            if radiance is not None:
                _step_payload["radiance"] = radiance.state_dict()
            try:
                torch.save(_step_payload, step_out)
                print(f"  [ckpt] saved step checkpoint → {step_out.name}", flush=True)
            except OSError as e:
                if e.errno not in _DISK_FULL_ERRNOS:
                    raise
                # redundant backup — latest checkpoint already holds progress; skip
                step_out.unlink(missing_ok=True)
                _safe_print(f"  [ckpt] disk full — skipped step checkpoint {step_out.name}")
            # Disabled to keep training fast: the 10k-step MC normal-map dump +
            # per-view diag PNGs are heavy diagnostics. Re-enable if needed.
            # _dump_mc_normal_maps(f, views, NORMAL_DUMP_VIEWS, step, run_dir,
            #                      device, mc_res=NORMAL_DUMP_MC_RES,
            #                      bound=eval_cfg.bound(train_cfg.use_blender),
            #                      trace_cfg=trace_cfg, train_cfg=train_cfg,
            #                      alt_nn=alt_nn,
            #                      coverage_res=NORMAL_DUMP_COVERAGE_RES)

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
                photo_str += (f"  ncc[zncc_all={pm['ncc_zncc']:.3f} "
                              f"zncc_used={pm.get('ncc_zncc_used', 0.0):.3f} "
                              f"kept={_k}/{_t}/{_v} ({_kf:.2f}|{_tf:.2f})]")
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
            sa_str = ""
            if _sa_active and sa_stats:
                sa_str = (f"  sa_pull {sa_pull.item():.4f}[w={train_cfg.w_soft_argmin}]"
                          f"  sa_τ={sa_stats['sa_tau']:.3f}"
                          f"  sa_Δt={sa_stats['sa_t_err']:.3f}"
                          f"  sa_sig={sa_stats['sa_signal_frac']:.2f}")
            rgb_str = ""
            if radiance is not None:
                rgb_str = f"  rgb {rgb.item():.4f}[w={train_cfg.w_rgb} -> {train_cfg.w_rgb * rgb.item():.4f}]"
            _hit_bg_str = (f"+bg{hit_bg.sum()}(ph:{ph_stats['n_mask_bg']})"
                           if trace_cfg.bsphere_radius > 0 else "")
            if train_cfg.w_idr_mask > 0 and idr_stats:
                print(f"  idr_mask {idr_mask.item():.4f}[w={train_cfg.w_idr_mask} α={alpha:.0f}]  "
                      f"pout={idr_stats['idr_n_pout']}  "
                      f"fn={idr_stats['idr_n_fn']}(S={idr_stats['idr_S_fn']:.3f} sdf={idr_stats['idr_sdf_fn']:.4f})  "
                      f"fp={idr_stats['idr_n_fp']}(S={idr_stats['idr_S_fp']:.3f} sdf={idr_stats['idr_sdf_fp']:.4f})")
            print(f"step {step:5d}  cams {vi.unique().numel():2d}  "
                  f"loss {loss.item():.4f}  {photo_str}  sil {sil.item():.4f}  "
                  f"mask_fg {mask_fg.item():.4f}[w={train_cfg.w_mask_fg}]  "
                  f"mask_bg {mask_bg.item():.4f}[w={train_cfg.w_mask_bg}]  "
                  f"hit {hit.sum()}{_hit_bg_str}/{train_cfg.batch}  x_r {xh_str}  "
                  f"mask {pm['n_mask']}/{pm['n_in_frame']}/{pm['n_not_occl']}/"
                  f"{pm['n_cos_ok']}/{pm['n_total']}  "
                  f"f(o)<0 {frac_fo_neg:.2f}  t_far {frac_t_far:.2f}  "
                  f"sfm {sfm.item():.4f}  geo_sdf {sfm_geo.item():.4f}  free {fs.item():.4f}  "
                  f"surf {surf.item():.4f}  mvs {mvs.item():.4f}  "
                  f"msdf {msdf.item():.4f}[w={_eff_w_msdf}]  beh {beh.item():.4f}  "
                  f"rf {rf.item():.4f}  eik {eik.item():.4f}[w={train_cfg.w_eikonal}]  "
                  f"nrm {nrm.item():.4f}  ∇head {grad_norm:.6f}{neus_trace_str}{sa_str}{rgb_str}")

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
                try:
                    torch.save({"f": f.state_dict(), "step": step, "score": score,
                                "architecture": f.architecture, "group_size": f.group_size,
                                "depth": f.depth, "activation": f.activation,
                                "input_encoding": f.input_encoding,
                                "multires": f.multires}, BEST_OUT)
                    print(f"  [best_geo@{step}] score={score:.4f} → {BEST_OUT.name}")
                    torch.cuda.empty_cache()
                    _render_poses(f, views, step, run_dir, device, trace_cfg=trace_cfg, crop_fg=is_mvm, radiance=radiance)
                    render_src = render_dir / f"render_{step:05d}.png"
                    if render_src.exists():
                        shutil.copy(render_src, render_dir / "render_best_geo.png")
                except OSError as e:
                    if e.errno not in _DISK_FULL_ERRNOS:
                        raise
                    _safe_print(f"  [best_geo@{step}] disk full — skipped save/render")

            # best_photo block disabled: with a single active loss (e.g. NCC-only,
            # w_photo=0), best_photo and best_loss track the same quantity, so
            # firing both was double-saving the ckpt and double-rendering 4
            # sphere-traced views every 50 steps. Keep best_loss only.

            _cur_loss = loss.item()
            loss_ckpt_ok = train_cfg.use_blender or hr_log > 0.01
            if _cur_loss < best_loss_score and loss_ckpt_ok:
                best_loss_score = _cur_loss
                try:
                    torch.save({"f": f.state_dict(), "step": step, "loss": _cur_loss,
                                "architecture": f.architecture, "group_size": f.group_size,
                                "depth": f.depth, "activation": f.activation,
                                "input_encoding": f.input_encoding,
                                "multires": f.multires}, BEST_LOSS_OUT)
                    print(f"  [best_loss@{step}] loss={_cur_loss:.4f} → {BEST_LOSS_OUT.name}")
                    torch.cuda.empty_cache()
                    _render_poses(f, views, step, run_dir, device, trace_cfg=trace_cfg, crop_fg=is_mvm, radiance=radiance)
                    render_src = render_dir / f"render_{step:05d}.png"
                    if render_src.exists():
                        shutil.copy(render_src, render_dir / "render_best_loss.png")
                except OSError as e:
                    if e.errno not in _DISK_FULL_ERRNOS:
                        raise
                    _safe_print(f"  [best_loss@{step}] disk full — skipped save/render")
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
            tnt_official = None
            bmvs_official = None
            blender_official = None
            if train_cfg.use_blender:
                if (gt_pts is not None and eval_cfg.blender_chamfer_freq > 0
                        and step % eval_cfg.blender_chamfer_freq == 0):
                    cd = _mc_chamfer(f, gt_pts, device, bound=eval_cfg.bound_blender, mc_level=eval_cfg.mc_level)
                    if cd is not None:
                        print(f"  [chamfer@{step:5d}] sym={cd['chamfer']:.6f}  "
                              f"precision={cd['precision']:.6f}  completeness={cd['completeness']:.6f}")
                    else:
                        print(f"  [chamfer@{step:5d}] n/a (surface not in bounds)")
                if (eval_cfg.blender_official_freq > 0
                        and step % eval_cfg.blender_official_freq == 0):
                    off_dir = run_dir / "blender_official" / f"step_{step:06d}"
                    try:
                        blender_official = _run_blender_official_eval(
                            f, scene, off_dir, device,
                            bound=eval_cfg.blender_official_bound,
                            res=eval_cfg.blender_official_res,
                            n_samples=eval_cfg.blender_official_n_samples,
                            mask_crop=eval_cfg.blender_official_mask_crop,
                            mask_dilate_px=eval_cfg.blender_official_mask_dilate_px,
                            mask_crop_min_ratio=eval_cfg.blender_official_mask_crop_min_ratio,
                            mask_crop_min_views=eval_cfg.blender_official_mask_crop_min_views,
                            mc_level=eval_cfg.mc_level,
                        )
                    except OSError as e:
                        if e.errno not in _DISK_FULL_ERRNOS:
                            raise
                        blender_official = None
                        _safe_print(f"  [blender_official@{step:5d}] disk full — skipped eval")
                    if blender_official is not None:
                        print(f"  [blender_official@{step:5d}] "
                              f"chamfer={blender_official['chamfer']:.6f}  "
                              f"acc={blender_official['accuracy']:.6f}  "
                              f"comp={blender_official['completeness']:.6f}  "
                              f"x100={blender_official['chamfer_x100']:.4f}  out={off_dir}")
                    else:
                        print(f"  [blender_official@{step:5d}] n/a (surface not in bounds / no GT)")
            elif not train_cfg.use_blender:
                sfm_surf_due = eval_cfg.dtu_chamfer_freq > 0 and step % eval_cfg.dtu_chamfer_freq == 0
                official_due = (
                    eval_cfg.dtu_official_freq > 0 and step >= 0
                    and step % eval_cfg.dtu_official_freq == 0
                    and dtu_scale_mat is not None and dtu_scan_id is not None
                    and eval_cfg.dtu_eval_dir is not None
                )
                tnt_official_due = (
                    eval_cfg.tnt_official_freq > 0 and step >= 0
                    and step % eval_cfg.tnt_official_freq == 0
                    and eval_cfg.tnt_eval_dir is not None
                    and tnt_scene_name is not None
                )
                bmvs_official_due = (
                    eval_cfg.bmvs_official_freq > 0 and step >= 0
                    and step % eval_cfg.bmvs_official_freq == 0
                    and bmvs_scale_mat is not None and bmvs_gt_mesh_path is not None
                )
                # colour render on the DTU-official cadence (step 0 then every
                # dtu_official_freq): predicted RGB from the colour MLP alongside
                # Phong/GT/hit. Only when the colour MLP is active.
                if (radiance is not None and eval_cfg.dtu_official_freq > 0
                        and step % eval_cfg.dtu_official_freq == 0):
                    torch.cuda.empty_cache()
                    _render_poses(f, views, step, run_dir, device,
                                  trace_cfg=trace_cfg, crop_fg=is_mvm, radiance=radiance)
                if sfm_surf_due and (train_cfg.w_sfm > 0 or train_cfg.w_geo_sdf > 0) and sfm_pts.numel() > 3:
                    sfm_surf = _mc_sfm_surface_distance(
                        f, sfm_pts, device,
                        bound=eval_cfg.bound_dtu,
                        res=eval_cfg.dtu_chamfer_res,
                        mc_level=eval_cfg.mc_level,
                    )
                    if sfm_surf is not None:
                        print(f"  [sfm_surf@{step:5d}] mean={sfm_surf['mean']:.4f}  "
                              f"p50={sfm_surf['p50']:.4f}  p90={sfm_surf['p90']:.4f}  "
                              f"p99={sfm_surf['p99']:.4f}")
                    else:
                        print(f"  [sfm_surf@{step:5d}] n/a (surface not in bounds)")
                if official_due:
                    off_dir = run_dir / "dtu_official" / f"step_{step:06d}"
                    mesh_ply = _extract_world_mesh_for_dtu(
                        f, dtu_scale_mat, device, off_dir / "pred_world_mesh.ply",
                        bound=eval_cfg.dtu_official_bound,
                        res=eval_cfg.dtu_official_res,
                        mc_level=eval_cfg.mc_level,
                    )
                    if mesh_ply is None:
                        print(f"  [dtu_official@{step:5d}] n/a (surface not in bounds)")
                    else:
                        dtu_official = _run_dtu_official_eval(
                            mesh_ply, dtu_scan_id, eval_cfg.dtu_eval_dir, off_dir,
                            scene=scene,
                            mask_crop=eval_cfg.dtu_official_mask_crop,
                            mask_dilate_px=eval_cfg.dtu_official_mask_dilate_px,
                            mask_crop_min_ratio=eval_cfg.dtu_official_mask_crop_min_ratio,
                            mask_crop_min_views=eval_cfg.dtu_official_mask_crop_min_views,
                        )
                        if dtu_official is not None:
                            print(f"  [dtu_official@{step:5d}] chamfer={dtu_official['chamfer']:.4f}mm  "
                                  f"acc={dtu_official['accuracy']:.4f}mm  "
                                  f"comp={dtu_official['completeness']:.4f}mm  out={off_dir}")
                # MVMannequin parallel path: same cadence, different eval recipe.
                mvm_official_due = (
                    eval_cfg.dtu_official_freq > 0 and step >= 0
                    and step % eval_cfg.dtu_official_freq == 0 and is_mvm
                )
                if mvm_official_due:
                    off_dir = run_dir / "mvmannequin_official" / f"step_{step:06d}"
                    mvm_official = _run_mvmannequin_official_eval(
                        f, scene, off_dir,
                        bound=eval_cfg.dtu_official_bound,
                        res=eval_cfg.dtu_official_res,
                        device=device,
                        mc_level=eval_cfg.mc_level,
                    )
                    if mvm_official is not None:
                        print(f"  [mvm_official@{step:5d}] chamfer={mvm_official['chamfer']:.4f}mm  "
                              f"acc={mvm_official['accuracy']:.4f}mm  "
                              f"comp={mvm_official['completeness']:.4f}mm  out={off_dir}")
                if tnt_official_due:
                    off_dir = run_dir / "tnt_official" / f"step_{step:06d}"
                    points_ply = _extract_tnt_eval_points(
                        f, scene, off_dir, device,
                        bound=eval_cfg.tnt_official_bound,
                        res=eval_cfg.tnt_official_res,
                        n_samples=eval_cfg.tnt_official_n_samples,
                        mc_level=eval_cfg.mc_level,
                    )
                    if points_ply is not None:
                        tnt_official = _run_tnt_official_eval(
                            points_ply, scene, eval_cfg.tnt_eval_dir,
                            tnt_scene_name, off_dir,
                            frame=eval_cfg.tnt_official_frame,
                        )
                        if tnt_official is not None:
                            print(f"  [tnt_official@{step:5d}] "
                                  f"F={tnt_official['fscore']:.4f}  "
                                  f"P={tnt_official['precision']:.4f}  "
                                  f"R={tnt_official['recall']:.4f}  "
                                  f"tau={tnt_official['tau']:.4f}  out={off_dir}")
                if bmvs_official_due:
                    off_dir = run_dir / "bmvs_official" / f"step_{step:06d}"
                    bmvs_official = _run_bmvs_official_eval(
                        f, bmvs_gt_mesh_path, bmvs_scale_mat, off_dir, device,
                        bound=eval_cfg.bmvs_official_bound,
                        res=eval_cfg.bmvs_official_res,
                        n_samples=eval_cfg.bmvs_official_n_samples,
                        protocol=eval_cfg.bmvs_official_protocol,
                        ground_axis=eval_cfg.bmvs_official_ground_axis,
                        ground_value=eval_cfg.bmvs_official_ground_value,
                        scene_dir=scene,
                        mc_level=eval_cfg.mc_level,
                    )
                    if bmvs_official is not None:
                        print(f"  [bmvs_official@{step:5d}] chamfer={bmvs_official['chamfer']:.6f}  "
                              f"acc={bmvs_official['accuracy']:.6f}  "
                              f"comp={bmvs_official['completeness']:.6f}  out={off_dir}")
                    else:
                        print(f"  [bmvs_official@{step:5d}] n/a (surface not in bounds)")

            if use_wandb:
                import wandb
                log = {"loss": loss.item(), "photo": ph.item(), "sil": sil.item(),
                       "sfm": sfm.item(), "geo_sdf": sfm_geo.item(),
                       "eik": eik.item(),
                       "eik_weighted": train_cfg.w_eikonal * eik.item(),
                       "photo_l1": ph_stats["l1"], "photo_ncc": ph_stats["ncc"],
                       "photo_ncc_weighted": ph_stats.get("ncc_weighted", 0.0),
                       "photo_ncc_normal": ph_stats.get("ncc_normal", 0.0),
                       "photo_ncc_normal_weighted": ph_stats.get("ncc_normal_weighted", 0.0),
                       "photo_ncc_zncc": ph_stats.get("ncc_zncc", 0.0),
                       "photo_ncc_zncc_used": ph_stats.get("ncc_zncc_used", 0.0),
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
                if radiance is not None:
                    log["rgb_l1"] = rgb.item()
                    log["rgb_weighted"] = train_cfg.w_rgb * rgb.item()
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
                if tnt_official is not None:
                    log["tnt_official_fscore"] = tnt_official["fscore"]
                    log["tnt_official_precision"] = tnt_official["precision"]
                    log["tnt_official_recall"] = tnt_official["recall"]
                    log["tnt_official_tau"] = tnt_official["tau"]
                if bmvs_official is not None:
                    log["bmvs_official_chamfer"] = bmvs_official["chamfer"]
                    log["bmvs_official_acc"] = bmvs_official["accuracy"]
                    log["bmvs_official_comp"] = bmvs_official["completeness"]
                    log["bmvs_official_chamfer_raw"] = bmvs_official["chamfer_raw_units"]
                if blender_official is not None:
                    log["blender_official_chamfer"] = blender_official["chamfer"]
                    log["blender_official_acc"] = blender_official["accuracy"]
                    log["blender_official_comp"] = blender_official["completeness"]
                    log["blender_official_chamfer_x100"] = blender_official["chamfer_x100"]
                if (render_dir / "render_{:05d}.png".format(step)).exists():
                    log["render"] = wandb.Image(str(render_dir / "render_{:05d}.png".format(step)))
                wandb.log(log, step=step)

    # --- save final ---
    final_payload = {"f": f.state_dict(), "architecture": f.architecture,
                     "group_size": f.group_size, "depth": f.depth,
                     "activation": f.activation,
                     "input_encoding": f.input_encoding,
                     "multires": f.multires, "step": step}
    _disk_full_retry(torch.save, final_payload, OUT, _desc="final checkpoint")
    FINAL_OUT = ckpt_dir / "checkpoint_final.pt"
    _disk_full_retry(torch.save, final_payload, FINAL_OUT, _desc="checkpoint_final")
    print(f"saved → {OUT}")
    print(f"final → {FINAL_OUT.name}")
    try:
        _render_poses(f, views, step, run_dir, device, trace_cfg=trace_cfg, crop_fg=is_mvm, radiance=radiance)
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
    ap.add_argument("--use-masks", action=argparse.BooleanOptionalAction, default=None,
                    help="use dataset masks for fg/bg sampling and photo-loss gates "
                         f"(default {_tc.use_masks}; --no-use-masks ignores them, "
                         "and overrides the value in --config)")
    ap.add_argument("--bake-background", action=argparse.BooleanOptionalAction, default=None,
                    help="zero the RGB background via the dataset mask at load "
                         f"(default {_tc.bake_background}; --no-bake-background keeps the "
                         "real photographed background → a TRULY mask-free run, no "
                         "silhouette leak into the photo loss; overrides --config)")
    ap.add_argument("--grad-weighted-sampling", action="store_true",
                    default=_tc.grad_weighted_sampling,
                    help="sample fg rays ∝ image-gradient magnitude (fine-detail focus)")
    ap.add_argument("--grad-sampling-alpha", type=float,
                    default=_tc.grad_sampling_alpha,
                    help="grad/uniform mix for --grad-weighted-sampling (0=uniform, 1=pure grad)")
    ap.add_argument("--fg-fraction", type=float, default=_tc.fg_fraction,
                    help="foreground-ray share when sampling both fg/bg strata")
    ap.add_argument("--force-fg-bg-split", action="store_true",
                    help="sample both fg/bg strata even without a bg-sensitive loss")
    ap.add_argument("--init-hit-sampling", dest="init_hit_sampling",
                    action="store_true", default=_tc.init_hit_sampling,
                    help="mask-free fg: trace det rays vs the init SDF once and set fg=hit "
                         "(object-focused sampling without a segmentation mask)")
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
                    choices=["groupsort", "nact", "softplus", "centered_softplus", "softplus_cpl", "softmax_cpl", "softplus_cpl_maxmin"],
                    help="activation: groupsort, nact, softplus, centered_softplus, softplus_cpl, softmax_cpl, or softplus_cpl_maxmin")
    ap.add_argument("--input-encoding", type=str, default=_mc.input_encoding,
                    choices=["identity", "pe"], help="input encoding before the 1-Lipschitz backbone")
    ap.add_argument("--architecture", type=str, default=_mc.architecture, choices=["cpl", "mlp", "neus"],
                    help="cpl: 1-Lipschitz CPL (default); neus: Softplus+skip MLP; mlp: plain ReLU MLP")
    ap.add_argument("--multires", type=int, default=_mc.multires,
                    help="number of positional encoding frequencies (input_encoding=pe)")
    _trc = TraceConfig()
    ap.add_argument("--trace-iters", type=int, default=_trc.iters,
                    help="max sphere-tracing iterations (increase when using PE)")
    ap.add_argument("--occ-iters", type=int, default=_trc.occ_iters,
                    help="max iters for the photo-loss occlusion trace (-1 = same as --trace-iters). "
                         "The occ trace is a thresholded visibility boolean, so it tolerates far "
                         "fewer iters than the primary trace.")
    ap.add_argument("--occ-newton-steps", type=int, default=_trc.occ_newton_steps,
                    help="Newton steps for the occlusion trace (-1 = same as primary; 0 = none). "
                         "Each Newton step is ~3x a regular iter (fwd+bwd) and is pointless for a boolean.")
    ap.add_argument("--occ-eps", type=float, default=_trc.occ_eps,
                    help="hit threshold for the occlusion trace (<0 = same as --eps)")
    ap.add_argument("--occ-depth-slack", type=float, default=_trc.occ_depth_slack,
                    help="depth tolerance for the occ visibility test, normalized units "
                         "(default 1e-2 ~= 10x eps). Must exceed eps (self-occlusion "
                         "residual); floored by grazing-angle error, capped by the "
                         "thinnest occluder you need to catch. Lower it for thin/"
                         "self-occluding scenes (e.g. DTU scan37).")
    ap.add_argument("--eps", type=float, default=_trc.eps,
                    help="sphere-trace hit threshold |f(x)|<eps, in normalized units. "
                         "DTU 1 unit~0.25m so 1e-3~0.25mm; MVMannequin 1 unit~1m so "
                         "1e-3~1mm (use ~2.5e-4 to match DTU's metric precision)")
    ap.add_argument("--bsphere-radius", type=float, default=_trc.bsphere_radius,
                    help=">0: use per-ray bounding-sphere exit as t_far; rays that reach "
                         "the sphere become hit_bg and participate in photo loss with "
                         "photometrically inconsistent colors → gradient fills holes. "
                         "Set to ~1.5× the object bounding-sphere radius.")
    ap.add_argument("--bsphere-start-radius", type=float, default=_trc.bsphere_start_radius,
                    help=">0: START each trace at the ray's near intersection with this "
                         "origin-centred bounding sphere (skip the empty camera→object gap) "
                         "instead of at the camera. Set to the object englobing radius "
                         "(~1.05-1.2 for origin-normalised DTU). 0 = trace from camera.")
    ap.add_argument("--t-far", type=float, default=_trc.t_far,
                    help="global ray cut-off distance (used when --bsphere-radius=0). "
                         "Must exceed max camera distance + object radius.")
    _ic = InitConfig()
    ap.add_argument("--init",       type=str,   default=_ic.init,
                    choices=["sphere", "hull", "colmap", "points"],
                    help="warm-start: sphere, silhouette visual hull, voxelisation "
                         "of a COLMAP dense mesh (needs --init-mesh), or 'points' = "
                         "Poisson surface reconstructed straight from the sparse SfM "
                         "cloud (sparse_sfm_points.txt; no masks, no dense MVS)")
    ap.add_argument("--init-mesh",  type=str,   default=_ic.init_mesh,
                    help="path to a triangle mesh in NSVF-COLMAP frame "
                         "(e.g. outputs/colmap_barn/poisson.ply); only used "
                         "when --init colmap")
    ap.add_argument("--init-sdf-grid", type=str, default=_ic.init_sdf_grid,
                    help="hull init: fit to this precomputed SDF grid instead of "
                         "silhouette carving; inside is sdf < 0")
    ap.add_argument("--good-views", type=Path, default=None,
                    help="file of view indices to keep (drops noisy-mask views "
                         "from the multi-view training pool); see analysis/"
                         "cat_hull_coverage_check.py for generating one")
    ap.add_argument("--init-steps", type=int,   default=_ic.steps,
                    help="gradient steps for the hull/sphere warm-start")
    ap.add_argument("--radius",     type=float, default=_ic.radius,
                    help="sphere-init radius (None = auto from COLMAP p60)")
    ap.add_argument("--hull-res",   type=int,   default=_ic.hull_res,
                    help="voxel resolution for hull carving")
    ap.add_argument("--hull-sfm-roi", action="store_true", default=_ic.hull_sfm_roi,
                    help="crop hull-init occupancy to a padded sparse-SFM AABB; "
                         "useful when close-up views leave outer voxels unconstrained")
    ap.add_argument("--hull-min-views", type=int, default=_ic.hull_min_views,
                    help="require hull voxels to project inside at least this many views")
    ap.add_argument("--hull-border-aware", action="store_true", default=_ic.hull_border_aware,
                    help="carve off-frame voxels through image edges the silhouette does not touch; "
                         "removes visual-hull bloat when the object spills past the frame (e.g. scan24)")
    ap.add_argument("--w-sfm-free", type=float, default=_ic.w_sfm_free,
                    help="hull init: enforce f>0 along camera→COLMAP-point sight-lines "
                         "(carves the hull back toward the real surface, incl. concavities)")
    ap.add_argument("--sfm-free-eps", type=float, default=_ic.sfm_free_eps,
                    help="hull init: stop the SFM free-space ray this far (world units) before the point")
    ap.add_argument("--w-depth-surface", type=float, default=_ic.w_depth_surface,
                    help="blender hull-init only: weight on GT depth surface samples")
    ap.add_argument("--points-poisson-depth", type=int, default=_ic.points_poisson_depth,
                    help="points init: screened-Poisson octree depth (8=coarse, 9=fine)")
    ap.add_argument("--points-trim-quantile", type=float, default=_ic.points_trim_quantile,
                    help="points init: drop Poisson vertices below this density quantile "
                         "(removes balloon extrapolation in unobserved regions)")
    ap.add_argument("--points-normal-knn", type=int, default=_ic.points_normal_knn,
                    help="points init: kNN for PCA normal estimation + orientation MST")
    ap.add_argument("--w-photo",    type=float, default=_tc.w_photo)
    ap.add_argument("--w-rgb",      type=float, default=_tc.w_rgb,
                    help="IDR-style learned view-dependent colour MLP weight (0=off)")
    ap.add_argument("--rgb-hidden", type=int,   default=_tc.rgb_hidden)
    ap.add_argument("--rgb-depth",  type=int,   default=_tc.rgb_depth)
    ap.add_argument("--rgb-no-view-dep", action="store_true",
                    help="colour MLP: diffuse albedo only (drop view direction)")
    ap.add_argument("--rgb-pe", action="store_true",
                    help="colour MLP: Fourier positional encoding on position "
                         "(not Lipschitz-constrained; sharper appearance)")
    ap.add_argument("--rgb-multires", type=int, default=_tc.rgb_multires,
                    help="colour MLP PE bands when --rgb-pe (default 6)")
    ap.add_argument("--w-feature",  type=float, default=_tc.w_feature,
                    help="weight for cosine distance on precomputed feature maps")
    ap.add_argument("--feature-maps", type=Path, default=_tc.feature_maps,
                    help="path to a .pt file produced by archive/precompute_mast3r_features.py")
    ap.add_argument("--n-alt",      type=int,   default=_tc.n_alt,
                    help="nearest-neighbour alt cameras per ray (pool size; "
                         "e.g. 10 with --ncc-topk 4, or 6 for the legacy pool)")
    ap.add_argument("--view-selection", type=str, default=_tc.view_selection,
                    choices=["nearest", "pairs_file", "arccos", "arccos_nn"],
                    help="how to pick the n_alt source views per reference: "
                         "'nearest' (camera-centre NN, default), 'pairs_file' "
                         "(first n_alt ranked ids from --pairs-path), 'arccos' "
                         "(angular distance of viewing dirs; also enables 2-level "
                         "uniform-camera ray sampling), or 'arccos_nn' (same "
                         "angular-distance ranking but ref-view + ray sampling "
                         "stay exactly as 'nearest')")
    ap.add_argument("--pairs-path", type=Path, default=_tc.pairs_path,
                    help="MVSNet/NeuralWarp pair.txt; required for "
                         "--view-selection pairs_file (scores ignored)")
    ap.add_argument("--w-ncc",      type=float, default=_tc.w_ncc)
    ap.add_argument("--w-ncc-normal", type=float, default=_tc.w_ncc_normal,
                    help="weight of the normal-branch PMVS NCC term "
                         "L=w_ncc·NCC(x,detach(n))+w_ncc_normal·NCC(detach(x),n); "
                         ">0 enables a differentiable normal (double-backward)")
    ap.add_argument("--ncc-detach-normals", action=argparse.BooleanOptionalAction,
                    default=_tc.ncc_detach_normals,
                    help="detach normals in the position-branch NCC tangent patch "
                         "(use --no-ncc-detach-normals for the normal-gradient ablation)")
    ap.add_argument("--ncc-attach-normal-point", action=argparse.BooleanOptionalAction,
                    default=_tc.ncc_attach_normal_point,
                    help="'detach nothing' ablation: evaluate the normal at the "
                         "differentiable x_theta so the loss also carries the "
                         "position→normal term ∂n/∂x·∂x_theta/∂θ. Needs "
                         "--no-ncc-detach-normals; adds a curvature-weighted term.")
    ap.add_argument("--ncc-patch", type=int, default=_tc.ncc_patch,
                    help="PMVS patch side P (PxP sample grid)")
    ap.add_argument("--ncc-half-pix", type=float, default=_tc.ncc_half_pix,
                    help="PMVS patch half-width in reference-view pixels")
    ap.add_argument("--ncc-world-patch", type=float, default=_tc.ncc_world_patch,
                    help="object-fixed patch footprint in WORLD units (full grid "
                         "span). >0 → constant metric footprint, view-independent "
                         "(overrides --ncc-half-pix sizing). <0 → legacy per-view "
                         "sizing. e.g. Ignatius 8mm @0.883 m/unit ≈ 0.009")
    ap.add_argument("--ncc-normal-patch", type=int, default=_tc.ncc_normal_patch,
                    help="normal-branch patch P (<0 → share --ncc-patch); the "
                         "normal-branch ZNCC leverage scales with patch extent, "
                         "so it usually wants a larger patch, e.g. 9 or 11")
    ap.add_argument("--ncc-normal-half-pix", type=float, default=_tc.ncc_normal_half_pix,
                    help="normal-branch patch half-width in px (<0 → share "
                         "--ncc-half-pix); pair with --ncc-normal-patch≈2·hp+1")
    ap.add_argument("--ncc-min", type=float, default=_tc.ncc_min,
                    help="PMVS photometric gate: drop pairs with ZNCC below this")
    ap.add_argument("--ncc-sat-tau", type=float, default=_tc.ncc_sat_tau,
                    help="reference-saturation gate: drop rays whose reference pixel "
                         "max-channel ≥ τ (specular highlight). ≤0 disabled; τ≈0.9 typical")
    ap.add_argument("--ncc-topk", type=int, default=_tc.ncc_topk,
                    help="0: mean over all valid alt views; >0: per-point top-K "
                         "best ZNCC across the n_alt pool (robust MVS, use 3-4 "
                         "with n_alt~10)")
    ap.add_argument("--ncc-abs-tau", type=float, default=_tc.ncc_abs_tau,
                    help=">=0: fixed-batch NCC reward -(1/|B|)Σ H_i(z_i-tau); tau is "
                         "an absolute keep/carve bar in the loss VALUE (not a gate; "
                         "ncc_min ignored). <0: legacy kept-mean loss")
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
    ap.add_argument("--ncc-bilateral-gamma", type=float, default=_tc.ncc_bilateral_gamma,
                    help="Gipuma adaptive-support γ in w=exp(-|Ip-Iq|/γ), reference view, "
                         "intensities in [0,1]; fixed large patch, 0=disabled")
    ap.add_argument("--ncc-bilateral-gamma-end", type=float, default=_tc.ncc_bilateral_gamma_end,
                    help="anneal γ exponentially to this by end of training; γ↓ shrinks the "
                         "effective patch (coarse→fine). e.g. 1.0 → 0.05")
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
    ap.add_argument("--w-geo-sdf",      type=float, default=_tc.w_geo_sdf,
                    help="pure Geo-Neus L1 SDF loss on COLMAP points (surface term only, "
                         "no free-space/behind bundle). Independent of --w-sfm.")
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
    ap.add_argument("--lr-warm-restart", action="store_true",
                    help="on --resume, keep weights but reset the optimiser + cosine LR "
                         "schedule (fresh lr->eta_min anneal over the remaining steps); "
                         "use when extending a converged run whose LR floored out")
    ap.add_argument("--dtu-eval-dir", default=None,
                    help="path to DTU evaluation data (SampleSet/ + ObsMask/ subdirs)")
    _ec = EvalConfig()
    ap.add_argument("--dtu-chamfer-freq", type=int, default=_ec.dtu_chamfer_freq,
                    help="cadence (steps) for the cheap in-training sfm_surf diagnostic (0=off)")
    ap.add_argument("--blender-chamfer-freq", type=int, default=_ec.blender_chamfer_freq,
                    help="cadence (steps) for in-training Blender GT chamfer (0=off)")
    ap.add_argument("--blender-official-freq", type=int, default=_ec.blender_official_freq,
                    help="cadence (steps) for full official HF-NeuS Blender Chamfer w/ fg-mask crop (0=off)")
    ap.add_argument("--blender-official-res", type=int, default=_ec.blender_official_res,
                    help="MC resolution for periodic official Blender Chamfer")
    ap.add_argument("--blender-official-bound", type=float, default=_ec.blender_official_bound,
                    help="MC bound for periodic official Blender Chamfer")
    ap.add_argument("--blender-official-n-samples", type=int, default=_ec.blender_official_n_samples,
                    help="area-uniform surface samples per mesh for periodic official Blender Chamfer")
    ap.add_argument("--blender-official-mask-crop", action=argparse.BooleanOptionalAction,
                    default=_ec.blender_official_mask_crop,
                    help="crop periodic official Blender eval mesh with dilated Blender alpha masks "
                         f"(default {_ec.blender_official_mask_crop})")
    ap.add_argument("--blender-official-mask-dilate-px", type=int,
                    default=_ec.blender_official_mask_dilate_px,
                    help="mask dilation radius in pixels for periodic official Blender eval crop")
    ap.add_argument("--blender-official-mask-crop-min-ratio", type=float,
                    default=_ec.blender_official_mask_crop_min_ratio,
                    help="required fraction of in-frame mask projections for official Blender eval crop")
    ap.add_argument("--blender-official-mask-crop-min-views", type=int,
                    default=_ec.blender_official_mask_crop_min_views,
                    help="minimum in-frame views for official Blender eval crop")
    ap.add_argument("--dtu-chamfer-res", type=int, default=_ec.dtu_chamfer_res,
                    help="MC resolution for the sfm_surf diagnostic")
    ap.add_argument("--dtu-official-freq", type=int, default=_ec.dtu_official_freq,
                    help="DTUeval-python official Chamfer frequency in steps (0=off)")
    ap.add_argument("--dtu-official-res", type=int, default=_ec.dtu_official_res,
                    help="MC resolution for periodic DTUeval-python official Chamfer")
    ap.add_argument("--dtu-official-bound", type=float, default=_ec.dtu_official_bound,
                    help="MC bound for periodic DTUeval-python official Chamfer")
    ap.add_argument("--dtu-official-mask-crop", action=argparse.BooleanOptionalAction,
                    default=_ec.dtu_official_mask_crop,
                    help="crop periodic DTU official eval mesh with dilated DTU foreground masks "
                         f"(default {_ec.dtu_official_mask_crop})")
    ap.add_argument("--dtu-official-mask-dilate-px", type=int,
                    default=_ec.dtu_official_mask_dilate_px,
                    help="mask dilation radius in pixels for periodic DTU official eval crop")
    ap.add_argument("--dtu-official-mask-crop-min-ratio", type=float,
                    default=_ec.dtu_official_mask_crop_min_ratio,
                    help="required fraction of in-frame mask projections for periodic DTU official eval crop")
    ap.add_argument("--dtu-official-mask-crop-min-views", type=int,
                    default=_ec.dtu_official_mask_crop_min_views,
                    help="minimum in-frame views for periodic DTU official eval crop")
    ap.add_argument("--tnt-eval-dir", default=None,
                    help="root or scene dir with official TnT GT assets "
                         "(<scene>.ply / _trans.txt / .json / _COLMAP_SfM.log)")
    ap.add_argument("--tnt-official-scene", default=None,
                    help="official TnT scene name; useful when training from a staged scene dir")
    ap.add_argument("--tnt-official-frame",
                    choices=["colmap-pose", "nsvf", "colmap-local", "colmap-sfm"],
                    default=None,
                    help="official TnT alignment frame; current COLMAP-derived scenes use colmap-pose")
    ap.add_argument("--tnt-official-freq", type=int, default=None,
                    help="official TnT F-score frequency in steps (0=off)")
    ap.add_argument("--tnt-official-res", type=int, default=None,
                    help="MC resolution for periodic official TnT F-score")
    ap.add_argument("--tnt-official-bound", type=float, default=None,
                    help="MC bound for periodic official TnT F-score")
    ap.add_argument("--tnt-official-n-samples", type=int, default=None,
                    help="number of sampled surface points for periodic official TnT F-score")
    ap.add_argument("--bmvs-eval-dir", default=None,
                    help="GT root with <relpath> or GT_meshes/<relpath> for known bmvs_* scenes")
    ap.add_argument("--bmvs-gt-mesh", default=None,
                    help="explicit raw BlendedMVS GTMeshRaw.ply (overrides --bmvs-eval-dir lookup)")
    ap.add_argument("--bmvs-official-freq", type=int, default=None,
                    help="BlendedMVS Chamfer frequency in steps (0=off); step 0 included")
    ap.add_argument("--bmvs-official-res", type=int, default=None,
                    help="MC resolution for periodic BlendedMVS Chamfer")
    ap.add_argument("--bmvs-official-bound", type=float, default=None,
                    help="MC bound for periodic BlendedMVS Chamfer")
    ap.add_argument("--bmvs-official-n-samples", type=int, default=None,
                    help="samples per surface for periodic BlendedMVS Chamfer (VolSDF B.2 uses 100K)")
    ap.add_argument("--bmvs-official-protocol", choices=["volsdf", "probesdf"], default=None,
                    help="BlendedMVS Chamfer protocol (default volsdf = paper supplementary B.2)")
    ap.add_argument("--bmvs-official-ground-axis", type=int, default=None,
                    help="volsdf: axis (0=x,1=y,2=z) normal to the ground plane")
    ap.add_argument("--bmvs-official-ground-value", type=float, default=None,
                    help="volsdf: drop geometry below this offset (normalized frame); omit to skip")
    ap.add_argument("--mc-level", type=float, default=_ec.mc_level,
                    help="marching-cubes isovalue (default 0.0); slightly >0 (e.g. 0.005) "
                         "trims noisy near-zero wandering in under-supervised pockets")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="explicit run directory (overrides auto-timestamped name); ignored on --resume")
    ap.add_argument("--viewer",      action="store_true")
    ap.add_argument("--viewer-res",  type=int, default=256)
    ap.add_argument("--profile", action="store_true",
                    help="one-shot compute/memory breakdown of the model at startup")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                    help="torch.compile(dynamic=True) the hot-path SDF forward "
                         f"(default {_tc.compile}; fp32-exact, no TF32/autocast). "
                         "Use --no-compile to disable; overrides the value in --config.")
    ap.add_argument("--viewer-port", type=int, default=8080)
    ap.add_argument("--render-down", type=int, default=1,
                    help="downsample for sphere-traced PNG renders (1=full res, 2=half)")
    ap.add_argument("--mc-res",           type=int,   default=256,
                    help="marching-cubes grid resolution for eval/viewer")
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
            trace=dataclasses.replace(
                run_cfg.trace,
                occ_iters=args.occ_iters,
                occ_newton_steps=args.occ_newton_steps,
                occ_eps=args.occ_eps,
                occ_depth_slack=args.occ_depth_slack,
            ),
            eval=dataclasses.replace(
                run_cfg.eval,
                dtu_eval_dir=Path(args.dtu_eval_dir) if args.dtu_eval_dir else run_cfg.eval.dtu_eval_dir,
                dtu_chamfer_freq=args.dtu_chamfer_freq,
                blender_chamfer_freq=args.blender_chamfer_freq,
                blender_official_freq=args.blender_official_freq,
                blender_official_res=args.blender_official_res,
                blender_official_bound=args.blender_official_bound,
                blender_official_n_samples=args.blender_official_n_samples,
                blender_official_mask_crop=args.blender_official_mask_crop,
                blender_official_mask_dilate_px=args.blender_official_mask_dilate_px,
                blender_official_mask_crop_min_ratio=args.blender_official_mask_crop_min_ratio,
                blender_official_mask_crop_min_views=args.blender_official_mask_crop_min_views,
                dtu_chamfer_res=args.dtu_chamfer_res,
                dtu_official_freq=args.dtu_official_freq,
                dtu_official_res=args.dtu_official_res,
                dtu_official_bound=args.dtu_official_bound,
                dtu_official_mask_crop=args.dtu_official_mask_crop,
                dtu_official_mask_dilate_px=args.dtu_official_mask_dilate_px,
                dtu_official_mask_crop_min_ratio=args.dtu_official_mask_crop_min_ratio,
                dtu_official_mask_crop_min_views=args.dtu_official_mask_crop_min_views,
                tnt_eval_dir=Path(args.tnt_eval_dir) if args.tnt_eval_dir else run_cfg.eval.tnt_eval_dir,
                tnt_official_scene=(args.tnt_official_scene
                                    if args.tnt_official_scene is not None
                                    else run_cfg.eval.tnt_official_scene),
                tnt_official_frame=(args.tnt_official_frame
                                    if args.tnt_official_frame is not None
                                    else run_cfg.eval.tnt_official_frame),
                tnt_official_freq=(args.tnt_official_freq
                                   if args.tnt_official_freq is not None
                                   else run_cfg.eval.tnt_official_freq),
                tnt_official_res=(args.tnt_official_res
                                  if args.tnt_official_res is not None
                                  else run_cfg.eval.tnt_official_res),
                tnt_official_bound=(args.tnt_official_bound
                                    if args.tnt_official_bound is not None
                                    else run_cfg.eval.tnt_official_bound),
                tnt_official_n_samples=(args.tnt_official_n_samples
                                        if args.tnt_official_n_samples is not None
                                        else run_cfg.eval.tnt_official_n_samples),
                bmvs_eval_dir=(Path(args.bmvs_eval_dir) if args.bmvs_eval_dir
                               else run_cfg.eval.bmvs_eval_dir),
                bmvs_gt_mesh=(Path(args.bmvs_gt_mesh) if args.bmvs_gt_mesh
                              else run_cfg.eval.bmvs_gt_mesh),
                bmvs_official_freq=(args.bmvs_official_freq
                                    if args.bmvs_official_freq is not None
                                    else run_cfg.eval.bmvs_official_freq),
                bmvs_official_res=(args.bmvs_official_res
                                   if args.bmvs_official_res is not None
                                   else run_cfg.eval.bmvs_official_res),
                bmvs_official_bound=(args.bmvs_official_bound
                                     if args.bmvs_official_bound is not None
                                     else run_cfg.eval.bmvs_official_bound),
                bmvs_official_n_samples=(args.bmvs_official_n_samples
                                         if args.bmvs_official_n_samples is not None
                                         else run_cfg.eval.bmvs_official_n_samples),
                bmvs_official_protocol=(args.bmvs_official_protocol
                                        if args.bmvs_official_protocol is not None
                                        else run_cfg.eval.bmvs_official_protocol),
                bmvs_official_ground_axis=(args.bmvs_official_ground_axis
                                           if args.bmvs_official_ground_axis is not None
                                           else run_cfg.eval.bmvs_official_ground_axis),
                bmvs_official_ground_value=(args.bmvs_official_ground_value
                                            if args.bmvs_official_ground_value is not None
                                            else run_cfg.eval.bmvs_official_ground_value),
                mc_level=args.mc_level,
            ),
        )
        if args.steps != _tc.steps:
            # Honour an explicit --steps on the --config/--resume path (e.g. extending
            # a finished run to a larger horizon for a warm restart). Without this the
            # config's own `steps` would silently win.
            run_cfg = dataclasses.replace(
                run_cfg, train=dataclasses.replace(run_cfg.train, steps=args.steps))
        if args.down != _tc.down:
            # Honour an explicit --down on the --config path (same pattern as --steps).
            # Without this the config's own `down` silently wins, so --down 2 is a no-op
            # and the full-res deterministic ray set (H*W*V) OOMs on many-view scenes.
            run_cfg = dataclasses.replace(
                run_cfg, train=dataclasses.replace(run_cfg.train, down=args.down))
        if args.t_far != _trc.t_far:
            # Honour an explicit --t-far on the --config/--resume path (same pattern as
            # --steps). Without this the config's own `t_far` silently wins. Needed when a
            # shared source config's flat t_far is too short for a scene whose cameras sit
            # farther from the origin (e.g. bmvs_jade cams at ~7-8 vs t_far=5 → 0 hits).
            run_cfg = dataclasses.replace(
                run_cfg, trace=dataclasses.replace(run_cfg.trace, t_far=args.t_far))
        if args.profile:
            run_cfg = dataclasses.replace(
                run_cfg, train=dataclasses.replace(run_cfg.train, profile=True))
        if args.compile is not None:
            run_cfg = dataclasses.replace(
                run_cfg, train=dataclasses.replace(run_cfg.train, compile=args.compile))
        if args.use_masks is not None:
            run_cfg = dataclasses.replace(
                run_cfg, train=dataclasses.replace(run_cfg.train, use_masks=args.use_masks))
        if args.bake_background is not None:
            run_cfg = dataclasses.replace(
                run_cfg, train=dataclasses.replace(run_cfg.train, bake_background=args.bake_background))
        # view/sampling overrides on the --config path (same `!= default` pattern as --steps)
        if args.view_selection != _tc.view_selection:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, view_selection=args.view_selection, pairs_path=args.pairs_path))
        if args.init_hit_sampling != _tc.init_hit_sampling:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, init_hit_sampling=args.init_hit_sampling))
        if args.fg_fraction != _tc.fg_fraction:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, fg_fraction=args.fg_fraction))
        if args.ncc_abs_tau != _tc.ncc_abs_tau:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, ncc_abs_tau=args.ncc_abs_tau))
        if args.ncc_sat_tau != _tc.ncc_sat_tau:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, ncc_sat_tau=args.ncc_sat_tau))
        if args.ncc_detach_normals != _tc.ncc_detach_normals:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, ncc_detach_normals=args.ncc_detach_normals))
        if args.ncc_attach_normal_point != _tc.ncc_attach_normal_point:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, ncc_attach_normal_point=args.ncc_attach_normal_point))
        if args.w_geo_sdf != _tc.w_geo_sdf:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, w_geo_sdf=args.w_geo_sdf))
        # colour MLP overrides on the --config path (same `!= default` pattern as
        # --steps): without these an explicit --w-rgb is silently dropped and the
        # config's own w_rgb=0 wins, so the colour MLP is never built.
        if args.w_rgb != _tc.w_rgb:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, w_rgb=args.w_rgb,
                rgb_hidden=args.rgb_hidden, rgb_depth=args.rgb_depth,
                rgb_view_dep=not args.rgb_no_view_dep,
                rgb_input_encoding=("pe" if args.rgb_pe else "identity"),
                rgb_multires=args.rgb_multires))
        if args.force_fg_bg_split != _tc.force_fg_bg_split:
            run_cfg = dataclasses.replace(run_cfg, train=dataclasses.replace(
                run_cfg.train, force_fg_bg_split=args.force_fg_bg_split))
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
            trace=TraceConfig(iters=args.trace_iters, eps=args.eps,
                              occ_iters=args.occ_iters,
                              occ_newton_steps=args.occ_newton_steps,
                              occ_eps=args.occ_eps,
                              occ_depth_slack=args.occ_depth_slack,
                              bsphere_radius=args.bsphere_radius,
                              bsphere_start_radius=args.bsphere_start_radius,
                              t_far=args.t_far, sdf_min_beta=args.sdf_min_beta),
            init=InitConfig(
                init=args.init,
                steps=args.init_steps,
                radius=args.radius,
                hull_res=args.hull_res,
                hull_sfm_roi=args.hull_sfm_roi,
                hull_min_views=args.hull_min_views,
                hull_border_aware=args.hull_border_aware,
                w_sfm_free=args.w_sfm_free,
                sfm_free_eps=args.sfm_free_eps,
                init_mesh=args.init_mesh,
                init_sdf_grid=args.init_sdf_grid,
                points_poisson_depth=args.points_poisson_depth,
                points_trim_quantile=args.points_trim_quantile,
                points_normal_knn=args.points_normal_knn,
            ),
            train=TrainConfig(
                steps=args.steps, batch=args.batch, lr=args.lr, down=args.down,
                profile=args.profile,
                compile=(_tc.compile if args.compile is None else args.compile),
                use_masks=(_tc.use_masks if args.use_masks is None else args.use_masks),
                bake_background=(_tc.bake_background if args.bake_background is None else args.bake_background),
                grad_weighted_sampling=args.grad_weighted_sampling,
                grad_sampling_alpha=args.grad_sampling_alpha,
                fg_fraction=args.fg_fraction,
                force_fg_bg_split=args.force_fg_bg_split,
                init_hit_sampling=args.init_hit_sampling,
                use_blender=args.blender, single_view=args.single_view,
                w_photo=args.w_photo, w_feature=args.w_feature, feature_maps=args.feature_maps,
                w_rgb=args.w_rgb, rgb_hidden=args.rgb_hidden, rgb_depth=args.rgb_depth,
                rgb_view_dep=not args.rgb_no_view_dep,
                rgb_input_encoding=("pe" if args.rgb_pe else "identity"),
                rgb_multires=args.rgb_multires,
                n_alt=args.n_alt,
                view_selection=args.view_selection, pairs_path=args.pairs_path,
                w_ncc=args.w_ncc, w_ncc_normal=args.w_ncc_normal,
                ncc_detach_normals=args.ncc_detach_normals,
                ncc_attach_normal_point=args.ncc_attach_normal_point,
                ncc_patch=args.ncc_patch,
                ncc_half_pix=args.ncc_half_pix, ncc_min=args.ncc_min,
                ncc_sat_tau=args.ncc_sat_tau,
                ncc_world_patch=args.ncc_world_patch,
                ncc_topk=args.ncc_topk,
                ncc_abs_tau=args.ncc_abs_tau,
                ncc_color=args.ncc_color,
                ncc_grad_alpha=args.ncc_grad_alpha,
                ncc_normal_patch=args.ncc_normal_patch,
                ncc_normal_half_pix=args.ncc_normal_half_pix,
                sample_mode=args.sample_mode,
                gaussian_sigma=args.gaussian_sigma, gaussian_sigma_end=args.gaussian_sigma_end,
                gaussian_radius=args.gaussian_radius,
                ncc_patch_wsigma=args.ncc_patch_wsigma,
                ncc_patch_wsigma_end=args.ncc_patch_wsigma_end,
                ncc_bilateral_gamma=args.ncc_bilateral_gamma,
                ncc_bilateral_gamma_end=args.ncc_bilateral_gamma_end,
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
                w_geo_sdf=args.w_geo_sdf,
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
                blender_chamfer_freq=args.blender_chamfer_freq,
                blender_official_freq=args.blender_official_freq,
                blender_official_res=args.blender_official_res,
                blender_official_bound=args.blender_official_bound,
                blender_official_n_samples=args.blender_official_n_samples,
                blender_official_mask_crop=args.blender_official_mask_crop,
                blender_official_mask_dilate_px=args.blender_official_mask_dilate_px,
                blender_official_mask_crop_min_ratio=args.blender_official_mask_crop_min_ratio,
                blender_official_mask_crop_min_views=args.blender_official_mask_crop_min_views,
                dtu_chamfer_res=args.dtu_chamfer_res,
                dtu_official_freq=args.dtu_official_freq,
                dtu_official_res=args.dtu_official_res,
                dtu_official_bound=args.dtu_official_bound,
                dtu_official_mask_crop=args.dtu_official_mask_crop,
                dtu_official_mask_dilate_px=args.dtu_official_mask_dilate_px,
                dtu_official_mask_crop_min_ratio=args.dtu_official_mask_crop_min_ratio,
                dtu_official_mask_crop_min_views=args.dtu_official_mask_crop_min_views,
                tnt_eval_dir=Path(args.tnt_eval_dir) if args.tnt_eval_dir else None,
                tnt_official_scene=args.tnt_official_scene,
                tnt_official_frame=(_ec.tnt_official_frame if args.tnt_official_frame is None
                                    else args.tnt_official_frame),
                tnt_official_freq=(_ec.tnt_official_freq if args.tnt_official_freq is None
                                   else args.tnt_official_freq),
                tnt_official_res=(_ec.tnt_official_res if args.tnt_official_res is None
                                  else args.tnt_official_res),
                tnt_official_bound=(_ec.tnt_official_bound if args.tnt_official_bound is None
                                    else args.tnt_official_bound),
                tnt_official_n_samples=(_ec.tnt_official_n_samples
                                        if args.tnt_official_n_samples is None
                                        else args.tnt_official_n_samples),
                bmvs_eval_dir=(Path(args.bmvs_eval_dir) if args.bmvs_eval_dir else None),
                bmvs_gt_mesh=(Path(args.bmvs_gt_mesh) if args.bmvs_gt_mesh else None),
                bmvs_official_freq=(_ec.bmvs_official_freq if args.bmvs_official_freq is None
                                    else args.bmvs_official_freq),
                bmvs_official_res=(_ec.bmvs_official_res if args.bmvs_official_res is None
                                   else args.bmvs_official_res),
                bmvs_official_bound=(_ec.bmvs_official_bound if args.bmvs_official_bound is None
                                     else args.bmvs_official_bound),
                bmvs_official_n_samples=(_ec.bmvs_official_n_samples
                                         if args.bmvs_official_n_samples is None
                                         else args.bmvs_official_n_samples),
                bmvs_official_protocol=(_ec.bmvs_official_protocol
                                        if args.bmvs_official_protocol is None
                                        else args.bmvs_official_protocol),
                bmvs_official_ground_axis=(_ec.bmvs_official_ground_axis
                                           if args.bmvs_official_ground_axis is None
                                           else args.bmvs_official_ground_axis),
                bmvs_official_ground_value=(_ec.bmvs_official_ground_value
                                            if args.bmvs_official_ground_value is None
                                            else args.bmvs_official_ground_value),
                mc_level=args.mc_level,
            ),
            scene=scene_path,
        )

    view_keep = None
    if args.good_views is not None:
        from .data import load_view_keep
        view_keep = load_view_keep(args.good_views)
        print(f"  good-views: keeping {len(view_keep)} views from {args.good_views}")

    ckpt_path: Path | None = None
    if not args.no_train:
        ckpt_path = train(run_cfg, use_wandb=args.wandb, resume=args.resume,
                          run_dir=args.run_dir, view_keep=view_keep,
                          lr_warm_restart=args.lr_warm_restart)

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
                          dtu_chamfer_res=args.dtu_chamfer_res,
                          dtu_chamfer_freq=args.dtu_chamfer_freq,
                          blender_chamfer_freq=args.blender_chamfer_freq,
                          blender_official_freq=args.blender_official_freq,
                          blender_official_res=args.blender_official_res,
                          blender_official_bound=args.blender_official_bound,
                          blender_official_n_samples=args.blender_official_n_samples,
                          blender_official_mask_crop=args.blender_official_mask_crop,
                          blender_official_mask_dilate_px=args.blender_official_mask_dilate_px,
                          blender_official_mask_crop_min_ratio=args.blender_official_mask_crop_min_ratio,
                          blender_official_mask_crop_min_views=args.blender_official_mask_crop_min_views,
                          dtu_official_freq=args.dtu_official_freq,
                          dtu_official_res=args.dtu_official_res,
                          dtu_official_bound=args.dtu_official_bound,
                          dtu_official_mask_crop=args.dtu_official_mask_crop,
                          dtu_official_mask_dilate_px=args.dtu_official_mask_dilate_px,
                          dtu_official_mask_crop_min_ratio=args.dtu_official_mask_crop_min_ratio,
                          dtu_official_mask_crop_min_views=args.dtu_official_mask_crop_min_views,
                          tnt_eval_dir=Path(args.tnt_eval_dir) if args.tnt_eval_dir else None,
                          tnt_official_scene=args.tnt_official_scene,
                          tnt_official_frame=(_ec.tnt_official_frame if args.tnt_official_frame is None
                                              else args.tnt_official_frame),
                          tnt_official_freq=(_ec.tnt_official_freq if args.tnt_official_freq is None
                                             else args.tnt_official_freq),
                          tnt_official_res=(_ec.tnt_official_res if args.tnt_official_res is None
                                            else args.tnt_official_res),
                          tnt_official_bound=(_ec.tnt_official_bound if args.tnt_official_bound is None
                                              else args.tnt_official_bound),
                          tnt_official_n_samples=(_ec.tnt_official_n_samples
                                                  if args.tnt_official_n_samples is None
                                                  else args.tnt_official_n_samples),
                          mc_level=args.mc_level)
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
