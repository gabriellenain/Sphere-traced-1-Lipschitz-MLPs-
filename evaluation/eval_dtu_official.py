#!/usr/bin/env python3
"""Official DTU evaluation using DTUeval-python.

Steps:
1. Extract world-space mesh from checkpoint (or accept --mesh directly)
2. Clone DTUeval-python if not present
3. Run eval.py on the predicted mesh (ObsMask + Plane filter — same as papers)
4. Generate before/after plane-filter PNG and chamfer error PNG
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

DTUEVAL_REPO = "https://github.com/jzhangbs/DTUeval-python.git"
DTUEVAL_DIR  = Path(__file__).parent / "DTUeval-python"
PIXI_PYTHON  = Path("/home/glenain/nerfstudio/.pixi/envs/default/bin/python3")


def _ensure_dtueval() -> Path:
    if not DTUEVAL_DIR.exists():
        print(f"[setup] cloning DTUeval-python -> {DTUEVAL_DIR}", flush=True)
        subprocess.run(["git", "clone", "--depth=1", DTUEVAL_REPO, str(DTUEVAL_DIR)], check=True)
    eval_script = DTUEVAL_DIR / "eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"eval.py not found in {DTUEVAL_DIR}")
    return eval_script


def _is_neus_checkpoint(ckpt_path: Path) -> bool:
    """NeuS-baseline checkpoints live at <run>/checkpoints/ckpt_*.pth and the
    run dir has a NeuS-style run.conf next to them. Heuristic distinct from
    1-Lip .pt checkpoints."""
    return (ckpt_path.suffix == ".pth"
            and (ckpt_path.parent.parent / "run.conf").exists())


def _extract_neus_mesh(ckpt_path: Path, res: int) -> Path:
    """Mirror the NeuS run dir (symlinked data/ + checkpoints/), then call
    NeuS's Runner.validate_mesh in-process at the requested MC res. Returns
    the freshly-extracted world-space PLY."""
    import os, tempfile, re as _re
    src_run  = ckpt_path.parent.parent
    src_conf = src_run / "run.conf"
    conf_txt = src_conf.read_text()
    # data_dir may contain scanN literally, or be a local data/ folder of
    # symlinks to /.../scanN/{image,mask,cameras.npz}. Try both.
    m = _re.search(r"scan(\d+)", conf_txt)
    if not m:
        for child in (src_run / "data").iterdir() if (src_run / "data").exists() else []:
            try:
                tgt = child.resolve()
            except OSError:
                continue
            m = _re.search(r"scan(\d+)", str(tgt))
            if m:
                break
    if not m:
        raise RuntimeError(f"could not parse scan id from {src_conf} or {src_run}/data")
    case = f"scan{m.group(1)}"

    eval_dir = Path(tempfile.mkdtemp(prefix=f"neus_eval_res{res}_", dir=src_run))
    (eval_dir / "data").symlink_to(src_run / "data")
    (eval_dir / "checkpoints").symlink_to(src_run / "checkpoints")
    new_conf = _re.sub(r"^(\s*base_exp_dir\s*=\s*).*$",
                       rf"\1{eval_dir}", conf_txt, flags=_re.M)
    eval_conf = eval_dir / "run.conf"
    eval_conf.write_text(new_conf)

    NEUS_DIR = Path(__file__).parent / "baselines" / "NeuS"
    cwd0 = os.getcwd()
    sys.path.insert(0, str(NEUS_DIR))
    os.chdir(NEUS_DIR)
    try:
        # NeuS calls torch.load() without weights_only; PyTorch ≥2.6 defaults to
        # True and blocks pickled numpy scalars in the NeuS checkpoints. The
        # checkpoint is local + trusted, so force weights_only=False here.
        import torch as _torch
        _orig_load = _torch.load
        def _trusted_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        _torch.load = _trusted_load
        try:
            from exp_runner import Runner  # noqa: E402
            runner = Runner(str(eval_conf), "validate_mesh", case, is_continue=True)
            print(f"[neus] validate_mesh res={res} world=True (case={case}, "
                  f"loaded iter={runner.iter_step})", flush=True)
            runner.validate_mesh(world_space=True, resolution=res, threshold=0.0)
        finally:
            _torch.load = _orig_load
    finally:
        os.chdir(cwd0)

    meshes = sorted((eval_dir / "meshes").glob("*.ply"))
    if not meshes:
        raise RuntimeError(f"NeuS produced no mesh in {eval_dir}/meshes/")
    return meshes[-1]


def _extract_world_mesh(ckpt_path: Path, bound: float, res: int, device: str, scene: Path) -> Path:
    if _is_neus_checkpoint(ckpt_path):
        return _extract_neus_mesh(ckpt_path, res)
    import trimesh
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_dtu_chamfer import (
        _extract_mesh_from_model, _extract_mesh_from_refined,
        _is_refined_checkpoint, _resolve_coarse_pt, _to_world,
    )
    cam_dict  = np.load(scene / "cameras.npz")
    scale_mat = cam_dict["scale_mat_0"].astype(np.float64)
    if _is_refined_checkpoint(ckpt_path):
        coarse = _resolve_coarse_pt(ckpt_path)
        print(f"[mesh] refined: coarse={coarse}  psi={ckpt_path}", flush=True)
        verts, faces, _ = _extract_mesh_from_refined(coarse, ckpt_path, bound, res, device)
    else:
        print(f"[mesh] extracting from {ckpt_path} (res={res}, bound={bound})", flush=True)
        verts, faces, _ = _extract_mesh_from_model(ckpt_path, bound, res, device)
    verts_world = _to_world(verts, scale_mat)
    out_ply = ckpt_path.parent / "pred_world_mesh.ply"
    trimesh.Trimesh(vertices=verts_world, faces=faces, process=False).export(str(out_ply))
    print(f"[mesh] saved -> {out_ply}", flush=True)
    return out_ply


def _compute_gt_curvature_normals(pts: np.ndarray, k: int = 20,
                                   chunk: int = 50_000
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """PCA curvature and normals for a GT point cloud.

    curvature = λ_min / (λ_min + λ_mid + λ_max)  (surface variation).
    normal    = eigenvector of λ_min.
    Returns (curvature [N], normals [N,3]).
    """
    from scipy.spatial import cKDTree

    tree  = cKDTree(pts)
    curv  = np.zeros(len(pts), dtype=np.float32)
    norms = np.zeros((len(pts), 3), dtype=np.float32)

    for start in range(0, len(pts), chunk):
        end = min(start + chunk, len(pts))
        _, nn_idx = tree.query(pts[start:end], k=k, workers=-1)
        neighbors = pts[nn_idx]
        centered  = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov       = np.einsum("...ni,...nj->...ij", centered, centered) / k
        ev, evec  = np.linalg.eigh(cov)                 # ascending eigenvalues
        total     = ev.sum(axis=1) + 1e-10
        curv[start:end]  = (ev[:, 0] / total).astype(np.float32)
        norms[start:end] = evec[:, :, 0].astype(np.float32)

    return curv, norms


def _plane_up_rotation(P: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping the DTU ground-plane normal (the true vertical;
    points with P·[x,1]>0 are *above* it) onto +Z, so the statue stands
    upright in every projection. Shortest-arc (Rodrigues) rotation."""
    n = np.asarray(P, dtype=np.float64).reshape(-1)[:3]
    n = n / (np.linalg.norm(n) + 1e-12)          # up = +normal (toward object)
    ez = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, ez)
    c = float(np.dot(n, ez))
    s = float(np.linalg.norm(v))
    if s < 1e-9:                                  # already (anti)parallel
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _apply_rot(pts: np.ndarray, rot: np.ndarray | None) -> np.ndarray:
    """Rotate points into the upright (plane) frame for display only."""
    if rot is None:
        return pts
    return (pts.astype(np.float32) @ rot.T.astype(np.float32))


def _dark_ax(ax, labelsize: int = 7) -> None:
    """White ticks/labels/title so values are readable on the dark theme."""
    ax.tick_params(colors="white", labelsize=labelsize)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")


def _dark_cb(cb, vmax: float, label: str, fmt: str = "{:.2f}",
             n: int = 5) -> None:
    """Give a colorbar explicit, white numeric ticks (else it reads as an
    unlabelled colour strip on the dark background)."""
    ticks = np.linspace(0.0, float(vmax), n)
    cb.set_ticks(ticks)
    cb.set_ticklabels([fmt.format(t) for t in ticks])
    cb.set_label(label, fontsize=8, color="white")
    cb.ax.tick_params(colors="white", labelsize=7)
    cb.outline.set_edgecolor("#444")


def _dark_legend(ax, **kw):
    """Legend with white text on the dark theme (default kw can be overridden)."""
    opts = dict(fontsize=7, framealpha=0.25, facecolor="#0d0d0d",
                edgecolor="#444", labelcolor="white")
    opts.update(kw)
    return ax.legend(**opts)


def _obs_inbound(pts: np.ndarray, obs: np.ndarray, BB: np.ndarray,
                 Res: float) -> np.ndarray:
    """Boolean ObsMask membership test, identical to the stl filtering used by
    DTUeval-python — so pred can be filtered exactly like the GT is."""
    in_bb = np.all((pts >= BB[0]) & (pts <= BB[1]), axis=1)
    idx   = np.clip(np.round((pts - BB[0]) / Res).astype(int),
                    0, np.array(obs.shape) - 1)
    return in_bb & obs[idx[:, 0], idx[:, 1], idx[:, 2]]


def _load_gt_filtered(dtu_eval_dir: Path, scan_id: int):
    """Return (gt_obs, gt_above, gt_above_curv, gt_above_normals,
    obs_params, rot).

    Applies ObsMask then Plane filter.  Also returns a per-point curvature
    proxy and vertex normals for the plane-filtered subset (used for Coverage
    and Normal Consistency on high-curvature regions), plus obs_params =
    (obs, BB, Res) so the prediction can be ObsMask-filtered the same way the
    GT is when measuring accuracy (DTUeval-python protocol).
    """
    from scipy.io import loadmat
    import trimesh

    ply_path = dtu_eval_dir / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
    obj  = trimesh.load(str(ply_path), process=False)
    stl  = np.asarray(obj.vertices, dtype=np.float32)

    mat = loadmat(str(dtu_eval_dir / "ObsMask" / f"ObsMask{scan_id}_10.mat"))
    obs, BB, Res = mat["ObsMask"].astype(bool), mat["BB"].astype(np.float64), float(mat["Res"].flat[0])
    in_obs = _obs_inbound(stl, obs, BB, Res)
    gt_obs = stl[in_obs]

    P   = loadmat(str(dtu_eval_dir / "ObsMask" / f"Plane{scan_id}.mat"))["P"]
    hom = np.concatenate([gt_obs, np.ones((len(gt_obs), 1), dtype=np.float32)], axis=1)
    above = (P.reshape(1, 4) * hom).sum(-1) > 0
    above_global = np.where(in_obs)[0][above]   # indices into full vertex array
    gt_above     = stl[above_global]

    print(f"[gt]  ObsMask: {len(gt_obs):,}  →  Plane filter: {len(gt_above):,} "
          f"(removed {(~above).sum():,})", flush=True)

    print("[gt]  computing curvature + normals on filtered points…", flush=True)
    gt_above_curv, gt_above_normals = _compute_gt_curvature_normals(gt_above)

    rot = _plane_up_rotation(P)   # upright (plane-aligned) display frame
    return gt_obs, gt_above, gt_above_curv, gt_above_normals, (obs, BB, Res), rot


def _sample_pred(mesh_ply: Path, n: int = 500_000, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pts [N,3], normals [N,3]) sampled uniformly on the pred surface."""
    import trimesh
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    return pts.astype(np.float32), normals


def _load_sdf_model(ckpt_path: Path, device: str):
    """Load the coarse fθ network from a checkpoint (resolves to the coarse
    .pt if a refined deformation checkpoint is passed)."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_dtu_chamfer import _is_refined_checkpoint, _resolve_coarse_pt

    if _is_refined_checkpoint(ckpt_path):
        ckpt_path = _resolve_coarse_pt(ckpt_path)
        print(f"[nθ]  refined ckpt — using coarse fθ: {ckpt_path}", flush=True)

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    architecture = ckpt.get("architecture", "cpl")
    hidden = ckpt.get("hidden", 256)
    depth  = ckpt.get("depth", 8)
    group_size = ckpt.get("group_size", 2)
    activation = ckpt.get("activation", "groupsort")
    input_enc  = ckpt.get("input_encoding", "identity")
    if input_enc == "neus":
        input_enc = "pe"
    multires = ckpt.get("multires", 6)
    for k, v in state.items():
        if "weight" in k and v.ndim >= 2:
            hidden = v.shape[-1]
            break

    from lip_tracer.model import make_model
    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_enc,
                   multires=multires, architecture=architecture)
    f.load_state_dict(state, strict=False)
    return f.to(device).eval()


def _analytic_surface_normals(f, pts_world: np.ndarray, scale_mat: np.ndarray,
                              device: str, chunk: int = 32768) -> np.ndarray:
    """nθ(x) = ∇fθ(x) / ‖∇fθ(x)‖ evaluated at the given world-space points.

    Points are mapped world→normalized by the inverse of the (isotropic
    scale + translation) scale_mat; that scaling preserves directions, so the
    returned unit normals are directly comparable to the world-space GT
    normals (sign is irrelevant — callers use |cos|).
    """
    import torch

    s = float(scale_mat[0, 0])
    t = scale_mat[:3, 3].astype(np.float32)
    x_norm = (pts_world.astype(np.float32) - t) / s

    out = np.empty((len(x_norm), 3), dtype=np.float32)
    n = len(x_norm)
    n_chunks = (n + chunk - 1) // chunk
    log_every = max(1, n_chunks // 10)
    print(f"[nθ]  evaluating ∇fθ at {n:,} surface pts "
          f"({n_chunks} chunks, device={device})", flush=True)
    for ci, i in enumerate(range(0, n, chunk)):
        x = torch.from_numpy(x_norm[i:i + chunk]).to(device).requires_grad_(True)
        with torch.enable_grad():
            y = f(x)
            g = torch.autograd.grad(y.sum(), x, create_graph=False)[0]
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        out[i:i + chunk] = g.detach().cpu().numpy()
        if (ci + 1) % log_every == 0 or (ci + 1) == n_chunks:
            print(f"      [{min(i + chunk, n):>9,} / {n:,}]", flush=True)
    return out


def _compute_coverage_nc(gt_above: np.ndarray, gt_normals: np.ndarray,
                          gt_curv: np.ndarray,
                          pred_pts: np.ndarray, pred_normals: np.ndarray,
                          top_frac: float = 0.25,
                          thresh: float = 0.5
                          ) -> tuple[float, float, int, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Coverage and Normal Consistency restricted to high-curvature GT points.

    - Coverage: fraction of high-curv GT pts whose nearest pred pt is < thresh mm.
    - NC: mean |cos(angle)| between GT normal and nearest pred normal.
    Returns (coverage, nc, n_hc_pts, hc_pts, angle_deg, hc_dist, hc_normals)
    where angle_deg is the per-point normal angular error in [0, 90] degrees,
    hc_dist is the per-point NN distance (mm) to the pred surface, and
    hc_normals are the GT normals of the high-curvature subset.
    """
    from scipy.spatial import cKDTree

    k = max(1, int(top_frac * len(gt_above)))
    hc_idx     = np.argpartition(gt_curv, -k)[-k:]
    hc_pts     = gt_above[hc_idx]
    hc_normals = gt_normals[hc_idx]

    print(f"[hc]  {k:,} high-curvature GT pts (top {top_frac*100:.0f}%)", flush=True)

    dist, nn_idx = cKDTree(pred_pts).query(hc_pts, k=1, workers=-1)
    coverage = float((dist < thresh).mean())

    nn_normals = pred_normals[nn_idx]
    cos       = np.abs((hc_normals * nn_normals).sum(axis=1)).clip(0.0, 1.0)
    nc        = float(cos.mean())
    angle_deg = np.degrees(np.arccos(cos)).astype(np.float32)

    return (coverage, nc, k, hc_pts.astype(np.float32), angle_deg,
            dist.astype(np.float32), hc_normals.astype(np.float32))


def _render_plane_filter_png(gt_obs: np.ndarray, gt_above: np.ndarray,
                              scan_id: int, out_path: Path,
                              rot: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_obs   = _apply_rot(gt_obs, rot)
    gt_above = _apply_rot(gt_above, rot)

    BG   = "#0d0d0d"
    # Plane-aligned frame: axis 2 = up. Side views first (statue upright).
    PROJ = [(0, 2, "u", "up"), (1, 2, "v", "up"), (0, 1, "u", "v")]
    rng  = np.random.default_rng(0)

    def sub(p, n=200_000):
        return p[rng.choice(len(p), n, replace=False)] if len(p) > n else p

    rows = [
        (sub(gt_obs),   "#ff7f0e", "ObsMask only"),
        (sub(gt_above), "#00c8ff", f"ObsMask + Plane filter  ({len(gt_above):,} pts)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    for row, (pts, color, title) in enumerate(rows):
        for col, (i, j, xl, yl) in enumerate(PROJ):
            ax = axes[row, col]; ax.set_facecolor(BG)
            ax.scatter(pts[:, i], pts[:, j], s=0.2, c=color, alpha=0.5,
                       linewidths=0, rasterized=True, label=title)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=8)
            lbl = f"{title}\n{yl}" if col == 0 else yl
            ax.set_ylabel(lbl, fontsize=8)
            _dark_ax(ax)
            if col == 0:
                _dark_legend(ax, markerscale=12, loc="upper right")
    fig.suptitle(f"DTU scan{scan_id} GT — ObsMask vs ObsMask+Plane  "
                 f"({len(gt_obs):,} → {len(gt_above):,} pts, "
                 f"−{len(gt_obs)-len(gt_above):,} pedestal pts)",
                 color="white", fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] plane filter -> {out_path}", flush=True)


def _render_error_png(pred: np.ndarray, gt: np.ndarray,
                      acc: float, comp: float,
                      scan_id: int, out_path: Path,
                      coverage: float | None = None, nc: float | None = None,
                      n_hc: int | None = None, curv_top: float | None = None,
                      curv_thresh: float | None = None,
                      acc_dist: np.ndarray | None = None,
                      comp_dist: np.ndarray | None = None,
                      rot: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.spatial import cKDTree

    if acc_dist is None or comp_dist is None:
        print("[png] computing NN distances for error map…", flush=True)
        acc_dist  = cKDTree(gt).query(pred, k=1, workers=-1)[0].astype(np.float32)
        comp_dist = cKDTree(pred).query(gt,  k=1, workers=-1)[0].astype(np.float32)

    # Rotate coordinates into the upright plane frame (distances unaffected).
    pred = _apply_rot(pred, rot)
    gt   = _apply_rot(gt, rot)

    BG, CMAP = "#0d0d0d", "plasma"
    rng = np.random.default_rng(0)

    def sub(p, d, n=300_000):
        if len(p) > n:
            idx = rng.choice(len(p), n, replace=False)
            return p[idx], d[idx]
        return p, d

    pred_s, acc_s   = sub(pred, acc_dist)
    gt_s,   comp_s  = sub(gt,   comp_dist)

    vm   = max(float(np.percentile(acc_s, 95)), float(np.percentile(comp_s, 95)), 1e-3)
    norm = Normalize(vmin=0, vmax=vm)
    # Plane-aligned frame: axis 2 = up. Side views first (statue upright).
    PROJ = [(0, 2, "u", "up"), (1, 2, "v", "up"), (0, 1, "u", "v")]

    fig = plt.figure(figsize=(22, 15), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.32,
                            width_ratios=[1, 1, 1, 0.85])
    ROW_LABELS = ["ACCURACY  pred→GT", "COMPLETENESS  GT→pred (plane-filtered)",
                  "OVERLAY  pred=blue  GT=orange"]

    for row, (pts, dists) in enumerate([(pred_s, acc_s), (gt_s, comp_s)]):
        for col, (i, j, xl, yl) in enumerate(PROJ):
            ax = fig.add_subplot(gs[row, col], facecolor=BG)
            order = np.argsort(dists)
            sc = ax.scatter(pts[order, i], pts[order, j],
                            c=dists[order], cmap=CMAP, norm=norm,
                            s=0.4, linewidths=0, alpha=0.85, rasterized=True)
            ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
            lbl = f"{ROW_LABELS[row]}\n{yl}" if col == 0 else yl
            ax.set_ylabel(lbl, fontsize=8, labelpad=4)
            _dark_ax(ax)
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            _dark_cb(cb, vm, "distance (mm)")

    pred_ov = sub(pred, acc_dist, 150_000)[0]
    gt_ov   = sub(gt,   comp_dist, 150_000)[0]
    for col, (i, j, xl, yl) in enumerate(PROJ):
        ax = fig.add_subplot(gs[2, col], facecolor=BG)
        ax.scatter(gt_ov[:,   i], gt_ov[:,   j], s=0.3, color="#ff7f0e", alpha=0.5, linewidths=0, rasterized=True, label="GT")
        ax.scatter(pred_ov[:, i], pred_ov[:, j], s=0.3, color="#1f77b4", alpha=0.6, linewidths=0, rasterized=True, label="pred")
        ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
        lbl = f"{ROW_LABELS[2]}\n{yl}" if col == 0 else yl
        ax.set_ylabel(lbl, fontsize=8, labelpad=4)
        _dark_ax(ax)
        if col == 0:
            _dark_legend(ax, markerscale=8, loc="upper right", framealpha=0.3)

    BINS = 120
    for ax_col, (d, color, label, mean_v) in enumerate([
        (acc_s,  "#00c8ff", "Accuracy",     acc),
        (comp_s, "#ff9900", "Completeness", comp),
    ]):
        ax = fig.add_subplot(gs[ax_col, 3], facecolor=BG)
        clip = min(float(d.max()), vm * 3)
        bins = np.linspace(0, clip, BINS)
        ax.hist(d[d <= clip], bins=bins, color=color, alpha=0.75, density=True)
        for v, ls, lbl in [
            (float(d.mean()), "--", f"mean {d.mean():.2f}"),
            (float(np.median(d)), ":", f"p50  {np.median(d):.2f}"),
            (float(np.percentile(d, 90)), "-.", f"p90  {np.percentile(d,90):.2f}"),
        ]:
            ax.axvline(v, color="white", lw=0.9, ls=ls, alpha=0.8, label=lbl)
        ax.axvline(vm, color="#ff4444", lw=1.0, alpha=0.9, label=f"vmax {vm:.2f}")
        ax.set_xlabel("distance (mm)", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"{label} distribution", fontsize=9, pad=4)
        _dark_ax(ax); _dark_legend(ax)

    ax_txt = fig.add_subplot(gs[2, 3], facecolor=BG); ax_txt.axis("off")
    chamfer = 0.5 * (acc + comp)
    hc_lines = ""
    if coverage is not None:
        hc_lines = (f"\n── high-curv detail ──\n"
                    f"coverage     {coverage:.4f}  (<{curv_thresh:.2f} mm)\n"
                    f"NC           {nc:.4f}\n"
                    f"cov × NC     {coverage*nc:.4f}\n"
                    f"hc pts       {n_hc:,}  (top {curv_top*100:.0f}%)")
    ax_txt.text(0.05, 0.95,
        f"accuracy     {acc:.4f}\ncompleteness {comp:.4f}\nchamfer      {chamfer:.4f}\n\n"
        f"acc  p50  {float(np.median(acc_s)):.3f}\nacc  p90  {float(np.percentile(acc_s,90)):.3f}\n"
        f"comp p50  {float(np.median(comp_s)):.3f}\ncomp p90  {float(np.percentile(comp_s,90)):.3f}\n\n"
        f"pred pts  {len(pred):,}\ngt   pts  {len(gt):,}\nvmax      {vm:.2f} mm\n"
        f"protocol  ObsMask+Plane" + hc_lines,
        transform=ax_txt.transAxes, fontsize=9, va="top", ha="left",
        family="monospace", color="white", linespacing=1.7)

    fig.suptitle(
        f"DTU scan{scan_id}  ·  acc={acc:.3f}  comp={comp:.3f}  chamfer={chamfer:.3f}"
        f"  (colour scale 0–{vm:.2f} mm)  [official protocol]",
        fontsize=13, y=0.995, color="white")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] error map   -> {out_path}", flush=True)


def _render_normal_error_png(hc_pts: np.ndarray, angle_deg: np.ndarray,
                             nc: float, coverage: float, product: float,
                             scan_id: int, out_path: Path,
                             curv_top: float, curv_thresh: float,
                             ana_angle_deg: np.ndarray | None = None,
                             ana_nc: float | None = None,
                             rot: np.ndarray | None = None) -> None:
    """Spatial map of per-point normal angular error (3 projections) plus the
    angle-error distribution histogram, on the high-curvature GT subset.

    When ana_angle_deg is given (analytic nθ=∇fθ/‖∇fθ‖ vs GT normal), the
    spatial maps and the primary histogram show that analytic error and the
    mesh-face-normal error is overlaid for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    BG, CMAP = "#0d0d0d", "inferno"
    has_ana = ana_angle_deg is not None
    prim = ana_angle_deg if has_ana else angle_deg   # what the maps colour by
    prim_tag = "nθ=∇fθ/‖∇fθ‖" if has_ana else "mesh-face normal"

    hc_pts = _apply_rot(hc_pts, rot)   # upright plane frame (display only)

    rng = np.random.default_rng(0)
    if len(hc_pts) > 300_000:
        sel = rng.choice(len(hc_pts), 300_000, replace=False)
        pts_s, ang_s = hc_pts[sel], prim[sel]
    else:
        pts_s, ang_s = hc_pts, prim

    vm   = max(float(np.percentile(prim, 95)), 1.0)
    norm = Normalize(vmin=0.0, vmax=vm)
    # Plane-aligned frame: axis 2 = up. Side views first (statue upright).
    PROJ = [(0, 2, "u", "up"), (1, 2, "v", "up"), (0, 1, "u", "v")]

    fig = plt.figure(figsize=(22, 6.4), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.30,
                            width_ratios=[1, 1, 1, 0.9])

    order = np.argsort(ang_s)
    for col, (i, j, xl, yl) in enumerate(PROJ):
        ax = fig.add_subplot(gs[0, col], facecolor=BG)
        sc = ax.scatter(pts_s[order, i], pts_s[order, j],
                        c=ang_s[order], cmap=CMAP, norm=norm,
                        s=0.4, linewidths=0, alpha=0.85, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
        lbl = (f"NORMAL ERROR [{prim_tag}]  high-curv GT "
               f"(top {curv_top*100:.0f}%)\n{yl}"
               if col == 0 else yl)
        ax.set_ylabel(lbl, fontsize=8, labelpad=4)
        _dark_ax(ax)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        _dark_cb(cb, vm, "angle err (deg)", fmt="{:.1f}")

    ax = fig.add_subplot(gs[0, 3], facecolor=BG)
    clip = min(float(max(prim.max(), angle_deg.max())), 90.0)
    bins = np.linspace(0.0, clip, 90)
    if has_ana:
        ax.hist(ana_angle_deg, bins=bins, color="#00e0ff", alpha=0.75,
                density=True, label="analytic nθ")
        ax.hist(angle_deg, bins=bins, histtype="step", lw=1.2,
                color="#ffae00", density=True, label="mesh normal")
    else:
        ax.hist(angle_deg, bins=bins, color="#ffae00", alpha=0.8, density=True,
                label="mesh normal")
    for v, ls, lbl in [
        (float(prim.mean()),            "--", f"mean {prim.mean():.2f}°"),
        (float(np.median(prim)),        ":",  f"p50  {np.median(prim):.2f}°"),
        (float(np.percentile(prim, 90)), "-.",
         f"p90  {np.percentile(prim,90):.2f}°"),
    ]:
        ax.axvline(v, color="white", lw=0.9, ls=ls, alpha=0.8, label=lbl)
    ax.set_xlabel("normal angle error (deg)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_title(f"angle-error distribution [{prim_tag}]", fontsize=9, pad=4)
    _dark_ax(ax); _dark_legend(ax)

    ana_lines = ""
    if has_ana:
        ana_lines = (f"\nNC mesh   {nc:.4f}\n"
                     f"NC nθ     {ana_nc:.4f}\n"
                     f"cov×NC nθ {coverage*ana_nc:.4f}")
    ax.text(0.97, 0.62,
            f"NC        {nc:.4f}\ncoverage  {coverage:.4f}\n"
            f"cov × NC  {product:.4f}\n(<{curv_thresh:.2f} mm)" + ana_lines,
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            family="monospace", color="white", linespacing=1.6)

    ana_sub = (f"  ·  NC(nθ)={ana_nc:.3f}" if has_ana else "")
    fig.suptitle(
        f"DTU scan{scan_id}  ·  normal error on high-curv GT  ·  "
        f"NC(mesh)={nc:.3f}{ana_sub}  cov={coverage:.3f}  "
        f"(0–{vm:.1f}° scale)",
        fontsize=13, y=1.02, color="white")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] normal error -> {out_path}", flush=True)


def _render_worst_png(pred: np.ndarray, gt: np.ndarray,
                      acc_dist: np.ndarray, comp_dist: np.ndarray,
                      acc: float, comp: float,
                      scan_id: int, out_path: Path,
                      frac: float = 0.10,
                      rot: np.ndarray | None = None) -> None:
    """Scatter plot of the worst `frac` fraction of points by NN distance.

    Row 0 : worst accuracy  pts (pred → GT, distance ≥ p{100*(1-frac)})
    Row 1 : worst completeness pts (GT → pred, same threshold)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    BG, CMAP = "#0d0d0d", "plasma"
    PROJ = [(0, 2, "u", "up"), (1, 2, "v", "up"), (0, 1, "u", "v")]

    pct = 100.0 * (1.0 - frac)
    acc_thr  = float(np.percentile(acc_dist,  pct))
    comp_thr = float(np.percentile(comp_dist, pct))

    worst_pred = _apply_rot(pred[acc_dist  >= acc_thr],  rot)
    worst_gt   = _apply_rot(gt[comp_dist   >= comp_thr], rot)
    wd_pred    = acc_dist [acc_dist  >= acc_thr]
    wd_gt      = comp_dist[comp_dist >= comp_thr]

    vm = max(float(wd_pred.max()), float(wd_gt.max()), 1e-3)
    norm = Normalize(vmin=0.0, vmax=vm)

    fig = plt.figure(figsize=(22, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28)

    rows = [
        (worst_pred, wd_pred,
         f"WORST {frac*100:.0f}% ACCURACY  pred→GT  (≥{acc_thr:.2f} mm,  "
         f"N={len(worst_pred):,})"),
        (worst_gt, wd_gt,
         f"WORST {frac*100:.0f}% COMPLETENESS  GT→pred  (≥{comp_thr:.2f} mm,  "
         f"N={len(worst_gt):,})"),
    ]
    for row, (pts, dists, title) in enumerate(rows):
        order = np.argsort(dists)
        for col, (i, j, xl, yl) in enumerate(PROJ):
            ax = fig.add_subplot(gs[row, col], facecolor=BG)
            sc = ax.scatter(pts[order, i], pts[order, j],
                            c=dists[order], cmap=CMAP, norm=norm,
                            s=0.8, linewidths=0, alpha=0.9, rasterized=True)
            ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
            lbl = f"{title}\n{yl}" if col == 0 else yl
            ax.set_ylabel(lbl, fontsize=8, labelpad=4)
            _dark_ax(ax)
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            _dark_cb(cb, vm, "distance (mm)")

    fig.suptitle(
        f"DTU scan{scan_id}  ·  worst {frac*100:.0f}% points  ·  "
        f"acc={acc:.3f}  comp={comp:.3f}  "
        f"(colour scale 0–{vm:.2f} mm)",
        fontsize=13, y=0.995, color="white")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] worst {frac*100:.0f}% map -> {out_path}", flush=True)


def _render_threshold_curve_png(acc_dist: np.ndarray, comp_dist: np.ndarray,
                                 hc_dist: np.ndarray, curv_thresh: float,
                                 scan_id: int, out_path: Path,
                                 tau_max: float | None = None) -> None:
    """Curve of #points (and fraction) with NN distance < tau, as a function
    of tau, for accuracy (pred→GT), completeness (GT→pred) and the
    high-curvature coverage set. Left axis = fraction, right axis = count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG = "#0d0d0d"
    if tau_max is None:
        tau_max = float(max(np.percentile(acc_dist, 95),
                            np.percentile(comp_dist, 95),
                            2.0 * curv_thresh))
    tau = np.linspace(0.0, tau_max, 400)

    series = [
        (np.sort(acc_dist),  "#00c8ff", "accuracy  pred→GT"),
        (np.sort(comp_dist), "#ff9900", "completeness  GT→pred"),
        (np.sort(hc_dist),   "#7CFC00", "coverage  high-curv GT"),
    ]

    fig, (ax, axf) = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for a in (ax, axf):
        a.set_facecolor(BG)
        a.axvline(curv_thresh, color="#ff4444", lw=1.2, ls="--",
                  label=f"curv_thresh = {curv_thresh:.2f} mm")
        a.set_xlim(0.0, tau_max)
        a.set_xlabel("threshold τ (mm)", color="white", fontsize=11)
        a.tick_params(colors="white")
        for sp in a.spines.values(): sp.set_edgecolor("#444")
        a.grid(True, color="#333", lw=0.5, alpha=0.6)

    for sd, color, label in series:
        n = len(sd)
        cnt = np.searchsorted(sd, tau, side="right")   # #points with dist < τ
        ax.plot(tau, cnt, color=color, lw=1.8, label=f"{label}  (N={n:,})")
        axf.plot(tau, cnt / max(n, 1), color=color, lw=1.8, label=label)

    for q, ls in [(0.5, ":"), (0.9, "-.")]:
        axf.axhline(q, color="white", lw=0.7, ls=ls, alpha=0.5)

    ax.set_ylim(bottom=0.0); axf.set_ylim(0.0, 1.0)
    ax.set_ylabel("count of points with dist < τ", color="white", fontsize=11)
    axf.set_ylabel("fraction of points with dist < τ", color="white", fontsize=11)
    ax.set_title("count below threshold", color="white", fontsize=12)
    axf.set_title("fraction below threshold (per-series N)",
                  color="white", fontsize=12)
    for a in (ax, axf):
        a.legend(fontsize=9, framealpha=0.25, facecolor=BG,
                 labelcolor="white", loc="lower right")
    fig.suptitle(f"DTU scan{scan_id}  ·  points below threshold vs τ",
                 color="white", fontsize=14, y=1.0)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] threshold curve -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path)
    src.add_argument("--mesh", type=Path, help="pre-extracted world-space mesh PLY")
    ap.add_argument("--scene",        type=Path, default=None)
    ap.add_argument("--scan-id",      type=int,  default=None)
    ap.add_argument("--dtu-eval-dir", type=Path, required=True)
    ap.add_argument("--res",         type=int,   default=512)
    ap.add_argument("--bound",       type=float, default=1.0)
    ap.add_argument("--out",         type=Path,  default=None)
    ap.add_argument("--device",      type=str,   default="auto")
    ap.add_argument("--curv-top",    type=float, default=0.25,
                    help="Top fraction of GT pts by curvature used for Coverage/NC (default 0.25)")
    ap.add_argument("--curv-thresh", type=float, default=0.5,
                    help="Distance threshold in mm for Coverage (default 0.5)")
    ap.add_argument("--clean-largest-component", action="store_true",
                    help="IDR cleaning: keep only the largest connected component "
                         "(by area) before DTUeval. Matches the protocol used to "
                         "produce published NeuS/VolSDF/Geo-Neus DTU numbers.")
    args = ap.parse_args()

    import re, torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    # ── mesh ──────────────────────────────────────────────────────────────────
    if args.mesh is not None:
        mesh_ply = args.mesh
        print(f"[mesh] using pre-extracted: {mesh_ply}", flush=True)
    else:
        if args.scene is None and not _is_neus_checkpoint(args.ckpt):
            ap.error("--scene required with --ckpt (1-Lip checkpoints)")
        mesh_ply = _extract_world_mesh(args.ckpt, args.bound, args.res, device, args.scene)
    if not mesh_ply.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_ply}")

    # ── IDR cleaning (optional): largest connected component by area ──────────
    if args.clean_largest_component:
        import trimesh
        m = trimesh.load(str(mesh_ply), force="mesh", process=False)
        comps = m.split(only_watertight=False)
        if len(comps) > 1:
            areas = np.array([c.area for c in comps], dtype=np.float64)
            cleaned = comps[int(areas.argmax())]
            cleaned_ply = mesh_ply.with_name(mesh_ply.stem + "_clean.ply")
            cleaned.export(str(cleaned_ply))
            print(f"[clean] IDR largest-component: {len(comps)} → 1  "
                  f"(area frac={areas.max()/areas.sum():.3f})  → {cleaned_ply}",
                  flush=True)
            mesh_ply = cleaned_ply
        else:
            print("[clean] single component already — skipping", flush=True)

    # ── scan id ───────────────────────────────────────────────────────────────
    if args.scan_id is not None:
        scan_id = args.scan_id
    else:
        m = re.search(r"scan(\d+)", str(args.scene or mesh_ply))
        if not m:
            ap.error("Cannot infer scan id — pass --scan-id")
        scan_id = int(m.group(1))

    out_dir = args.out or mesh_ply.parent / "dtu_official"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_script = _ensure_dtueval()

    try:
        import open3d  # noqa: F401
        eval_python = sys.executable
    except ImportError:
        eval_python = str(PIXI_PYTHON) if PIXI_PYTHON.exists() else sys.executable
        print(f"[eval] open3d not in current env — using {eval_python}", flush=True)

    # ── PNG 1: before / after plane filter ───────────────────────────────────
    print("[gt]  loading GT + applying filters…", flush=True)
    gt_obs, gt_above, gt_above_curv, gt_above_normals, obs_params, rot = \
        _load_gt_filtered(args.dtu_eval_dir, scan_id)
    _render_plane_filter_png(gt_obs, gt_above, scan_id,
                             out_dir / "gt_plane_filter.png", rot=rot)

    # ── official eval ─────────────────────────────────────────────────────────
    cmd = [eval_python, str(eval_script),
           "--data", str(mesh_ply), "--scan", str(scan_id),
           "--mode", "mesh", "--dataset_dir", str(args.dtu_eval_dir),
           "--vis_out_dir", str(out_dir)]
    print(f"[eval] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print("[stderr]\n" + result.stderr, end="", flush=True)
    if result.returncode != 0:
        print(f"[eval] exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)

    metrics_line = result.stdout.strip().splitlines()[-1]
    acc, comp, chamfer = [float(v) for v in metrics_line.split()]

    # ── PNG 2: chamfer error map ──────────────────────────────────────────────
    print("[pred] sampling surface points for error map…", flush=True)
    pred_pts, pred_normals = _sample_pred(mesh_ply, n=500_000)

    # ── Coverage + NC on high-curvature GT points ─────────────────────────────
    coverage, nc, n_hc, hc_pts, angle_deg, hc_dist, hc_normals = \
        _compute_coverage_nc(
            gt_above, gt_above_normals, gt_above_curv,
            pred_pts, pred_normals,
            top_frac=args.curv_top, thresh=args.curv_thresh)
    product = coverage * nc

    # ── Analytic SDF normals nθ=∇fθ/‖∇fθ‖ at the surface pts (if a ckpt) ──────
    ana_angle_deg = ana_nc = None
    if args.ckpt is not None and args.scene is not None:
        try:
            scale_mat = np.load(args.scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
            f = _load_sdf_model(args.ckpt, device)
            ana_n = _analytic_surface_normals(f, hc_pts, scale_mat, device)
            cos_a = np.abs((hc_normals * ana_n).sum(axis=1)).clip(0.0, 1.0)
            ana_nc = float(cos_a.mean())
            ana_angle_deg = np.degrees(np.arccos(cos_a)).astype(np.float32)
            print(f"[nθ]  analytic NC = {ana_nc:.4f}  "
                  f"(mesh NC = {nc:.4f})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[nθ]  skipped analytic normals: {e}", flush=True)

    # NN distances — filtered EXACTLY like the official DTUeval Chamfer so the
    # error map and τ-curve are coherent with the reported acc/comp:
    #   accuracy     : ObsMask-filtered pred → ObsMask-only GT (gt_obs)
    #   completeness : ObsMask+Plane GT (gt_above) → full pred
    from scipy.spatial import cKDTree
    obs, BB, Res = obs_params
    pred_obs = pred_pts[_obs_inbound(pred_pts, obs, BB, Res)]
    print(f"[pred] ObsMask filter: {len(pred_pts):,} → {len(pred_obs):,} "
          f"pred pts (accuracy); computing NN distances (acc/comp)…",
          flush=True)
    acc_dist  = cKDTree(gt_obs).query(pred_obs,  k=1, workers=-1)[0].astype(np.float32)
    comp_dist = cKDTree(pred_pts).query(gt_above, k=1, workers=-1)[0].astype(np.float32)

    _render_error_png(pred_obs, gt_above, acc, comp, scan_id,
                      out_dir / "chamfer_error.png",
                      coverage=coverage, nc=nc, n_hc=n_hc,
                      curv_top=args.curv_top, curv_thresh=args.curv_thresh,
                      acc_dist=acc_dist, comp_dist=comp_dist, rot=rot)

    _render_worst_png(pred_obs, gt_above, acc_dist, comp_dist, acc, comp,
                      scan_id, out_dir / "worst10_error.png", rot=rot)

    _render_normal_error_png(hc_pts, angle_deg, nc, coverage, product,
                             scan_id, out_dir / "normal_error.png",
                             curv_top=args.curv_top,
                             curv_thresh=args.curv_thresh,
                             ana_angle_deg=ana_angle_deg, ana_nc=ana_nc,
                             rot=rot)

    _render_threshold_curve_png(acc_dist, comp_dist, hc_dist,
                                args.curv_thresh, scan_id,
                                out_dir / "threshold_curve.png")

    # ── summary ───────────────────────────────────────────────────────────────
    theta_for_p90 = ana_angle_deg if ana_angle_deg is not None else angle_deg
    p90_acc   = float(np.percentile(acc_dist, 90))
    p90_comp  = float(np.percentile(comp_dist, 90))
    p90_theta = float(np.percentile(theta_for_p90, 90))

    print(flush=True)
    print("=" * 50, flush=True)
    print(f"  scan:         {scan_id}", flush=True)
    print(f"  accuracy:     {acc:.4f} mm   (pred → GT)", flush=True)
    print(f"  completeness: {comp:.4f} mm   (GT → pred, plane-filtered)", flush=True)
    print(f"  chamfer:      {chamfer:.4f} mm", flush=True)
    print(f"  p90 acc:      {p90_acc:.4f} mm", flush=True)
    print(f"  p90 comp:     {p90_comp:.4f} mm", flush=True)
    print(f"  p90 θ:        {p90_theta:.4f} °   "
          f"({'analytic nθ' if ana_angle_deg is not None else 'mesh normal'})",
          flush=True)
    print(f"  coverage:     {coverage:.4f}   (high-curv GT, <{args.curv_thresh:.1f} mm, "
          f"top {args.curv_top*100:.0f}%)", flush=True)
    print(f"  NC:           {nc:.4f}   (normal consistency, high-curv GT)", flush=True)
    print(f"  cov × NC:     {product:.4f}   (coverage × normal consistency)", flush=True)
    if ana_nc is not None:
        print(f"  NC (nθ):      {ana_nc:.4f}   (analytic ∇fθ/‖∇fθ‖ vs GT normal)",
              flush=True)
        print(f"  cov × NC(nθ): {coverage*ana_nc:.4f}", flush=True)
    print("=" * 50, flush=True)

    payload = {"scan_id": scan_id, "mesh": str(mesh_ply),
               "dtu_eval_dir": str(args.dtu_eval_dir),
               "accuracy": acc, "completeness": comp, "chamfer": chamfer,
               "p90_acc_mm": p90_acc, "p90_comp_mm": p90_comp,
               "p90_theta_deg": p90_theta,
               "coverage_hc": coverage, "nc_hc": nc,
               "coverage_x_nc": product,
               "nc_analytic": ana_nc,
               "coverage_x_nc_analytic": (None if ana_nc is None
                                          else coverage * ana_nc),
               "curv_top_frac": args.curv_top, "curv_thresh_mm": args.curv_thresh,
               "n_hc_pts": n_hc,
               "n_gt_obsmask": int(len(gt_obs)), "n_gt_plane": int(len(gt_above)),
               "protocol": "DTUeval-python (ObsMask + Plane filter)"}
    (out_dir / "dtu_official.json").write_text(json.dumps(payload, indent=2))
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
