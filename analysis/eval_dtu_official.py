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

# Importable when run as `python analysis/<this>.py` from the repo root: Python
# only puts this script's own dir (analysis/) on sys.path, not the repo root,
# so `import lip_tracer` (and sibling helpers' lip_tracer imports) would fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

DTUEVAL_REPO = "https://github.com/jzhangbs/DTUeval-python.git"
DTUEVAL_DIR  = Path(__file__).parent / "DTUeval-python"
PIXI_PYTHON  = Path("/home/glenain/nerfstudio/.pixi/envs/default/bin/python3")

BMVS_GT_REL = {
    "bmvs_clock": "3-clock/GTMeshRaw.ply",
    "bmvs_sculpture": "14-sculpture/GTMeshRaw.ply",
}


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


def _checkpoint_scene(ckpt_path: Path) -> Path | None:
    """Return config.json's scene path for standard run/ckpt/checkpoint_*.pt."""
    for base in (ckpt_path.parent, ckpt_path.parent.parent):
        cfg = base / "config.json"
        if not cfg.exists():
            continue
        try:
            scene = json.loads(cfg.read_text()).get("scene")
        except Exception:
            continue
        if scene:
            return Path(scene)
    return None


def _is_bmvs_scene(scene: Path | None) -> bool:
    return scene is not None and scene.name in BMVS_GT_REL and (scene / "cameras_sphere.npz").exists()


def _extract_normalized_mesh(ckpt_path: Path, bound: float, res: int, device: str) -> Path:
    """Marching-cubes a 1-Lip checkpoint in the normalized training frame."""
    import trimesh
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_dtu_chamfer import _extract_mesh_from_model

    print(f"[mesh] extracting normalized BMVS mesh from {ckpt_path} "
          f"(res={res}, bound={bound})", flush=True)
    verts, faces, _ = _extract_mesh_from_model(ckpt_path, bound, res, device)
    out_ply = ckpt_path.parent / "pred_normalized_mesh.ply"
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(out_ply))
    print(f"[mesh] saved -> {out_ply}", flush=True)
    return out_ply


def _bmvs_gt_mesh_path(scene: Path, gt_root: Path, gt_mesh: Path | None) -> Path:
    if gt_mesh is not None:
        return gt_mesh
    rel = BMVS_GT_REL[scene.name]
    release_scene = scene.name.replace("bmvs_", "")
    candidates = [
        gt_root / "GT_meshes" / rel,
        gt_root / rel,
        gt_root / "BMVS" / release_scene / "GroundTruth.ply",
        gt_root / "meshes" / "BMVS" / release_scene / "GroundTruth.ply",
        gt_root / "GT_meshes" / release_scene / "GroundTruth.ply",
        scene.parent / "GT_meshes" / rel,
        scene / "GTMeshRaw.ply",
        scene / "GroundTruth.ply",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "BMVS GT mesh not found. Expected one of:\n  " +
        "\n  ".join(str(p) for p in candidates)
    )


def _load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    mesh.remove_unreferenced_vertices()
    return mesh


def _transform_mesh(mesh, mat: np.ndarray):
    import trimesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    vh = np.concatenate([v, np.ones((len(v), 1), dtype=np.float64)], axis=1)
    vt = (mat @ vh.T).T[:, :3]
    return trimesh.Trimesh(vertices=vt, faces=np.asarray(mesh.faces), process=False)


def _dilate_masks_disk(masks: np.ndarray, radius: int) -> np.ndarray:
    """NeuralWarp-style foreground mask dilation: skimage morphology.disk(r)."""
    masks = masks.astype(bool)
    if radius <= 0:
        return masks
    from skimage import morphology as morph

    elem = morph.disk(int(radius))
    return np.stack([morph.binary_dilation(m, elem) for m in masks]).astype(bool)


def _foreground_mask_crop_mesh(mesh_ply: Path, scene: Path, out_dir: Path,
                               dilate_px: int,
                               min_ratio: float, min_views: int,
                               chunk: int = 32768) -> tuple[Path, dict]:
    """Cull predicted mesh faces whose centroids project outside DTU masks.

    This mirrors NeuralWarp's eval-time convention closely enough for this
    evaluator: dilate object masks with a disk radius (default 12 px), project
    triangle centroids into the normalized DTU cameras, and keep faces that land
    inside the dilated foreground masks in the required visible views.
    """
    import trimesh
    import lip_tracer.data as data_mod

    if scene is None:
        raise ValueError("--foreground-mask-crop requires --scene")

    views = data_mod.load_views(scene, down=1)
    masks = views["masks"].detach().cpu().numpy().astype(bool)
    masks = _dilate_masks_disk(masks, dilate_px)
    K = views["K"].detach().cpu().numpy().astype(np.float64)
    c2w = views["c2w"].detach().cpu().numpy().astype(np.float64)
    w2c = np.linalg.inv(c2w)
    R = w2c[:, :3, :3]
    tcw = w2c[:, :3, 3]
    H, W = int(views["H"]), int(views["W"])

    cam_path = scene / "cameras.npz"
    if not cam_path.exists():
        cam_path = scene / "cameras_sphere.npz"
    scale_mat = np.load(cam_path)["scale_mat_0"].astype(np.float64)
    s = float(scale_mat[0, 0])
    t = scale_mat[:3, 3].astype(np.float64)

    mesh = _load_mesh(mesh_ply)
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_ply}")
    cent_world = verts[faces].mean(axis=1)
    cent_norm = (cent_world - t) / s

    V = masks.shape[0]
    keep = np.zeros(len(faces), dtype=bool)
    seen_total = np.zeros(len(faces), dtype=np.uint16)
    fg_total = np.zeros(len(faces), dtype=np.uint16)
    min_views = max(1, int(min_views))
    min_ratio = float(min_ratio)

    print(f"[mask-crop] projecting {len(faces):,} face centroids into {V} "
          f"dilated masks (disk radius={dilate_px}px, min_ratio={min_ratio:g}, "
          f"min_views={min_views})", flush=True)
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
        seen_total[start:end] = np.minimum(seen, np.iinfo(np.uint16).max)
        fg_total[start:end] = np.minimum(fg_seen, np.iinfo(np.uint16).max)

    if not keep.any():
        raise RuntimeError("[mask-crop] foreground mask crop removed every face")

    cropped = trimesh.Trimesh(vertices=verts, faces=faces[keep], process=False)
    cropped.remove_unreferenced_vertices()
    suffix = f"_fgmask_dilate{int(dilate_px)}"
    if min_ratio < 1.0:
        suffix += f"_ratio{min_ratio:.2f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped_ply = out_dir / (mesh_ply.stem + suffix + ".ply")
    cropped.export(str(cropped_ply))

    stats = {
        "enabled": True,
        "source": "NeuralWarp-style object masks: skimage.morphology.disk dilation, projected mesh crop",
        "dilate_px": int(dilate_px),
        "min_ratio": min_ratio,
        "min_views": min_views,
        "faces_before": int(len(faces)),
        "faces_after": int(keep.sum()),
        "face_keep_fraction": float(keep.mean()),
        "vertices_before": int(len(verts)),
        "vertices_after": int(len(cropped.vertices)),
        "seen_views_mean": float(seen_total.mean()),
        "fg_views_mean": float(fg_total.mean()),
        "mesh": str(cropped_ply),
    }
    print(f"[mask-crop] faces {stats['faces_before']:,} → "
          f"{stats['faces_after']:,} ({100.0 * stats['face_keep_fraction']:.1f}%) "
          f"→ {cropped_ply}", flush=True)
    return cropped_ply, stats


def _bmvs_chamfer(pred_mesh, gt_mesh, max_dist: float, n_points: int,
                  seed: int = 0) -> tuple[dict, np.ndarray, np.ndarray, str]:
    """ProbeSDF-style BMVS Chamfer. Prefer point-to-mesh SDF; sample-NN fallback."""
    try:
        from pysdf import SDF

        sdf_pred = SDF(pred_mesh.vertices, pred_mesh.faces)
        d_comp = np.abs(sdf_pred(gt_mesh.vertices))       # GT -> pred
        sdf_gt = SDF(gt_mesh.vertices, gt_mesh.faces)
        d_acc = np.abs(sdf_gt(pred_mesh.vertices))        # pred -> GT
        mode = "pysdf vertex-to-mesh"
    except Exception as e:  # noqa: BLE001
        print(f"[bmvs] pysdf unavailable ({e}); using sampled NN fallback",
              flush=True)
        import trimesh
        from scipy.spatial import cKDTree

        pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, n_points, seed=seed)
        gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, n_points, seed=seed + 1)
        d_acc = cKDTree(gt_pts).query(pred_pts, k=1, workers=-1)[0]
        d_comp = cKDTree(pred_pts).query(gt_pts, k=1, workers=-1)[0]
        mode = f"sampled NN ({n_points:,} pts)"

    acc_keep = d_acc < max_dist
    comp_keep = d_comp < max_dist
    acc = float(d_acc[acc_keep].mean()) if acc_keep.any() else float("nan")
    comp = float(d_comp[comp_keep].mean()) if comp_keep.any() else float("nan")
    return {
        "accuracy": acc,
        "completeness": comp,
        "chamfer": 0.5 * (acc + comp),
        "accuracy_p50": float(np.median(d_acc)),
        "accuracy_p90": float(np.percentile(d_acc, 90)),
        "completeness_p50": float(np.median(d_comp)),
        "completeness_p90": float(np.percentile(d_comp, 90)),
        "n_accuracy_kept": int(acc_keep.sum()),
        "n_completeness_kept": int(comp_keep.sum()),
        "n_accuracy_total": int(len(d_acc)),
        "n_completeness_total": int(len(d_comp)),
    }, d_acc, d_comp, mode


def _run_bmvs_eval(args, device: str) -> None:
    """Evaluate BMVS checkpoints inside this DTU entrypoint."""
    scene = args.scene
    out_dir = args.out
    if out_dir is None:
        stem = args.ckpt.parent.parent if args.ckpt is not None else args.mesh.parent
        out_dir = stem / "bmvs_chamfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = _bmvs_gt_mesh_path(scene, args.bmvs_gt_root, args.bmvs_gt_mesh)
    scale_mat = np.load(scene / "cameras_sphere.npz")["scale_mat_0"].astype(np.float64)
    scale_mat_inv = np.linalg.inv(scale_mat)
    raw_units_per_norm = float(np.linalg.norm(scale_mat[:3, :3], axis=0)[0])

    if args.mesh is not None:
        pred_ply = args.mesh
        print(f"[bmvs] using pre-extracted normalized mesh: {pred_ply}", flush=True)
    else:
        pred_ply = _extract_normalized_mesh(args.ckpt, args.bound, args.res, device)
    pred_mesh = _load_mesh(pred_ply)

    gt_mesh = _load_mesh(gt_path)
    gt_space = args.bmvs_gt_space
    if gt_space == "auto":
        # ProbeSDF's released BMVS GroundTruth.ply files are already in the
        # normalized evaluation frame. The original BlendedMVS GTMeshRaw.ply
        # files are raw and need inv(scale_mat_0).
        gt_space = "normalized" if gt_path.name == "GroundTruth.ply" else "raw"
    if gt_space == "raw":
        gt_mesh = _transform_mesh(gt_mesh, scale_mat_inv)

    metrics, d_acc, d_comp, dist_mode = _bmvs_chamfer(
        pred_mesh, gt_mesh, args.bmvs_max_dist, args.bmvs_n_points)
    metrics.update({
        "dataset": "BlendedMVS",
        "scene": scene.name,
        "mesh": str(pred_ply),
        "gt_mesh": str(gt_path),
        "gt_space": gt_space,
        "protocol": "ProbeSDF-style BMVS: GT raw mesh transformed by inv(cameras_sphere scale_mat_0), distances > max_dist ignored",
        "distance_mode": dist_mode,
        "max_dist_normalized": args.bmvs_max_dist,
        "raw_units_per_normalized_unit": raw_units_per_norm,
        "accuracy_x1000": metrics["accuracy"] * 1000.0,
        "completeness_x1000": metrics["completeness"] * 1000.0,
        "chamfer_x1000": metrics["chamfer"] * 1000.0,
        "accuracy_raw_units": metrics["accuracy"] * raw_units_per_norm,
        "completeness_raw_units": metrics["completeness"] * raw_units_per_norm,
        "chamfer_raw_units": metrics["chamfer"] * raw_units_per_norm,
    })
    (out_dir / "bmvs_chamfer.json").write_text(json.dumps(metrics, indent=2))
    np.save(out_dir / "dists_accuracy.npy", d_acc.astype(np.float32))
    np.save(out_dir / "dists_completeness.npy", d_comp.astype(np.float32))

    print("=" * 50, flush=True)
    print(f"  BMVS scene:    {scene.name}", flush=True)
    print(f"  accuracy:      {metrics['accuracy_x1000']:.4f} x1e-3 norm   (pred → GT)", flush=True)
    print(f"  completeness:  {metrics['completeness_x1000']:.4f} x1e-3 norm   (GT → pred)", flush=True)
    print(f"  chamfer:       {metrics['chamfer_x1000']:.4f} x1e-3 norm", flush=True)
    print(f"  distance mode: {dist_mode}", flush=True)
    print("=" * 50, flush=True)
    print(f"[done] {out_dir}", flush=True)


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


def _load_trace_cfg(ckpt_path: Path):
    """Read the run's TraceConfig from config.json next to the checkpoint, so the
    sphere-trace comparison uses EXACTLY the run's trace params (iters / eps /
    newton / bracketing / bsphere). Falls back to defaults if not found."""
    from lip_tracer.config import TraceConfig
    for run_dir in (ckpt_path.parent, ckpt_path.parent.parent):
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            print(f"[trace] trace cfg from {cfg_path}", flush=True)
            return TraceConfig(**cfg["trace"])
    print(f"[trace] no config.json near {ckpt_path} — using TraceConfig defaults",
          flush=True)
    from lip_tracer.config import TraceConfig as _TC
    return _TC()


def _project_points_to_sdf(f, pts_world: np.ndarray, scale_mat: np.ndarray,
                           cfg, device: str, max_step: float,
                           n_iter: int = 8, chunk: int = 200_000
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Project MC-sampled surface points onto the exact fθ=0 isosurface by
    minimal-displacement Newton steps along the gradient:

        x ← x − f(x)·∇f / ‖∇f‖²

    The MC samples already sit ≈on the surface (on flat triangle chords), so each
    point moves only its own |f| residual (the marching-cubes linear-interp
    error) to land on fθ=0 — NO ray standoff, so the cloud is not redistributed
    tangentially. The per-iteration step is clamped to `max_step` (normalized
    units) to stay robust where ‖∇f‖ is small on the non-metric PE field.
    Returns (refined_world [N,3], converged [N] bool, |f| < cfg.eps after)."""
    import torch

    s = float(scale_mat[0, 0])
    t = scale_mat[:3, 3].astype(np.float32)
    x_norm = (pts_world.astype(np.float32) - t) / s

    refined  = pts_world.astype(np.float32).copy()
    conv_all = np.zeros(len(pts_world), dtype=bool)
    n = len(x_norm)
    print(f"[trace] projecting {n:,} MC samples onto fθ=0  "
          f"(Newton×{n_iter}, eps={cfg.eps}, max_step={max_step:.4g})", flush=True)
    for i in range(0, n, chunk):
        x = torch.from_numpy(x_norm[i:i + chunk]).to(device)
        for _ in range(n_iter):
            xr = x.detach().requires_grad_(True)
            with torch.enable_grad():
                fx = f(xr)
                g  = torch.autograd.grad(fx.sum(), xr)[0]
            # Newton toward surface along the gradient; clamp step magnitude.
            step = (fx.detach() / g.pow(2).sum(-1).clamp(min=1e-12)).unsqueeze(-1) * g.detach()
            sn   = step.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            step = step * (sn.clamp(max=max_step) / sn)
            x = x - step
        fx_final = f(x).detach().abs().cpu().numpy()
        refined[i:i + chunk]  = x.detach().cpu().numpy() * s + t
        conv_all[i:i + chunk] = fx_final < cfg.eps
    print(f"[trace] |f|<eps for {conv_all.sum():,}/{n:,} "
          f"({100.0 * conv_all.mean():.1f}%) after projection", flush=True)
    return refined, conv_all


def _sphere_trace_to_surface(f, pts_world: np.ndarray, scale_mat: np.ndarray,
                             cfg, device: str, standoff: float,
                             chunk: int = 200_000
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Refine vertices onto fθ=0 with the project's FULL sphere tracer
    (trace_nograd: sphere-marching + PE sign-change bracketing + cfg.newton_steps
    Newton refinement — exactly the run's trace).

    Each vertex defines a ray along its outward normal ∇fθ/‖∇fθ‖; we start
    `standoff` (normalized units) outside and trace inward. Hits landing within
    `standoff` of the vertex replace it; misses or spurious far crossings keep
    the original vertex. Returns (refined_world [N,3], hit [N] bool)."""
    import torch
    from lip_tracer.sphere_tracing import trace_nograd

    s = float(scale_mat[0, 0])
    t = scale_mat[:3, 3].astype(np.float32)
    x_norm = (pts_world.astype(np.float32) - t) / s

    refined = pts_world.astype(np.float32).copy()
    hit_all = np.zeros(len(pts_world), dtype=bool)
    n = len(x_norm)
    print(f"[trace] full sphere-tracing {n:,} mesh vertices onto fθ=0  "
          f"(standoff={standoff:.4g}, iters={cfg.iters}, eps={cfg.eps}, "
          f"newton={cfg.newton_steps})", flush=True)
    for i in range(0, n, chunk):
        x  = torch.from_numpy(x_norm[i:i + chunk]).to(device)
        xr = x.clone().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(xr).sum(), xr)[0]
        u = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12)   # outward normal
        x_hit, _, hit = trace_nograd(f, x + standoff * u, -u, cfg)
        move = (x_hit - x).norm(dim=-1)                          # along-ray dist to vertex
        ok = (hit & (move < standoff)).cpu().numpy()
        x_hit_world = x_hit.detach().cpu().numpy() * s + t
        seg = refined[i:i + chunk]
        seg[ok] = x_hit_world[ok]
        hit_all[i:i + chunk] = ok
    print(f"[trace] accepted {hit_all.sum():,}/{n:,} "
          f"({100.0 * hit_all.mean():.1f}%) sphere-trace hits", flush=True)
    return refined, hit_all


def _camera_trace_to_surface(f, pts_world: np.ndarray, scale_mat: np.ndarray,
                             views: dict, cfg, device: str, tau: float,
                             chunk: int = 50_000
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Refine vertices by sphere tracing along *real camera rays* (the tracer's
    rendering use). For each vertex x_mc:

      1. find every camera in which x_mc projects inside the foreground mask;
      2. among those, pick the most head-on (max |n·view_dir|, n=∇fθ/‖∇fθ‖) to
         minimise grazing/lateral error;
      3. trace from that camera centre o through the ray u=normalize(x_mc−o);
      4. accept the hit only if ‖x_st−x_mc‖ < tau (normalized) — this rejects
         self-occlusion (the ray hitting a nearer surface) and grazing misses;
      5. otherwise keep x_mc.

    Projection uses the same normalized-frame c2w/K/masks the training loop uses
    (see data.colmap_visibility_matrix). Returns (refined_world [N,3], hit [N])."""
    import torch
    from lip_tracer.sphere_tracing import trace_nograd

    s   = float(scale_mat[0, 0])
    tsm = scale_mat[:3, 3].astype(np.float32)
    x_all = (pts_world.astype(np.float32) - tsm) / s
    N = len(x_all)

    c2w     = views["c2w"].to(device).float()
    w2c     = torch.linalg.inv(c2w)
    R, tcw  = w2c[:, :3, :3], w2c[:, :3, 3]          # world→cam (normalized)
    K       = views["K"].to(device).float()
    masks   = views["masks"].to(device).bool()        # (V,H,W)
    centers = c2w[:, :3, 3]                            # (V,3) normalized cam centres
    H, W    = int(views["H"]), int(views["W"])
    V       = centers.shape[0]

    refined = pts_world.astype(np.float32).copy()
    hit_all = np.zeros(N, dtype=bool)
    n_novis = 0
    print(f"[trace] camera-ray tracing {N:,} mesh vertices ({V} cameras, "
          f"tau={tau:.4g}, iters={cfg.iters}, eps={cfg.eps})", flush=True)
    for i in range(0, N, chunk):
        x  = torch.from_numpy(x_all[i:i + chunk]).to(device)     # (n,3)
        xr = x.clone().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(xr).sum(), xr)[0]
        nrm = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12)   # (n,3) outward normal

        xc  = torch.einsum("vij,nj->vni", R, x) + tcw[:, None, :]  # (V,n,3)
        uvh = torch.einsum("vij,vnj->vni", K, xc)
        uv  = uvh[..., :2] / uvh[..., 2:3].clamp_min(1e-6)
        u_, v_ = uv[..., 0], uv[..., 1]
        in_b = (u_ >= 0) & (u_ < W) & (v_ >= 0) & (v_ < H) & (xc[..., 2] > 1e-4)
        ui  = u_.round().long().clamp(0, W - 1)
        vi  = v_.round().long().clamp(0, H - 1)
        vrow = torch.arange(V, device=device)[:, None].expand_as(ui)
        vis  = in_b & masks[vrow, vi, ui]                         # (V,n) sees & in-mask

        dirv = x[None] - centers[:, None, :]                      # (V,n,3) cam→point
        dirv = dirv / dirv.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        align = (dirv * nrm[None]).sum(-1).abs()                  # (V,n) head-on score
        align = torch.where(vis, align, torch.full_like(align, -1.0))
        best  = align.argmax(dim=0)                               # (n,) chosen camera
        has_vis = vis.any(dim=0)                                  # (n,)

        o = centers[best]                                         # (n,3)
        d = x - o
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        x_hit, _, hit = trace_nograd(f, o, d, cfg)
        move = (x_hit - x).norm(dim=-1)
        ok = (hit & has_vis & (move < tau)).cpu().numpy()
        x_hit_world = x_hit.detach().cpu().numpy() * s + tsm
        seg = refined[i:i + chunk]
        seg[ok] = x_hit_world[ok]
        hit_all[i:i + chunk] = ok
        n_novis += int((~has_vis).sum().item())
    print(f"[trace] camera-ray accepted {hit_all.sum():,}/{N:,} "
          f"({100.0 * hit_all.mean():.1f}%);  {n_novis:,} vertices had no "
          f"in-mask camera (kept original)", flush=True)
    return refined, hit_all


def _run_dtueval(eval_python: str, eval_script: Path, data_path: Path,
                 scan_id: int, dtu_eval_dir: Path, out_dir: Path,
                 mode: str = "mesh") -> tuple[float, float, float]:
    """Run DTUeval-python on a mesh/pcd PLY and parse (acc, comp, chamfer) —
    the official ObsMask + Plane protocol. mode='mesh' samples the surface
    densely (area-uniform, the published protocol); mode='pcd' takes the points
    verbatim."""
    cmd = [eval_python, str(eval_script), "--data", str(data_path),
           "--scan", str(scan_id), "--mode", mode,
           "--dataset_dir", str(dtu_eval_dir), "--vis_out_dir", str(out_dir)]
    print(f"[eval-{mode}] {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stderr:
        print("[stderr]\n" + res.stderr, end="", flush=True)
    if res.returncode != 0:
        raise RuntimeError(f"DTUeval {mode} mode failed (code {res.returncode})")
    return tuple(float(v) for v in res.stdout.strip().splitlines()[-1].split())


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


def _render_error_png_camview(pred: np.ndarray, gt: np.ndarray,
                              acc: float, comp: float,
                              scan_id: int, out_path: Path,
                              scene: Path, view: int,
                              acc_dist: np.ndarray, comp_dist: np.ndarray,
                              coverage: float | None = None,
                              nc: float | None = None, n_hc: int | None = None,
                              curv_top: float | None = None,
                              curv_thresh: float | None = None) -> None:
    """Same per-point Chamfer error as `_render_error_png`, but the spatial
    scatter is the projection into DTU camera `view`'s image plane (instead of
    the 3 orthographic plane-frame axes). pred/gt are raw DTU world (mm) and
    cameras.npz world_mat_<view> is the 3x4 K[R|t] for raw mm → pixels, so the
    projection is a direct matmul. Colour scale / norm / histograms / stats box
    are identical to the orthographic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    cam_path = scene / "cameras.npz"
    if not cam_path.exists():
        cam_path = scene / "cameras_sphere.npz"
    P = np.load(cam_path)[f"world_mat_{view}"][:3, :4].astype(np.float64)

    H = W = None
    img_path = scene / "image" / f"{view:06d}.png"
    photo = None
    if img_path.exists():
        from PIL import Image
        photo = np.array(Image.open(img_path))[..., :3]
        H, W = photo.shape[:2]
    else:
        H, W = 1200, 1600

    def project(pts):
        """Raw mm → (u, v, in_frame). v keeps image convention (y down)."""
        Xh = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        uvw = Xh @ P.T
        z = uvw[:, 2]
        front = z > 1e-6
        u = np.full(len(pts), np.nan); v = np.full(len(pts), np.nan)
        u[front] = uvw[front, 0] / z[front]
        v[front] = uvw[front, 1] / z[front]
        infr = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return u, v, infr

    BG, CMAP = "#0d0d0d", "plasma"
    rng = np.random.default_rng(0)

    def sub(p, d, n=300_000):
        if len(p) > n:
            idx = rng.choice(len(p), n, replace=False)
            return p[idx], d[idx]
        return p, d

    pred_s, acc_s = sub(pred, acc_dist)
    gt_s,   comp_s = sub(gt,   comp_dist)
    vm   = max(float(np.percentile(acc_s, 95)), float(np.percentile(comp_s, 95)), 1e-3)
    norm = Normalize(vmin=0, vmax=vm)

    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.18, wspace=0.28,
                            width_ratios=[1.25, 1])
    ROW_LABELS = ["ACCURACY  pred→GT",
                  "COMPLETENESS  GT→pred (plane-filtered)",
                  "OVERLAY  pred=blue  GT=orange"]

    def setup_img_ax(ax):
        ax.set_aspect("equal"); ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xlabel("u (px)", fontsize=8); ax.set_ylabel("v (px)", fontsize=8)
        if photo is not None:
            ax.imshow(photo, extent=[0, W, H, 0], alpha=0.18, zorder=0)
        _dark_ax(ax)

    # rows 0,1 : error-coloured projections
    for row, (pts, dists) in enumerate([(pred_s, acc_s), (gt_s, comp_s)]):
        ax = fig.add_subplot(gs[row, 0], facecolor=BG)
        u, v, infr = project(pts)
        order = np.argsort(dists[infr])[::-1]      # near-zero on top
        sc = ax.scatter(u[infr][order], v[infr][order], c=dists[infr][order],
                        cmap=CMAP, norm=norm, s=0.5, linewidths=0,
                        alpha=0.85, rasterized=True, zorder=2)
        setup_img_ax(ax)
        ax.set_title(f"{ROW_LABELS[row]}  ·  view {view}  "
                     f"({int(infr.sum()):,}/{len(pts):,} in frame)", fontsize=9)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        _dark_cb(cb, vm, "distance (mm)")

    # row 2 : overlay
    ax = fig.add_subplot(gs[2, 0], facecolor=BG)
    gu, gv, gfr = project(sub(gt, comp_dist, 150_000)[0])
    pu, pv, pfr = project(sub(pred, acc_dist, 150_000)[0])
    ax.scatter(gu[gfr], gv[gfr], s=0.4, color="#ff7f0e", alpha=0.5,
               linewidths=0, rasterized=True, label="GT", zorder=2)
    ax.scatter(pu[pfr], pv[pfr], s=0.4, color="#1f77b4", alpha=0.6,
               linewidths=0, rasterized=True, label="pred", zorder=3)
    setup_img_ax(ax)
    ax.set_title(f"{ROW_LABELS[2]}  ·  view {view}", fontsize=9)
    _dark_legend(ax, markerscale=8, loc="upper right", framealpha=0.3)

    # right column : histograms (rows 0,1) + stats text (row 2)
    BINS = 120
    for ax_col, (d, color, label, mean_v) in enumerate([
        (acc_s,  "#00c8ff", "Accuracy",     acc),
        (comp_s, "#ff9900", "Completeness", comp),
    ]):
        ax = fig.add_subplot(gs[ax_col, 1], facecolor=BG)
        clip = min(float(d.max()), vm * 3)
        bins = np.linspace(0, clip, BINS)
        ax.hist(d[d <= clip], bins=bins, color=color, alpha=0.75, density=True)
        for v_, ls, lbl in [
            (float(d.mean()), "--", f"mean {d.mean():.2f}"),
            (float(np.median(d)), ":", f"p50  {np.median(d):.2f}"),
            (float(np.percentile(d, 90)), "-.", f"p90  {np.percentile(d,90):.2f}"),
        ]:
            ax.axvline(v_, color="white", lw=0.9, ls=ls, alpha=0.8, label=lbl)
        ax.axvline(vm, color="#ff4444", lw=1.0, alpha=0.9, label=f"vmax {vm:.2f}")
        ax.set_xlabel("distance (mm)", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"{label} distribution", fontsize=9, pad=4)
        _dark_ax(ax); _dark_legend(ax)

    ax_txt = fig.add_subplot(gs[2, 1], facecolor=BG); ax_txt.axis("off")
    chamfer = 0.5 * (acc + comp)
    hc_lines = ""
    if coverage is not None:
        hc_lines = (f"\n── high-curv detail ──\n"
                    f"coverage     {coverage:.4f}  (<{curv_thresh:.2f} mm)\n"
                    f"NC           {nc:.4f}\n"
                    f"cov × NC     {coverage*nc:.4f}\n"
                    f"hc pts       {n_hc:,}  (top {curv_top*100:.0f}%)")
    ax_txt.text(0.05, 0.95,
        f"camera view {view}\n\n"
        f"accuracy     {acc:.4f}\ncompleteness {comp:.4f}\nchamfer      {chamfer:.4f}\n\n"
        f"acc  p50  {float(np.median(acc_s)):.3f}\nacc  p90  {float(np.percentile(acc_s,90)):.3f}\n"
        f"comp p50  {float(np.median(comp_s)):.3f}\ncomp p90  {float(np.percentile(comp_s,90)):.3f}\n\n"
        f"pred pts  {len(pred):,}\ngt   pts  {len(gt):,}\nvmax      {vm:.2f} mm\n"
        f"protocol  ObsMask+Plane" + hc_lines,
        transform=ax_txt.transAxes, fontsize=9, va="top", ha="left",
        family="monospace", color="white", linespacing=1.7)

    fig.suptitle(
        f"DTU scan{scan_id}  ·  acc={acc:.3f}  comp={comp:.3f}  chamfer={chamfer:.3f}"
        f"  (colour scale 0–{vm:.2f} mm)  ·  camera view {view}  [official protocol]",
        fontsize=13, y=0.995, color="white")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] error map (view {view}) -> {out_path}", flush=True)


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


def _export_error_ply(pred: np.ndarray, acc_dist: np.ndarray,
                      out_path: Path, frac: float = 0.10) -> None:
    """Minimal PLY of the worst `frac` pred points by accuracy error.

    Colours each surviving point from blue (low) → red (high) by its
    pred→GT distance so the spatial location of large accuracy errors is
    visible when overlaid on the mesh in any PLY viewer.
    """
    pct = 100.0 * (1.0 - frac)
    thr = float(np.percentile(acc_dist, pct))
    keep = acc_dist >= thr
    pts = pred[keep].astype(np.float64)
    d = acc_dist[keep].astype(np.float64)
    if len(pts) == 0:
        print(f"[ply] no high-error pred pts (≥{thr:.2f} mm) — skipped", flush=True)
        return

    vm = max(float(d.max()), 1e-3)
    t = (d / vm).clip(0.0, 1.0)
    r = (t * 255.0).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1.0 - t) * 255.0).astype(np.uint8)

    with open(out_path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(pts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for (x, y, z), ri, gi, bi in zip(pts, r, g, b):
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {ri} {gi} {bi}\n")
    print(f"[ply] worst {frac*100:.0f}% accuracy pred pts (≥{thr:.2f} mm, "
          f"N={len(pts):,}) -> {out_path}", flush=True)


def _bad_points_camera_analysis(pred: np.ndarray, normals: np.ndarray,
                                acc_dist: np.ndarray,
                                scene: Path | None, out_dir: Path,
                                thresh_mm: float = 2.0,
                                dilate_px: int = 12) -> None:
    """P_bad = {x in P_pred : d(x, P_gt) > thresh_mm}; which cameras see them.

    On DTU's inward-facing rig every camera has nearly all points in its
    frustum/foreground, so plain projection is saturated and uninformative.
    "Sees" here is therefore FRONT-FACING visibility: a camera sees a surface
    point only if the point's outward normal faces that camera (dot>0) AND it
    projects inside the dilated foreground mask. That is discriminative — each
    camera sees roughly the hemisphere of P_bad turned toward it.

    Saves bad_points.ply (coloured by #cameras that see it), bad_points_cameras
    .png (per-camera bars + spatial map with camera positions) and .json.
    """
    import json as _json
    bad = acc_dist > thresh_mm
    P = pred[bad].astype(np.float64)
    N = normals[bad].astype(np.float64)
    N /= np.linalg.norm(N, axis=1, keepdims=True).clip(min=1e-9)
    d = acc_dist[bad].astype(np.float64)
    if len(P) == 0:
        print(f"[bad] no pred pts with d>GT > {thresh_mm:g} mm — skipped", flush=True)
        return

    n_views = None; cam_see = cam_front = None; cam_world = None
    if scene is None:
        print("[bad] no --scene → camera visibility skipped (saving ply only)", flush=True)
    else:
        import lip_tracer.data as data_mod
        views = data_mod.load_views(scene, down=1)
        K   = views["K"].detach().cpu().numpy().astype(np.float64)
        c2w = views["c2w"].detach().cpu().numpy().astype(np.float64)
        w2c = np.linalg.inv(c2w)
        R, tcw = w2c[:, :3, :3], w2c[:, :3, 3]
        H, W = int(views["H"]), int(views["W"])
        masks = None
        if views.get("masks") is not None:
            masks = _dilate_masks_disk(
                views["masks"].detach().cpu().numpy().astype(bool), dilate_px)

        cam_path = scene / "cameras.npz"
        if not cam_path.exists():
            cam_path = scene / "cameras_sphere.npz"
        scale_mat = np.load(cam_path)["scale_mat_0"].astype(np.float64)
        s = float(scale_mat[0, 0]); t = scale_mat[:3, 3].astype(np.float64)
        Pn = (P - t) / s
        cam_world = s * c2w[:, :3, 3] + t                     # (V,3) world mm

        V = c2w.shape[0]
        xc  = np.einsum("vij,nj->vni", R, Pn) + tcw[:, None, :]
        uvh = np.einsum("vij,vnj->vni", K, xc)
        z = uvh[..., 2]
        denom = np.where(z > 1e-6, z, 1.0)
        u = uvh[..., 0] / denom; v = uvh[..., 1] / denom
        in_fr = (z > 1e-6) & (xc[..., 2] > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if masks is not None:
            ui = np.rint(u).astype(np.int64).clip(0, W - 1)
            vi = np.rint(v).astype(np.int64).clip(0, H - 1)
            in_fg = np.zeros_like(in_fr)
            for vw in range(V):
                in_fg[vw] = in_fr[vw] & masks[vw, vi[vw], ui[vw]]
        else:
            in_fg = in_fr
        # front-facing: normal points toward camera centre
        view_dir = cam_world[:, None, :] - P[None, :, :]      # (V,N,3)
        view_dir /= np.linalg.norm(view_dir, axis=2, keepdims=True).clip(min=1e-9)
        front = np.einsum("vnj,nj->vn", view_dir, N) > 0.0
        vis = in_fg & front
        cam_see   = vis.sum(axis=1)          # (V,) #bad pts each cam sees
        cam_front = in_fg.sum(axis=1)        # (V,) ignoring orientation
        n_views   = vis.sum(axis=0)          # (N,) #cams per bad pt

        order = np.argsort(cam_see)[::-1]
        print(f"[bad] {len(P):,} pred pts with d>GT > {thresh_mm:g} mm  "
              f"(front-facing: mean {n_views.mean():.1f} cams/pt, "
              f"{int((n_views == 0).sum()):,} seen by 0 cams)", flush=True)
        print("[bad] top cameras by #P_bad front-facing & in-mask:", flush=True)
        for r in order[:8]:
            print(f"        cam {int(r):3d}:  sees={int(cam_see[r]):6d}  "
                  f"(in-mask {int(cam_front[r]):6d})", flush=True)
        _json.dump(
            {"thresh_mm": thresh_mm, "n_bad": int(len(P)),
             "metric": "front_facing_in_mask",
             "per_camera_sees": cam_see.astype(int).tolist(),
             "per_camera_in_mask": cam_front.astype(int).tolist(),
             "n_bad_seen_by_zero_cams": int((n_views == 0).sum())},
            open(out_dir / "bad_points_cameras.json", "w"), indent=2)

    # ── bad_points.ply (coloured viridis by #cameras, else by error) ─────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    cvals = n_views.astype(float) if n_views is not None else d
    clabel = "# cameras seeing point" if n_views is not None else "d→GT (mm)"
    cmax = max(float(np.max(cvals)), 1e-6)
    rgb = (cm.get_cmap("viridis")(cvals / cmax)[:, :3] * 255).astype(np.uint8)
    with open(out_dir / "bad_points.ply", "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(P)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for (x, y, zz), (ri, gi, bi) in zip(P, rgb):
            fh.write(f"{x:.6f} {y:.6f} {zz:.6f} {ri} {gi} {bi}\n")
    print(f"[ply] P_bad (d>{thresh_mm:g}mm, N={len(P):,}) -> {out_dir/'bad_points.ply'}",
          flush=True)

    # ── figure: per-camera bars + 3 spatial maps with camera positions ───────
    BG = "#111317"
    has_cam = cam_see is not None
    ncols = 4 if has_cam else 3
    fig = plt.figure(figsize=(5.4 * ncols, 5.0), facecolor=BG)
    if has_cam:
        ax0 = fig.add_subplot(1, ncols, 1, facecolor=BG)
        idx = np.arange(len(cam_see))
        colr = cm.get_cmap("viridis")(cam_see / max(cam_see.max(), 1))
        ax0.bar(idx, cam_see, color=colr, width=0.9)
        for r in np.argsort(cam_see)[::-1][:5]:
            ax0.annotate(str(int(r)), (r, cam_see[r]), color="white",
                         fontsize=7, ha="center", va="bottom")
        ax0.set_xlabel("camera index")
        ax0.set_ylabel(f"# P_bad front-facing & in-mask")
        ax0.set_title(f"which cameras see P_bad  (d>{thresh_mm:g}mm)", color="white")
        _dark_ax(ax0)
    PROJ = [(0, 2, "x", "z"), (1, 2, "y", "z"), (0, 1, "x", "y")]
    base = 1 if has_cam else 0
    order_pt = np.argsort(cvals)            # draw low-coverage points on top
    for k, (i, j, xl, yl) in enumerate(PROJ):
        ax = fig.add_subplot(1, ncols, base + k + 1, facecolor=BG)
        sc = ax.scatter(P[order_pt, i], P[order_pt, j], c=cvals[order_pt],
                        cmap="viridis", vmin=0, vmax=cmax,
                        s=3, linewidths=0, rasterized=True)
        if cam_world is not None:
            ax.scatter(cam_world[:, i], cam_world[:, j], marker="^",
                       c="#ff5a3c", s=22, linewidths=0, label="cameras")
        ax.set_aspect("equal"); ax.set_xlabel(xl); ax.set_ylabel(yl)
        _dark_ax(ax)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        _dark_cb(cb, cmax, clabel)
    ttl = (f"P_bad: pred pts with d→GT > {thresh_mm:g} mm  (N={len(P):,})  ·  "
           f"colour = #cameras seeing point  ·  ▲ = camera centres")
    fig.suptitle(ttl, color="white", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "bad_points_cameras.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] bad-point camera map -> {out_dir/'bad_points_cameras.png'}", flush=True)


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
    ap.add_argument("--dtu-eval-dir", type=Path, default=None)
    ap.add_argument("--res",         type=int,   default=512)
    ap.add_argument("--bound",       type=float, default=1.0)
    ap.add_argument("--out",         type=Path,  default=None)
    ap.add_argument("--device",      type=str,   default="auto")
    ap.add_argument("--bmvs-gt-root", type=Path, default=Path("data/bmvs_gt"),
                    help="Root containing BMVS GT_meshes/<scene>/GTMeshRaw.ply "
                         "(used only for bmvs_clock/bmvs_sculpture scenes)")
    ap.add_argument("--bmvs-gt-mesh", type=Path, default=None,
                    help="Override BMVS raw GTMeshRaw.ply path")
    ap.add_argument("--bmvs-gt-space", choices=["auto", "raw", "normalized"],
                    default="auto",
                    help="Coordinate frame for BMVS GT mesh. auto treats "
                         "GroundTruth.ply as normalized and GTMeshRaw.ply as raw")
    ap.add_argument("--bmvs-max-dist", type=float, default=0.025,
                    help="BMVS normalized-distance cutoff; ProbeSDF ignores "
                         "larger distances when averaging")
    ap.add_argument("--bmvs-n-points", type=int, default=1_000_000,
                    help="Sample count for BMVS fallback when pysdf is unavailable")
    ap.add_argument("--error-view",  type=int, default=None,
                    help="Also render chamfer_error_view<NN>.png: the same "
                         "per-point error projected into DTU camera <NN>'s image "
                         "plane (requires --scene).")
    ap.add_argument("--curv-top",    type=float, default=0.25,
                    help="Top fraction of GT pts by curvature used for Coverage/NC (default 0.25)")
    ap.add_argument("--curv-thresh", type=float, default=0.5,
                    help="Distance threshold in mm for Coverage (default 0.5)")
    ap.add_argument("--no-trace-compare", dest="trace_compare",
                    action="store_false", default=True,
                    help="Disable the sphere-trace comparison (Newton-project the "
                         "MC mesh vertices onto fθ=0 and re-run --mode mesh eval)")
    ap.add_argument("--trace-offset", type=float, default=0.0,
                    help="Per-iteration Newton-projection step clamp in normalized "
                         "units (0 → auto: 3 MC voxels)")
    ap.add_argument("--trace-cam-tau", type=float, default=0.0,
                    help="Accept a camera-ray sphere-trace hit only if it lands "
                         "within this distance (normalized) of the MC vertex; "
                         "rejects self-occlusion (0 → auto: 5 MC voxels)")
    ap.add_argument("--clean-largest-component", action="store_true",
                    help="IDR cleaning: keep only the largest connected component "
                         "(by area) before DTUeval. Matches the protocol used to "
                         "produce published NeuS/VolSDF/Geo-Neus DTU numbers.")
    ap.add_argument("--foreground-mask-crop", action="store_true",
                    help="NeuralWarp-style eval crop: dilate DTU object masks and "
                         "remove predicted mesh faces whose centroids project "
                         "outside the foreground masks before DTUeval.")
    ap.add_argument("--mask-dilate-px", type=int, default=12,
                    help="Foreground mask dilation radius in pixels for "
                         "--foreground-mask-crop (NeuralWarp default: 12).")
    ap.add_argument("--mask-crop-min-ratio", type=float, default=1.0,
                    help="Required fraction of in-frame projections that must be "
                         "inside foreground masks (1.0 = visual-hull all-views).")
    ap.add_argument("--mask-crop-min-views", type=int, default=1,
                    help="Minimum number of in-frame views required for a face "
                         "to survive --foreground-mask-crop.")
    args = ap.parse_args()

    import re
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    if args.scene is None and args.ckpt is not None and not _is_neus_checkpoint(args.ckpt):
        inferred_scene = _checkpoint_scene(args.ckpt)
        if inferred_scene is not None:
            args.scene = inferred_scene
            print(f"[info] scene auto-resolved from checkpoint config: {args.scene}",
                  flush=True)

    if _is_bmvs_scene(args.scene):
        _run_bmvs_eval(args, device)
        return

    if args.dtu_eval_dir is None:
        ap.error("--dtu-eval-dir is required for DTU scenes")

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

    out_dir = args.out or mesh_ply.parent / "dtu_official"
    out_dir.mkdir(parents=True, exist_ok=True)

    foreground_mask_crop = {"enabled": False}
    if args.foreground_mask_crop:
        if args.scene is None:
            ap.error("--foreground-mask-crop requires --scene")
        mesh_ply, foreground_mask_crop = _foreground_mask_crop_mesh(
            mesh_ply, args.scene, out_dir, args.mask_dilate_px,
            args.mask_crop_min_ratio, args.mask_crop_min_views)

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
    f = scale_mat = None
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
    _obs_keep = _obs_inbound(pred_pts, obs, BB, Res)
    pred_obs = pred_pts[_obs_keep]
    pred_obs_normals = pred_normals[_obs_keep]
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

    if args.error_view is not None and args.scene is not None:
        _render_error_png_camview(
            pred_obs, gt_above, acc, comp, scan_id,
            out_dir / f"chamfer_error_view{args.error_view:03d}.png",
            args.scene, args.error_view,
            acc_dist=acc_dist, comp_dist=comp_dist,
            coverage=coverage, nc=nc, n_hc=n_hc,
            curv_top=args.curv_top, curv_thresh=args.curv_thresh)

    _render_worst_png(pred_obs, gt_above, acc_dist, comp_dist, acc, comp,
                      scan_id, out_dir / "worst10_error.png", rot=rot)

    _export_error_ply(pred_obs, acc_dist, out_dir / "worst10_acc_error.ply")

    # P_bad = {x in P_pred : d(x, P_gt) > 2 mm} + which cameras see them
    _bad_points_camera_analysis(pred_obs, pred_obs_normals, acc_dist,
                                args.scene, out_dir, thresh_mm=2.0,
                                dilate_px=args.mask_dilate_px)

    _render_normal_error_png(hc_pts, angle_deg, nc, coverage, product,
                             scan_id, out_dir / "normal_error.png",
                             curv_top=args.curv_top,
                             curv_thresh=args.curv_thresh,
                             ana_angle_deg=ana_angle_deg, ana_nc=ana_nc,
                             rot=rot)

    _render_threshold_curve_png(acc_dist, comp_dist, hc_dist,
                                args.curv_thresh, scan_id,
                                out_dir / "threshold_curve.png")

    # ── sphere-trace (mesh-refine) comparison ─────────────────────────────────
    # Snap the marching-cubes mesh VERTICES onto the exact fθ=0 isosurface (the
    # MC vertices sit at the linearly-interpolated zero crossing on grid edges;
    # Newton projection along ∇f moves each onto the true zero, removing the MC
    # interpolation error), keep the same faces, and re-run the SAME --mode mesh
    # eval. mesh-vs-mesh, dense-vs-dense → the only difference is the exact zero
    # crossing, so this directly answers "did sphere tracing beat the mesh?".
    trace_metrics = None
    if args.trace_compare and args.ckpt is not None and args.scene is not None:
        try:
            import trimesh
            if f is None:
                f = _load_sdf_model(args.ckpt, device)
            if scale_mat is None:
                scale_mat = np.load(args.scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
            trace_cfg = _load_trace_cfg(args.ckpt)
            max_step = (args.trace_offset if args.trace_offset > 0
                        else 3.0 * (2.0 * args.bound / args.res))

            m_in  = trimesh.load(str(mesh_ply), force="mesh", process=False)
            verts = np.asarray(m_in.vertices, dtype=np.float32)
            cmp_dir = out_dir / "trace_compare"
            cmp_dir.mkdir(exist_ok=True)

            # (a) Newton-projection refinement
            v_newton, conv_n = _project_points_to_sdf(
                f, verts, scale_mat, trace_cfg, device, max_step)
            move_n = np.linalg.norm(v_newton - verts, axis=1)
            newton_ply = cmp_dir / "pred_traced_newton.ply"
            trimesh.Trimesh(vertices=v_newton, faces=m_in.faces,
                            process=False).export(str(newton_ply))

            # (b) full sphere-trace refinement (run's trace_nograd)
            standoff = 2.0 * (2.0 * args.bound / args.res)
            v_sphere, hit_s = _sphere_trace_to_surface(
                f, verts, scale_mat, trace_cfg, device, standoff)
            move_s = np.linalg.norm(v_sphere - verts, axis=1)
            sphere_ply = cmp_dir / "pred_traced_sphere.ply"
            trimesh.Trimesh(vertices=v_sphere, faces=m_in.faces,
                            process=False).export(str(sphere_ply))

            n_acc, n_comp, n_cham = _run_dtueval(
                eval_python, eval_script, newton_ply, scan_id,
                args.dtu_eval_dir, cmp_dir, mode="mesh")
            s_acc, s_comp, s_cham = _run_dtueval(
                eval_python, eval_script, sphere_ply, scan_id,
                args.dtu_eval_dir, cmp_dir, mode="mesh")

            # (c) camera-ray sphere-trace on the dense MC surface SAMPLES (pcd).
            # Occlusion-aware: each MC sample is traced from a camera that sees it
            # in-mask; hits landing within tau replace it, occluded/grazing keep
            # the sample. Evaluated as a point cloud (--mode pcd) against the same
            # pcd baseline of the raw samples — the mesh eval above is untouched.
            cam_metrics = None
            try:
                import lip_tracer.data as data_mod
                views = data_mod.load_views(args.scene, down=1)
                cam_tau = (args.trace_cam_tau if args.trace_cam_tau > 0
                           else 5.0 * (2.0 * args.bound / args.res))
                traced_pts, hit_c = _camera_trace_to_surface(
                    f, pred_pts, scale_mat, views, trace_cfg, device, cam_tau)
                mc_ply  = cmp_dir / "pred_samples_mc.ply"
                cam_ply = cmp_dir / "pred_samples_camera_spheretraced.ply"
                trimesh.PointCloud(pred_pts).export(str(mc_ply))
                trimesh.PointCloud(traced_pts).export(str(cam_ply))
                mc_pcd  = _run_dtueval(eval_python, eval_script, mc_ply, scan_id,
                                       args.dtu_eval_dir, cmp_dir, mode="pcd")
                cam_pcd = _run_dtueval(eval_python, eval_script, cam_ply, scan_id,
                                       args.dtu_eval_dir, cmp_dir, mode="pcd")
                cam_metrics = {
                    "tau_norm": cam_tau, "hit_frac": float(hit_c.mean()),
                    "samples_pcd": {"accuracy": mc_pcd[0],  "completeness": mc_pcd[1],  "chamfer": mc_pcd[2]},
                    "camera_pcd":  {"accuracy": cam_pcd[0], "completeness": cam_pcd[1], "chamfer": cam_pcd[2]},
                }
            except Exception as e:  # noqa: BLE001
                print(f"[trace] camera-ray (pcd) variant skipped: {e}", flush=True)

            trace_metrics = {
                "trace_cfg": {"iters": trace_cfg.iters, "eps": trace_cfg.eps,
                              "newton_steps": trace_cfg.newton_steps,
                              "bsphere_radius": trace_cfg.bsphere_radius,
                              "t_far": trace_cfg.t_far},
                "max_step_norm": max_step, "standoff_norm": standoff,
                "mesh":          {"accuracy": acc,    "completeness": comp,    "chamfer": chamfer},
                "traced_newton": {"accuracy": n_acc,  "completeness": n_comp,  "chamfer": n_cham,
                                  "converged_frac": float(conv_n.mean()),
                                  "vertex_move_norm_mean": float(move_n.mean())},
                "traced_sphere": {"accuracy": s_acc,  "completeness": s_comp,  "chamfer": s_cham,
                                  "hit_frac": float(hit_s.mean()),
                                  "vertex_move_norm_mean": float(move_s.mean())},
                "camera_pcd_compare": cam_metrics,
            }
            print("─" * 60, flush=True)
            print("  sphere-trace (mesh-refine) comparison (official --mode mesh)", flush=True)
            print(f"  mesh                 acc={acc:.4f}  comp={comp:.4f}  chamfer={chamfer:.4f}", flush=True)
            print(f"  traced mesh (newton) acc={n_acc:.4f}  comp={n_comp:.4f}  chamfer={n_cham:.4f}   "
                  f"Δ={n_cham-chamfer:+.4f}  (moved {move_n.mean():.4g}, {100*conv_n.mean():.1f}% |f|<eps)", flush=True)
            print(f"  traced mesh (sphere) acc={s_acc:.4f}  comp={s_comp:.4f}  chamfer={s_cham:.4f}   "
                  f"Δ={s_cham-chamfer:+.4f}  (moved {move_s.mean():.4g}, {100*hit_s.mean():.1f}% hit)", flush=True)
            print("─" * 60, flush=True)
            if cam_metrics is not None:
                mc_pcd, cam_pcd = cam_metrics["samples_pcd"], cam_metrics["camera_pcd"]
                print("  camera-ray sphere-trace on MC samples (official --mode pcd)", flush=True)
                print(f"  MC samples (pcd)     acc={mc_pcd['accuracy']:.4f}  comp={mc_pcd['completeness']:.4f}  chamfer={mc_pcd['chamfer']:.4f}", flush=True)
                print(f"  camera-traced (pcd)  acc={cam_pcd['accuracy']:.4f}  comp={cam_pcd['completeness']:.4f}  chamfer={cam_pcd['chamfer']:.4f}   "
                      f"Δ={cam_pcd['chamfer']-mc_pcd['chamfer']:+.4f}  ({100*hit_c.mean():.1f}% accepted)", flush=True)
                print("─" * 60, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[trace] sphere-trace comparison skipped: {e}", flush=True)

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
               "protocol": "DTUeval-python (ObsMask + Plane filter)",
               "foreground_mask_crop": foreground_mask_crop,
               "trace_compare": trace_metrics}
    (out_dir / "dtu_official.json").write_text(json.dumps(payload, indent=2))
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
