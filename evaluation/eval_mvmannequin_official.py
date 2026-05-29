#!/usr/bin/env python3
"""Official MVMannequin Chamfer eval — exact port of Inria-Morpheo's protocol
(scripts/run_mvmannequin.py:evalChamfer in https://gitlab.inria.fr/projects-morpheo/millimetrichumans).

Protocol:
  1. Load pred + GT meshes; bring both into the calibration world frame.
  2. Slice plane z > 0.05 m  (cuts the floor / mannequin stand).
  3. Take the largest connected component of the predicted mesh (drop floaters).
  4. ICP refinement (Open3D point-to-plane, Tukey loss, max_corr=25 mm) of
     pred -> gt; apply the inverse to gt so the prediction stays unmoved.
  5. Compute point-to-MESH signed distances (pysdf) both ways:
       dists0 = |sdf_pred(gt.verts)|     # completeness   (cd0)
       dists1 = |sdf_gt(pred.verts)|     # accuracy       (cd1)
  6. Clamp distances at 0.1 m (drop numerical-error outliers).
  7. cd0 = mean(dists0), cd1 = mean(dists1)   in meters (×1000 for mm).

Outputs to <out_dir>:
  pred_world_mesh.ply    predicted mesh in calibration world frame (m)
  gt_world_mesh.ply      GT mesh in calibration world frame (m), pre-slice
  cd0, cd1               text files with mean dists in meters
  dists0.npy, dists1.npy raw per-vertex distances
  transform              4x4 final GT-alignment matrix (m @ gtMeshTransform)
  gt_mesh_cmap.ply       GT vertices coloured by dists0
  test_mesh_cmap.ply     pred vertices coloured by dists1
  chamfer_error.png      3-projection scatter + histograms
  mvmannequin_official.json   metrics summary
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
PIXI_PYTHON = Path("/home/glenain/nerfstudio/.pixi/envs/default/bin/python3")

# Protocol constants (verbatim from Inria-Morpheo run_mvmannequin.py)
GROUND_CUTOFF_THRESHOLD = 0.05     # meters
ICP_MAX_CORR            = 0.025    # meters
DIST_CLAMP_M            = 0.1      # meters
HEATMAP_THRESHOLD_MM    = 15.0     # mm


# ─────────────────────── mesh extraction (training-venv side) ──────────────────────

def extract_pred_mesh(ckpt_path: Path, scene: Path, out_ply: Path,
                      bound: float = 1.0, res: int = 512, device: str = "auto"
                      ) -> Path:
    """Marching-cubes the SDF from ckpt and write the predicted mesh in the
    CALIBRATION WORLD frame (multiply normalized verts by scale_mat)."""
    import torch
    from skimage.measure import marching_cubes
    import trimesh
    sys.path.insert(0, str(REPO))
    from lip_tracer.model import make_model

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    arch = ckpt.get("architecture", "cpl")
    hidden = ckpt.get("hidden", 256)
    for k, v in state.items():
        if "weight" in k and v.ndim >= 2 and "head" not in k and "encoder" not in k:
            hidden = v.shape[-1]; break
    f = make_model(hidden=hidden, depth=ckpt.get("depth", 8),
                   group_size=ckpt.get("group_size", 2),
                   activation=ckpt.get("activation", "groupsort"),
                   input_encoding=ckpt.get("input_encoding", "pe"),
                   multires=ckpt.get("multires", 6),
                   architecture=arch).to(device)
    f.load_state_dict(state, strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i + 4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).detach().cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        raise RuntimeError(f"surface not in bounds (vol range [{vol.min():.3f},{vol.max():.3f}])")

    spacing = 2 * bound / (res - 1)
    v_norm, faces, *_ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    v_norm = (v_norm - bound).astype(np.float64)

    scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
    v_h = np.concatenate([v_norm, np.ones((len(v_norm), 1))], axis=1)
    v_world = (scale_mat @ v_h.T).T[:, :3]

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v_world, faces=faces, process=False).export(str(out_ply))
    return out_ply


def write_gt_world_mesh(scene: Path, out_ply: Path) -> Path:
    """gt_mesh.ply in <scene> is in NORMALIZED frame; transform to WORLD via scale_mat."""
    import trimesh
    src = scene / "gt_mesh.ply"
    if not src.exists():
        raise FileNotFoundError(f"missing {src} (run prepare_mvmannequin.py)")
    scale_mat = np.load(scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
    m = trimesh.load(str(src), process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    v_h = np.concatenate([v, np.ones((len(v), 1))], axis=1)
    v_world = (scale_mat @ v_h.T).T[:, :3]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v_world, faces=np.asarray(m.faces), process=False).export(str(out_ply))
    return out_ply


# ────────────────────────── protocol body (pixi-env side) ──────────────────────────

def run_protocol(pred_world_ply: Path, gt_world_ply: Path, out_dir: Path) -> dict:
    """Exact reproduction of Inria-Morpheo evalChamfer. Requires pysdf + open3d."""
    import trimesh
    import open3d as o3d
    from pysdf import SDF

    out_dir.mkdir(parents=True, exist_ok=True)

    gt_mesh   = trimesh.load_mesh(str(gt_world_ply))
    test_mesh = trimesh.load_mesh(str(pred_world_ply))
    print(f"[load] gt  verts={len(gt_mesh.vertices):>9,}  faces={len(gt_mesh.faces):>9,}", flush=True)
    print(f"[load] pred verts={len(test_mesh.vertices):>9,}  faces={len(test_mesh.faces):>9,}", flush=True)

    # 1. Slice ground plane (z > 0.05 m)
    gt_mesh   = gt_mesh.slice_plane((0, 0, GROUND_CUTOFF_THRESHOLD), (0, 0, 1))
    test_mesh = test_mesh.slice_plane((0, 0, GROUND_CUTOFF_THRESHOLD), (0, 0, 1))
    print(f"[slice z>{GROUND_CUTOFF_THRESHOLD}] gt={len(gt_mesh.vertices):,}  pred={len(test_mesh.vertices):,}", flush=True)

    # 2. Largest connected component of prediction
    parts = sorted(test_mesh.split(only_watertight=False),
                   key=lambda x: x.vertices.shape[0], reverse=True)
    if len(parts) > 1:
        print(f"[largest-cc] pred had {len(parts)} components; keeping "
              f"{len(parts[0].vertices):,} verts (was {len(test_mesh.vertices):,})", flush=True)
    test_mesh = parts[0]

    # 3. ICP refine (test -> gt; then apply inverse to gt so pred stays fixed)
    loss = o3d.pipelines.registration.TukeyLoss(k=1.0)
    p2l  = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(test_mesh.vertices)
    target = o3d.geometry.PointCloud()
    target.points  = o3d.utility.Vector3dVector(gt_mesh.vertices)
    target.normals = o3d.utility.Vector3dVector(gt_mesh.vertex_normals)
    reg = o3d.pipelines.registration.registration_icp(
        source, target, ICP_MAX_CORR, estimation_method=p2l)
    icp_inv = np.linalg.inv(reg.transformation)
    gt_mesh.apply_transform(icp_inv)
    print(f"[icp] fitness={reg.fitness:.4f}  inlier_rmse={reg.inlier_rmse*1000:.3f}mm  "
          f"max_corr={ICP_MAX_CORR*1000:.1f}mm", flush=True)

    # 4. Point-to-mesh signed distances (abs); clamp at 0.1 m
    sdf_gt   = SDF(gt_mesh.vertices,   gt_mesh.faces)
    sdf_test = SDF(test_mesh.vertices, test_mesh.faces)
    dists0 = np.abs(sdf_test(gt_mesh.vertices))    # completeness  (cd0)
    dists1 = np.abs(sdf_gt(test_mesh.vertices))    # accuracy      (cd1)
    n_out0 = int((dists0 > DIST_CLAMP_M).sum())
    n_out1 = int((dists1 > DIST_CLAMP_M).sum())
    dists0[dists0 > DIST_CLAMP_M] = DIST_CLAMP_M
    dists1[dists1 > DIST_CLAMP_M] = DIST_CLAMP_M
    cd0 = float(np.mean(dists0))
    cd1 = float(np.mean(dists1))
    chamfer = 0.5 * (cd0 + cd1)
    print(f"[chamfer] cd0(comp)={cd0*1000:.4f}mm  cd1(acc)={cd1*1000:.4f}mm  "
          f"chamfer={chamfer*1000:.4f}mm  clamped={n_out0}/{n_out1}", flush=True)

    # 5. Persist exactly what Inria-Morpheo saves
    np.savetxt(str(out_dir / "cd0"), np.array([cd0]))
    np.savetxt(str(out_dir / "cd1"), np.array([cd1]))
    np.save(str(out_dir / "dists0.npy"), np.array([dists0]))   # match official: wrap in [.]
    np.save(str(out_dir / "dists1.npy"), np.array([dists1]))
    np.savetxt(str(out_dir / "transform"), icp_inv)

    # 6. Coloured PLYs (jet 0..HEATMAP_THRESHOLD_MM)
    import matplotlib.cm as cm
    threshold = HEATMAP_THRESHOLD_MM * 1e-3
    jet = cm.get_cmap("jet")
    colors0 = jet(dists0 / threshold)
    gt_mesh.visual = trimesh.visual.color.ColorVisuals(
        mesh=gt_mesh, vertex_colors=np.uint8(np.clip(colors0, 0, 1) * 255))
    gt_mesh.export(str(out_dir / "gt_mesh_cmap.ply"), encoding="binary")
    colors1 = jet(dists1 / threshold)
    test_mesh.visual = trimesh.visual.color.ColorVisuals(
        mesh=test_mesh, vertex_colors=np.uint8(np.clip(colors1, 0, 1) * 255))
    test_mesh.export(str(out_dir / "test_mesh_cmap.ply"), encoding="binary")

    # 7. JSON summary
    payload = {
        "protocol": "Inria-Morpheo evalChamfer (z>0.05m, largest-cc, ICP-p2l(max_corr=0.025m,Tukey), pysdf, clamp@0.1m)",
        "completeness_mm": cd0 * 1000.0,
        "accuracy_mm":     cd1 * 1000.0,
        "chamfer_mm":      chamfer * 1000.0,
        "cd0_m": cd0, "cd1_m": cd1,
        "n_gt_verts":   int(len(gt_mesh.vertices)),
        "n_pred_verts": int(len(test_mesh.vertices)),
        "n_clamped_comp": n_out0,
        "n_clamped_acc":  n_out1,
        "icp_fitness":    float(reg.fitness),
        "icp_inlier_rmse_mm": float(reg.inlier_rmse * 1000),
        "ground_cutoff_m": GROUND_CUTOFF_THRESHOLD,
        "icp_max_corr_m":  ICP_MAX_CORR,
        "dist_clamp_m":    DIST_CLAMP_M,
    }
    (out_dir / "mvmannequin_official.json").write_text(json.dumps(payload, indent=2))

    # 8. Error PNG
    try:
        _render_error_png(test_mesh, gt_mesh, dists1, dists0,
                          payload["accuracy_mm"], payload["completeness_mm"],
                          out_dir / "chamfer_error.png")
    except Exception as e:
        print(f"[png] skipped: {e}", flush=True)

    print("=" * 60, flush=True)
    print(f"  accuracy     (cd1)  = {payload['accuracy_mm']:.4f} mm   pred -> GT", flush=True)
    print(f"  completeness (cd0)  = {payload['completeness_mm']:.4f} mm   GT   -> pred", flush=True)
    print(f"  chamfer             = {payload['chamfer_mm']:.4f} mm", flush=True)
    print("=" * 60, flush=True)
    return payload


def _render_error_png(test_mesh, gt_mesh, dists1, dists0,
                      acc_mm: float, comp_mm: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    pred = np.asarray(test_mesh.vertices)
    gt   = np.asarray(gt_mesh.vertices)
    acc_mm_arr  = dists1 * 1000.0
    comp_mm_arr = dists0 * 1000.0
    BG = "#0d0d0d"
    vm = HEATMAP_THRESHOLD_MM
    norm = Normalize(0, vm)
    PROJ = [(0, 2, "x", "z (up)"), (1, 2, "y", "z (up)"), (0, 1, "x", "y")]
    rng = np.random.default_rng(0)

    def sub(p, d, n=300_000):
        if len(p) > n:
            i = rng.choice(len(p), n, replace=False); return p[i], d[i]
        return p, d

    pred_s, acc_s = sub(pred, acc_mm_arr)
    gt_s,   comp_s = sub(gt,  comp_mm_arr)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=BG,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.85]})
    fig.patch.set_facecolor(BG)
    for row, (pts, dists, label) in enumerate([
        (pred_s, acc_s, "ACCURACY pred->GT"),
        (gt_s,   comp_s, "COMPLETENESS GT->pred")]):
        order = np.argsort(dists)
        for col, (i, j, xl, yl) in enumerate(PROJ):
            ax = axes[row, col]; ax.set_facecolor(BG)
            sc = ax.scatter(pts[order, i], pts[order, j], c=dists[order],
                            cmap="jet", norm=norm, s=0.4, linewidths=0,
                            alpha=0.85, rasterized=True)
            ax.set_aspect("equal"); ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(f"{label}\n{yl}" if col == 0 else yl, fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("distance (mm)", color="white", fontsize=8)
            cb.ax.tick_params(colors="white", labelsize=7)

        ax = axes[row, 3]; ax.set_facecolor(BG)
        clip = float(min(dists.max(), vm * 2))
        bins = np.linspace(0, clip, 100)
        ax.hist(dists[dists <= clip], bins=bins,
                color="#00c8ff" if row == 0 else "#ff9900",
                alpha=0.75, density=True)
        for v, ls, lbl in [
            (float(dists.mean()), "--", f"mean {dists.mean():.2f}"),
            (float(np.median(dists)), ":", f"p50  {np.median(dists):.2f}"),
            (float(np.percentile(dists, 90)), "-.", f"p90  {np.percentile(dists, 90):.2f}"),
        ]:
            ax.axvline(v, color="white", lw=0.9, ls=ls, alpha=0.8, label=lbl)
        ax.set_xlabel("mm", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"{label.split()[0]} distribution", fontsize=9, color="white")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.legend(fontsize=7, framealpha=0.25, facecolor=BG, edgecolor="#444",
                  labelcolor="white")

    chamfer = 0.5 * (acc_mm + comp_mm)
    fig.suptitle(f"MVMannequin  ·  acc={acc_mm:.3f}  comp={comp_mm:.3f}  "
                 f"chamfer={chamfer:.3f} mm  (heatmap clip {vm:.0f} mm)  "
                 f"[official Inria-Morpheo protocol]",
                 fontsize=12, y=0.99, color="white")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[png] -> {out_path}", flush=True)


# ───────────────────────────────── CLI ─────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="extract pred mesh via MC, requires torch + lip_tracer")
    src.add_argument("--mesh", type=Path, help="pre-extracted pred mesh in WORLD frame")
    ap.add_argument("--scene", type=Path, required=True,
                    help="data/mvmannequin_neus/<scene> (needs cameras.npz + gt_mesh.ply)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no-subprocess", action="store_true",
                    help="run eval in current interpreter (requires pysdf + open3d)")
    args = ap.parse_args()

    out_dir = args.out or (args.ckpt or args.mesh).parent / "mvmannequin_official"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── stage 1 (training-venv): produce pred + gt PLYs in world frame ─────────
    pred_ply = out_dir / "pred_world_mesh.ply"
    gt_ply   = out_dir / "gt_world_mesh.ply"
    if args.mesh is not None:
        import shutil
        if args.mesh.resolve() != pred_ply.resolve():
            shutil.copy(args.mesh, pred_ply)
    else:
        extract_pred_mesh(args.ckpt, args.scene, pred_ply,
                          bound=args.bound, res=args.res, device=args.device)
    if not gt_ply.exists():
        write_gt_world_mesh(args.scene, gt_ply)

    # ── stage 2 (pixi-env): run the protocol ───────────────────────────────────
    try:
        from pysdf import SDF  # noqa: F401
        import open3d  # noqa: F401
        in_correct_env = True
    except ImportError:
        in_correct_env = False

    if in_correct_env or args.no_subprocess:
        run_protocol(pred_ply, gt_ply, out_dir)
    else:
        # subprocess to pixi env
        cmd = [str(PIXI_PYTHON), str(REPO / "eval_mvmannequin_official.py"),
               "--mesh", str(pred_ply), "--scene", str(args.scene),
               "--out", str(out_dir), "--no-subprocess"]
        print(f"[subprocess] -> pixi env", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.stdout: print(r.stdout, end="", flush=True)
        if r.stderr: print(r.stderr, end="", flush=True)
        if r.returncode != 0:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
