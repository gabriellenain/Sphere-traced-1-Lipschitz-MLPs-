#!/usr/bin/env python
"""Per-scene occupancy on the identical 256³ / [±bound] grid training uses:

  GT (perfect)  — screened-Poisson watertight mesh of the DTU GT point cloud,
                  mapped into the normalized frame via the scan's scale_mat,
                  occupancy-queried on the grid. NOTE: DTU GT is a partial cloud
                  (missing occluded bottom) so Poisson hallucinates a closed
                  bottom → this is a slight OVER-estimate of the true volume.
  Visual hull   — 'occupied voxels' from that run's train.log (what init used).
  Reconstruction— interior fraction (f<0) of the converged SDF.

Ordering expectation:  hull ≥ GT ≈ reconstruction.  hull ≫ recon ⇒ bloated hull.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lip_tracer.model import make_model

GT_STL = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_eval/"
              "SampleSet/MVS Data/Points/stl")


def discover_runs(roots):
    best = {}
    for root in roots:
        for d in sorted(root.glob("run_*scan*"), reverse=True):
            m = re.search(r"(scan\d+)", d.name)
            if not m or m.group(1) in best:
                continue
            if not list((d / "ckpt").glob("*.pt")):
                continue
            log = d / "train.log"
            if log.exists() and "occupied voxels:" in log.read_text(errors="ignore"):
                best[m.group(1)] = d
    return best


def hull_pct(run):
    """Hull occupancy fraction. The 'occupied voxels' count is measured at
    hull_res (the carve resolution), so normalize by hull_res³ — NOT by the
    GT/recon grid res."""
    hull_res = json.loads((run / "config.json").read_text())["init"].get("hull_res", 256)
    for line in (run / "train.log").read_text(errors="ignore").splitlines():
        if "occupied voxels:" in line:
            n = int(re.search(r"(\d+)", line.split("occupied voxels:")[1]).group(1))
            return 100.0 * n / (hull_res ** 3)
    return float("nan")


@torch.no_grad()
def recon_pct(run, res, bound, device, chunk):
    cfg = json.loads((run / "config.json").read_text())["model"]
    ckpt = torch.load(sorted((run / "ckpt").glob("*.pt"),
                             key=lambda p: p.stat().st_mtime)[-1], map_location=device)
    f = make_model(**cfg).to(device).eval(); f.load_state_dict(ckpt["f"])
    lin = torch.linspace(-bound, bound, res)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    inside = 0
    for i in range(0, pts.shape[0], chunk):
        inside += int((f(pts[i:i+chunk].to(device)).squeeze(-1) < 0).sum())
    return 100.0 * inside / pts.shape[0]


def gt_pct(run, res, bound, depth, chunk, mode="ratio", min_ratio=0.6, min_views=5):
    """GT object volume: isolate the object from the full structured-light scan
    (ObsMask crop + reproject-into-foreground-mask filter — same object test the
    repo's DTU Chamfer uses), then screened-Poisson → occupancy on the grid.
    Without this filtering the support/background plane dominates the volume."""
    from compare_dtu_chamfer import _in_obs, _mask_filter_gt
    from scipy.io import loadmat
    cfg = json.loads((run / "config.json").read_text())
    scan = int(re.search(r"scan(\d+)", run.name).group(1))
    scene = Path(cfg["scene"])
    dtu_eval = Path(cfg.get("eval", {}).get("dtu_eval_dir")
                    or str(GT_STL.parent.parent))
    Sinv = np.linalg.inv(np.load(scene / "cameras.npz")["scale_mat_0"])

    pcd = o3d.io.read_point_cloud(str(dtu_eval / "Points" / "stl" / f"stl{scan:03d}_total.ply"))
    P = np.asarray(pcd.points)                       # world (mm)
    N = np.asarray(pcd.normals) if pcd.has_normals() else None

    mat = loadmat(str(dtu_eval / "ObsMask" / f"ObsMask{scan}_10.mat"))
    keep0 = _in_obs(P, mat["ObsMask"].astype(bool), mat["BB"].astype(np.float64),
                    float(mat["Res"].flat[0]))
    idx0 = np.where(keep0)[0]
    keep1, st = _mask_filter_gt(P[idx0], scene, mode=mode, min_ratio=min_ratio,
                                min_views=min_views)
    sel = idx0[keep1]
    print(f"    GT filter: {len(P)} -> obs {keep0.sum()} -> object {len(sel)} ({st})",
          flush=True)

    Pn = (Sinv @ np.c_[P[sel], np.ones(len(sel))].T).T[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pn)
    if N is not None:
        pcd.normals = o3d.utility.Vector3dVector(N[sel])   # uniform scale ⇒ dirs unchanged
    else:
        pcd.estimate_normals(); pcd.orient_normals_consistent_tangent_plane(30)
    # screened Poisson -> watertight; crop overshoot to the grid box
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    box = o3d.geometry.AxisAlignedBoundingBox([-bound]*3, [bound]*3)
    mesh = mesh.crop(box)
    scene_rc = o3d.t.geometry.RaycastingScene()
    scene_rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    q = np.stack([xx, yy, zz], -1).reshape(-1, 3).astype(np.float32)
    inside = 0
    for i in range(0, q.shape[0], chunk):
        occ = scene_rc.compute_occupancy(o3d.core.Tensor(q[i:i+chunk]))
        inside += int(occ.numpy().sum())
    return 100.0 * inside / q.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["/scratch/_projets_/willow/1-lip-tracer-new/outputs"])
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--out", default="occupancy_compare.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = args.res ** 3
    runs = discover_runs([Path(r) for r in args.roots])
    scans = sorted(runs, key=lambda s: int(re.search(r"\d+", s).group()))
    print(f"device={device} res={args.res} scenes={scans}", flush=True)

    gt, hull, recon = [], [], []
    for s in scans:
        print(f"[{s}] {runs[s].name}", flush=True)
        h = hull_pct(runs[s])
        g = gt_pct(runs[s], args.res, args.bound, args.poisson_depth, args.chunk)
        r = recon_pct(runs[s], args.res, args.bound, device, args.chunk)
        hull.append(h); gt.append(g); recon.append(r)
        print(f"    GT(perfect)={g:.3f}%  hull={h:.3f}%  recon={r:.3f}%  "
              f"hull/GT=×{h/g:.1f}  recon/GT=×{r/g:.2f}", flush=True)

    x = np.arange(len(scans)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.8*len(scans)), 5))
    bars = [
        (ax.bar(x - w, gt,    w, label="GT (Poisson, ≈perfect)", color="#59A14F"), gt),
        (ax.bar(x,     hull,  w, label="Visual hull (init)",     color="#4C78A8"), hull),
        (ax.bar(x + w, recon, w, label="Reconstruction (f<0)",   color="#E1812C"), recon),
    ]
    for b, vals in bars:
        for r, v in zip(b, vals):
            ax.text(r.get_x()+r.get_width()/2, v, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=7)
    for xi, (g, h) in enumerate(zip(gt, hull)):
        if g > 0:
            ax.text(xi, max(g, h)*1.12, f"hull ×{h/g:.1f}", ha="center",
                    fontsize=8, color="#B22222", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(scans)
    ax.set_ylabel(f"% of {args.res}³ grid occupied (box [±{args.bound}])")
    ax.set_title("DTU per-scene occupancy: GT (Poisson) vs visual hull vs reconstruction\n"
                 "GT bottom is hallucinated by Poisson → slight over-estimate; "
                 "red = hull/GT bloat factor")
    ax.legend(); ax.margins(y=0.20); fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
