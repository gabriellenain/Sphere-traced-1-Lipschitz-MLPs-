#!/usr/bin/env python3
"""Official TanksAndTemples F-score evaluation for a Lipschitz-SDF checkpoint.

Mirrors the style of analysis/eval_dtu_official.py: takes either --ckpt (we extract
the mesh in-process) or --mesh (already-extracted PLY), plus the GT assets
downloaded from tanksandtemples.org.

Pipeline:
    1. Extract world-space mesh from checkpoint (marching cubes -> un-normalise
       via <scene>/bbox.txt -> TnT world frame).
    2. Sample an area-weighted point cloud.
    3. Apply <scene>_trans.txt then 3-step ICP refinement (registration_vol_ds
       with voxel = tau, tau/2, tau/4), exactly as run.py does.
    4. Call EvaluateHisto from the vendored official toolbox (tnt_eval/),
       which writes precision/recall histograms + colour-coded PLYs.

The vendored files in tnt_eval/ are byte-identical to upstream — we only
add a runtime compat shim because Open3D >=0.10 renamed
`o3d.registration` -> `o3d.pipelines.registration`.

GT assets required (download from tanksandtemples.org):
    <gt_dir>/<scene>.ply         GT point cloud
    <gt_dir>/<scene>_trans.txt   4x4 alignment (recon -> GT frame)
    <gt_dir>/<scene>.json        Open3D SelectionPolygonVolume crop file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def extract_mesh_from_ckpt(ckpt_path: Path, scene_dir: Path,
                           mc_res: int, bound: float) -> "open3d.geometry.TriangleMesh":
    """Load checkpoint -> MC -> un-normalise to TnT world frame -> return mesh."""
    import open3d as o3d
    from skimage import measure
    from lip_tracer.model import make_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- rebuild model from checkpoint -----------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture", "cpl")
    if arch == "neus":
        hidden = ckpt["f"]["layers.0.weight"].shape[0]
    elif "head_weight" in ckpt["f"]:
        hidden = ckpt["f"]["head_weight"].shape[0]
    else:
        hidden = next(v.shape[1] for k, v in ckpt["f"].items()
                      if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
                      and not k.startswith("encoder"))
    if arch == "neus":
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("layers.") and k.endswith(".weight")))
    else:
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("net.") and k.endswith(".weight")
                                      and "_u" not in k))
    f = make_model(hidden=hidden, depth=depth,
                   group_size=ckpt.get("group_size", 2),
                   activation=ckpt.get("activation", "groupsort"),
                   input_encoding=ckpt.get("input_encoding", "identity"),
                   multires=ckpt.get("multires", 6),
                   architecture=arch).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    f.eval()
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    print(f"[ckpt] arch={arch} hidden={hidden} depth={depth} "
          f"step={ckpt.get('step')}", flush=True)

    # ---- marching cubes in normalised model frame ------------------------
    # Extract ONLY within the dataset scene bounding box (bbox.txt), not the
    # full ±bound cube. This is the data-defined region the normalisation was
    # built from; it excludes unobserved subsurface / far-field where the SDF
    # is unconstrained (e.g. the underground slab a COLMAP-occupancy init
    # leaves below the ground). The official TnT toolbox is then run unmodified
    # — same as Neuralangelo/NeuS, which bound extraction to the foreground.
    bbox = np.loadtxt(scene_dir / "bbox.txt", dtype=np.float32)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale  = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    lo_n = (bbox[:3]  - center) / scale          # normalised scene-box min  (per axis)
    hi_n = (bbox[3:6] - center) / scale          # normalised scene-box max  (per axis)
    pad  = 0.10 * (hi_n - lo_n)                   # 10% margin so we never clip the surface
    lo_n = np.clip(lo_n - pad, -bound, bound)
    hi_n = np.clip(hi_n + pad, -bound, bound)
    gx = torch.linspace(float(lo_n[0]), float(hi_n[0]), mc_res, device=device)
    gy = torch.linspace(float(lo_n[1]), float(hi_n[1]), mc_res, device=device)
    gz = torch.linspace(float(lo_n[2]), float(hi_n[2]), mc_res, device=device)
    xs, ys, zs = torch.meshgrid(gx, gy, gz, indexing="ij")
    pts = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
    print(f"[mc] evaluating {pts.shape[0]:,} pts at res={mc_res}  "
          f"scene-box(norm) lo={lo_n} hi={hi_n}", flush=True)
    with torch.no_grad():
        vals = torch.cat([f(p) for p in pts.split(65536)]).reshape(
            mc_res, mc_res, mc_res).cpu().numpy()
    if vals.min() > 0 or vals.max() < 0:
        raise SystemExit(f"no zero crossing: f in [{vals.min():.4f}, {vals.max():.4f}]")
    verts, faces, _, _ = measure.marching_cubes(vals, level=0.0)
    # map MC index-space verts back to per-axis normalised coords
    span = (hi_n - lo_n).astype(np.float32)
    verts = lo_n.astype(np.float32) + verts.astype(np.float32) / (mc_res - 1) * span
    print(f"[mc] {len(verts):,} verts  {len(faces):,} faces", flush=True)

    # ---- un-normalise to TnT world frame ---------------------------------
    verts_w = verts.astype(np.float32) * scale + center
    print(f"[norm] center={center}  scale={scale:.4f}", flush=True)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts_w.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    return mesh


def _read_colmap_images_bin(path: Path) -> dict[int, np.ndarray]:
    """Read COLMAP sparse images.bin → {frame_idx: c2w (4x4)}.

    frame_idx is parsed from the image name `0_<seq>_<frame>.png` (same
    convention as the scene pose files), so it pairs 1-to-1 with the shipped
    COLMAP-SfM log indices. COLMAP stores world-to-camera (R,t); we return
    camera-to-world so trajectory_alignment (which uses pose[:3,3] = camera
    centre) sees the right point.
    """
    import struct
    out: dict[int, np.ndarray] = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", fh.read(4))[0]               # image_id
            qw, qx, qy, qz = struct.unpack("<dddd", fh.read(32))
            tx, ty, tz = struct.unpack("<ddd", fh.read(24))
            struct.unpack("<I", fh.read(4))[0]               # camera_id
            name = b""
            while True:
                ch = fh.read(1)
                if ch == b"\x00":
                    break
                name += ch
            npts = struct.unpack("<Q", fh.read(8))[0]
            fh.read(npts * 24)                               # skip 2D points
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)],
            ], dtype=np.float64)
            t = np.array([tx, ty, tz], dtype=np.float64)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3]  = -R.T @ t
            try:
                frame_idx = int(name.decode().split("_")[2].split(".")[0])
            except (IndexError, ValueError):
                continue
            out[frame_idx] = c2w
    return out


def build_trajectories(scene_dir: Path, gt_dir: Path, scene: str,
                       out_dir: Path, pose_source: str = "colmap-pose"):
    """Build paired camera trajectories for trajectory_alignment.

    pose_source selects which camera poses define the recon's frame:
      "colmap-pose"  — raw c2w pose files (`pose/0_*.txt`). Correct for the
                       current COLMAP-derived TnT scenes where checkpoint
                       extraction maps the SDF back through bbox.txt into the
                       same raw pose frame.
      "nsvf"         — legacy alias for "colmap-pose".
      "colmap-local" — the LOCAL COLMAP sparse model (`colmap/sparse/0/
                       images.bin`). Correct for a mesh built FROM that sparse
                       model (COLMAP fused.ply / poisson.ply), which lives in
                       the local-COLMAP frame. Use this when the local sparse
                       model is present and the mesh was built directly from it.

    Both are paired by frame index against the shipped COLMAP-SfM log and fed
    to the official trajectory_alignment (Umeyama similarity via RANSAC).

    Returns (traj_ours, traj_colmap_filtered) — equal-length, same order.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tnt_eval"))
    from trajectory_io import CameraPose, read_trajectory, write_trajectory

    if pose_source == "colmap-local":
        images_bin = scene_dir / "colmap" / "sparse" / "0" / "images.bin"
        if not images_bin.exists():
            raise FileNotFoundError(
                f"colmap-local frame needs {images_bin}; build the sparse model first")
        traj_ours = _read_colmap_images_bin(images_bin)
        print(f"[trajectory] pose_source=colmap-local  {images_bin}", flush=True)
    else:
        # The pose files are already in the raw COLMAP/TnT frame. The training
        # loader normalises them internally via bbox.txt; the eval mesh has
        # already been mapped back to this raw frame, so do not un-normalise
        # translations a second time.
        pose_files = sorted((scene_dir / "pose").glob("0_*.txt"))
        traj_ours = {}
        for pp in pose_files:
            # filename: 0_<seq>_<frame_idx>.txt   (e.g. 0_0000_00000235.txt)
            frame_idx = int(pp.stem.split("_")[2])
            c2w = np.loadtxt(pp, dtype=np.float64).reshape(4, 4)
            traj_ours[frame_idx] = c2w

    # --- official COLMAP-SfM trajectory (one block per frame, index = meta[0])
    gt_traj = read_trajectory(str(gt_dir / f"{scene}_COLMAP_SfM.log"))
    traj_colmap_by_idx = {}
    for cam in gt_traj:
        meta = list(cam.metadata)
        traj_colmap_by_idx[int(meta[0])] = cam.pose

    # --- pair by frame index, keep only common ones ------------------------
    common = sorted(set(traj_ours) & set(traj_colmap_by_idx))
    print(f"[trajectory] |ours|={len(traj_ours)}  |colmap|={len(gt_traj)}  "
          f"|paired|={len(common)}", flush=True)
    if len(common) < 50:
        raise SystemExit(f"too few paired frames ({len(common)}); "
                         "check pose naming vs COLMAP-SfM log indices")

    ours_paired   = [CameraPose([i, i, 0], traj_ours[i])           for i in common]
    colmap_paired = [CameraPose([i, i, 0], traj_colmap_by_idx[i])  for i in common]

    # write both for debugging / reproducibility
    write_trajectory(ours_paired,   str(out_dir / f"{scene}_ours.log"))
    write_trajectory(colmap_paired, str(out_dir / f"{scene}_colmap_paired.log"))
    print(f"[trajectory] wrote {scene}_ours.log + {scene}_colmap_paired.log",
          flush=True)
    return ours_paired, colmap_paired


def run_tnt_eval(rec_pcd_path: Path, scene_dir: Path, gt_dir: Path,
                 scene: str, out_dir: Path, plot_stretch: float = 5.0,
                 frame: str = "colmap-pose"):
    """Invoke the vendored official toolbox end-to-end.

    `frame` selects how to bring the recon into the GT-LiDAR frame:
      "colmap-pose" — recon is in the raw pose/*.txt frame (current COLMAP-
                      derived TnT scenes; e.g. our MC-extracted SDF mesh after
                      bbox un-normalisation). We need `trajectory_alignment`
                      to compute pose-frame -> GT similarity transform (no
                      `mapping_reference.txt` needed — sparse path is only
                      taken when len(traj_to_register) > 1600).
      "nsvf"        — legacy alias for "colmap-pose".
      "colmap-sfm"  — recon is already in the official COLMAP-SfM frame
                      (e.g. COLMAP fused.ply / poisson.ply built from
                      data/tnt/Barn/colmap/sparse/0). The `Barn_trans.txt`
                      similarity matrix takes that frame directly to the
                      GT-LiDAR frame; we apply it as-is and skip Umeyama.
    """
    import open3d as o3d

    # --- shim: legacy o3d.registration -> o3d.pipelines.registration ------
    if not hasattr(o3d, "registration"):
        sys.modules["open3d.registration"] = o3d.pipelines.registration
        o3d.registration = o3d.pipelines.registration  # type: ignore[attr-defined]

    # Open3D >=0.10 also dropped `max_validation` on RANSACConvergenceCriteria
    # (replaced by `confidence`). The vendored toolbox sets this attribute, so
    # we wrap the constructor to silently absorb assignments to the old name.
    _RANSAC = o3d.pipelines.registration.RANSACConvergenceCriteria
    if not hasattr(_RANSAC(), "max_validation"):
        class _RANSACCompat(_RANSAC):
            @property
            def max_validation(self):
                return None

            @max_validation.setter
            def max_validation(self, _value):
                pass

        o3d.pipelines.registration.RANSACConvergenceCriteria = _RANSACCompat
        o3d.registration.RANSACConvergenceCriteria = _RANSACCompat  # type: ignore[attr-defined]

    # Open3D >=0.10 also added a `checkers` positional arg to
    # registration_ransac_based_on_correspondence between `ransac_n` and
    # `criteria`. The vendored toolbox uses the old 7-positional signature
    # with criteria as arg 7. Re-route through a wrapper that inserts an
    # empty checkers list when the old call shape is detected.
    _orig_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence

    def _ransac_compat(*args, **kwargs):
        if len(args) == 7 and "criteria" not in kwargs and "checkers" not in kwargs:
            args = args[:6] + ([], args[6])
        return _orig_ransac(*args, **kwargs)

    o3d.pipelines.registration.registration_ransac_based_on_correspondence = _ransac_compat
    o3d.registration.registration_ransac_based_on_correspondence = _ransac_compat  # type: ignore[attr-defined]

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tnt_eval"))
    from config import scenes_tau_dict
    from registration import registration_vol_ds, trajectory_alignment
    from evaluation import EvaluateHisto
    from util import make_dir
    from plot import plot_graph

    if scene not in scenes_tau_dict:
        raise ValueError(f"scene {scene!r} not in {sorted(scenes_tau_dict)}")
    dTau = scenes_tau_dict[scene]

    gt_ply   = gt_dir / f"{scene}.ply"
    trans_p  = gt_dir / f"{scene}_trans.txt"
    crop_p   = gt_dir / f"{scene}.json"
    sfm_log  = gt_dir / f"{scene}_COLMAP_SfM.log"
    for p in (gt_ply, trans_p, crop_p, sfm_log):
        if not p.exists():
            raise FileNotFoundError(f"GT asset missing: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    make_dir(str(out_dir))

    print(f"=== TnT [{scene}]  tau={dTau} ===", flush=True)
    pcd    = o3d.io.read_point_cloud(str(rec_pcd_path))
    gt_pcd = o3d.io.read_point_cloud(str(gt_ply))
    print(f"  |rec|={len(pcd.points):,}  |gt|={len(gt_pcd.points):,}", flush=True)

    gt_trans = np.loadtxt(trans_p)
    assert gt_trans.shape == (4, 4)
    crop_vol = o3d.visualization.read_selection_polygon_volume(str(crop_p))

    # --- 1. initial transform: recon -> GT-LiDAR frame --------------------
    if frame == "colmap-sfm":
        # Recon is already in the (shipped) COLMAP-SfM frame; _trans.txt -> GT.
        traj_transform = gt_trans
        print(f"[frame=colmap-sfm] using _trans.txt directly\n{traj_transform}",
              flush=True)
    elif frame == "colmap-local":
        # Recon lives in the LOCAL COLMAP frame (mesh built from sparse/0).
        # Align local-COLMAP cameras -> shipped log -> GT via Umeyama.
        traj_ours, traj_colmap = build_trajectories(
            scene_dir, gt_dir, scene, out_dir, pose_source="colmap-local")
        traj_transform = trajectory_alignment(None, traj_ours, traj_colmap,
                                              gt_trans, scene)
        print(f"[frame=colmap-local] trajectory_transform\n{traj_transform}", flush=True)
    else:
        # Recon is in the raw pose/*.txt frame; trajectory_alignment finds the
        # pose-frame -> GT-LiDAR similarity via camera correspondences.
        traj_ours, traj_colmap = build_trajectories(
            scene_dir, gt_dir, scene, out_dir, pose_source="colmap-pose")
        traj_transform = trajectory_alignment(None, traj_ours, traj_colmap,
                                              gt_trans, scene)
        print(f"[frame={frame}] trajectory_transform\n{traj_transform}", flush=True)

    # --- 2. ICP refinement (3 steps; exactly as run.py) -------------------
    r2 = registration_vol_ds(pcd, gt_pcd, traj_transform,    crop_vol, dTau,       dTau * 80, 20)
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, crop_vol, dTau / 2.0, dTau * 20, 20)
    r  = registration_vol_ds(pcd, gt_pcd, r3.transformation, crop_vol, dTau / 4.0, dTau *  2, 20)

    precision, recall, fscore, es, cs, et, ct = EvaluateHisto(
        pcd, gt_pcd, r.transformation, crop_vol,
        dTau / 2.0, dTau, str(out_dir), plot_stretch, scene,
    )
    print(f"\n=== {scene} @ tau={dTau} ===")
    print(f"  precision = {precision:.4f}")
    print(f"  recall    = {recall:.4f}")
    print(f"  F-score   = {fscore:.4f}")
    plot_graph(scene, fscore, dTau, es, cs, et, ct, plot_stretch, str(out_dir))

    import json
    (out_dir / "fscore.json").write_text(json.dumps({
        "scene": scene, "tau": dTau,
        "precision": precision, "recall": recall, "fscore": fscore,
        "transform": r.transformation.tolist(),
    }, indent=2))
    # --- optionally generate DTU-style PNG diagnostics (uses existing script)
    try:
        # tnt_error_pngs.py expects <scene>.precision.ply and <scene>.recall.ply
        tnt_png = Path(__file__).resolve().parent / "tnt_error_pngs.py"
        prec_ply = out_dir / f"{scene}.precision.ply"
        rec_ply = out_dir / f"{scene}.recall.ply"
        if tnt_png.exists() and prec_ply.exists() and rec_ply.exists():
            print(f"[png] running {tnt_png} -> generate DTU-style PNGs", flush=True)
            import subprocess
            subprocess.run([sys.executable, str(tnt_png), str(out_dir), "--scene", scene,
                            "--tau-mm", str(float(dTau))], check=False)
    except Exception as _e:  # don't fail the main eval if PNG generation fails
        print(f"[png] skipped DTU-style PNG generation: {_e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt", type=Path,
                     help="Lipschitz-SDF checkpoint .pt; we MC-extract + sample inside the job")
    grp.add_argument("--mesh", type=Path,
                     help="pre-extracted world-frame mesh PLY (skip MC)")
    ap.add_argument("--scene-dir", type=Path, required=True,
                    help="TnT scene dir, e.g. data/tnt/Barn (needs bbox.txt)")
    ap.add_argument("--gt-dir",    type=Path, required=True,
                    help="dir with <scene>.ply / _trans.txt / .json")
    ap.add_argument("--scene",     required=True,
                    help="scene name, e.g. Barn  (must match GT filenames)")
    ap.add_argument("--out",       type=Path, required=True)
    ap.add_argument("--mc-res",    type=int,   default=512)
    ap.add_argument("--bound",     type=float, default=1.5)
    ap.add_argument("--n-samples", type=int,   default=2_000_000)
    ap.add_argument("--frame", choices=["colmap-pose", "nsvf", "colmap-sfm", "colmap-local"],
                    default="colmap-pose",
                    help="coordinate frame of the recon: 'colmap-pose' for an "
                         "SDF checkpoint / mesh in the raw pose/*.txt frame; "
                         "'nsvf' is a legacy alias; 'colmap-local' for a COLMAP "
                         "MVS recon (fused.ply / poisson.ply) built from the "
                         "LOCAL sparse/0 model (aligns via local cameras); "
                         "'colmap-sfm' only if the recon is already in the "
                         "shipped COLMAP-SfM frame")
    args = ap.parse_args()

    import open3d as o3d
    args.out.mkdir(parents=True, exist_ok=True)

    if args.ckpt is not None:
        mesh = extract_mesh_from_ckpt(args.ckpt, args.scene_dir,
                                      args.mc_res, args.bound)
        mesh_p = args.out / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_p), mesh)
        print(f"[write] {mesh_p}", flush=True)
        pcd = mesh.sample_points_uniformly(number_of_points=args.n_samples)
        pcd_p = args.out / "points.ply"
        o3d.io.write_point_cloud(str(pcd_p), pcd)
        print(f"[write] {pcd_p}  ({args.n_samples:,} pts)", flush=True)
    else:
        if not args.mesh.exists():
            sys.exit(f"--mesh not found: {args.mesh}")
        mesh = o3d.io.read_triangle_mesh(str(args.mesh))
        if len(mesh.triangles) == 0:
            # input is already a point cloud (e.g. COLMAP fused.ply) — use as-is
            print(f"[input] {args.mesh} has no triangles; treating as point cloud",
                  flush=True)
            pcd_p = args.mesh
        else:
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=args.n_samples)
            pcd_p = args.out / "points.ply"
            o3d.io.write_point_cloud(str(pcd_p), pcd)
            print(f"[write] {pcd_p}  ({args.n_samples:,} pts)", flush=True)

    run_tnt_eval(pcd_p, args.scene_dir, args.gt_dir, args.scene, args.out,
                 frame=args.frame)


if __name__ == "__main__":
    main()
