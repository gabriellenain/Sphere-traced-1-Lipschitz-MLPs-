"""End-to-end COLMAP-sparse → IGR SDF init → PNG for an NSVF-format scene.

Steps:
  1. Convert NSVF intrinsics/poses to a COLMAP input model (cameras.txt,
     images.txt, points3D.txt empty), using the image IDs that COLMAP's
     feature_extractor assigns to each filename.
  2. Run COLMAP feature_extractor + sequential_matcher + point_triangulator.
  3. Read the resulting points3D.bin, normalise into the scene's bbox frame.
  4. IGR-style pretrain a 1-Lipschitz MLP: |f(p)|^2 on points + SAL off-surface
     penalty on random ambient samples.
  5. March-cubes the fitted SDF and render a single normal-map PNG from view 0.

Run via init_from_colmap.slurm (needs a GPU node for COLMAP SIFT + IGR fit).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lip_tracer.model import make_model  # noqa: E402
from lip_tracer.data import _find_epfl_strecha_urd, load_views  # noqa: E402


# ---------- COLMAP I/O helpers ----------------------------------------------

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (qw, qx, qy, qz). Matches COLMAP convention."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _read_points3d_bin(path: Path) -> np.ndarray:
    """Return (N, 3) float32 array of XYZ from a COLMAP points3D.bin."""
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                              # point3D_id
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)                              # rgb (uint8 x3)
            f.read(8)                              # error (double)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)                  # (image_id, point2D_idx) pairs
            pts.append(xyz)
    return np.asarray(pts, dtype=np.float32)


def _read_points3d_any(path: Path) -> np.ndarray:
    """Read XYZ from a COLMAP points3D.bin or points3D.txt."""
    if path.suffix == ".txt":
        pts = [[float(c) for c in ln.split()[1:4]]
               for ln in path.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
        return np.asarray(pts, dtype=np.float32)
    return _read_points3d_bin(path)


def _db_image_ids(db_path: Path) -> dict[str, int]:
    """Return {image_name: image_id} from a COLMAP database."""
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT image_id, name FROM images").fetchall()
    con.close()
    return {name: int(iid) for iid, name in rows}


# ---------- NSVF -> COLMAP input model --------------------------------------

def _write_colmap_input_model(scene: Path, model_dir: Path,
                              db_path: Path, W: int, H: int) -> int:
    """Write cameras.txt / images.txt / points3D.txt from NSVF data, using
    image IDs the feature_extractor already assigned in the database."""
    model_dir.mkdir(parents=True, exist_ok=True)

    K = np.loadtxt(scene / "intrinsics.txt", dtype=np.float64)[:3, :3]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    name_to_id = _db_image_ids(db_path)

    with open(model_dir / "cameras.txt", "w") as fh:
        fh.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        fh.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    n = 0
    with open(model_dir / "images.txt", "w") as fh:
        fh.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for pp in sorted((scene / "pose").glob("0_*.txt")):
            name = pp.stem + ".png"
            if name not in name_to_id:
                continue
            c2w = np.loadtxt(pp, dtype=np.float64).reshape(4, 4)
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            qw, qx, qy, qz = _rot_to_quat(R_w2c)
            iid = name_to_id[name]
            fh.write(f"{iid} {qw} {qx} {qy} {qz} "
                     f"{t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {name}\n\n")
            n += 1

    (model_dir / "points3D.txt").write_text("")
    return n


def _write_epfl_colmap_input_model(scene: Path, model_dir: Path,
                                   db_path: Path) -> int:
    """Write COLMAP cameras/images from EPFL fixed cameras in normalized frame."""
    urd = _find_epfl_strecha_urd(scene)
    if urd is None:
        raise FileNotFoundError(f"no EPFL *_dense/urd directory found under {scene}")
    image_paths = sorted(p for p in urd.glob("*.png") if not p.name.startswith("._"))
    if not image_paths:
        raise FileNotFoundError(f"no EPFL images under {urd}")
    views = load_views(scene, down=1)
    c2ws = views["c2w"].numpy().astype(np.float64)
    Ks = views["K"].numpy().astype(np.float64)
    name_to_id = _db_image_ids(db_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cameras.txt", "w") as fh:
        fh.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for i, ip in enumerate(image_paths):
            W, H = Image.open(ip).size
            K = Ks[i]
            fh.write(f"{i + 1} PINHOLE {W} {H} "
                     f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")

    n = 0
    with open(model_dir / "images.txt", "w") as fh:
        fh.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for i, ip in enumerate(image_paths):
            if ip.name not in name_to_id:
                continue
            c2w = c2ws[i]
            R_w2c = c2w[:3, :3].T
            t_w2c = -R_w2c @ c2w[:3, 3]
            qw, qx, qy, qz = _rot_to_quat(R_w2c)
            iid = name_to_id[ip.name]
            fh.write(f"{iid} {qw} {qx} {qy} {qz} "
                     f"{t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} {ip.name}\n\n")
            n += 1

    (model_dir / "points3D.txt").write_text("")
    return n


# ---------- COLMAP pipeline -------------------------------------------------

def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  -> {time.time() - t0:.1f}s", flush=True)


def run_colmap_triangulation(scene: Path, work: Path) -> Path:
    """Run COLMAP and return path to points3D.bin."""
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    out_model = sparse / "0"
    out_model.mkdir(exist_ok=True)
    if (out_model / "points3D.bin").exists():
        print(f"  [colmap] reusing {out_model/'points3D.bin'}")
        return out_model / "points3D.bin"

    is_epfl = _find_epfl_strecha_urd(scene) is not None
    img_dir = _find_epfl_strecha_urd(scene) if is_epfl else scene / "rgb"
    if img_dir is None:
        raise FileNotFoundError(f"no image directory found for {scene}")
    if not is_epfl:
        H, W = 1080, 1920  # fixed for NSVF-T&T (we don't resize before COLMAP)
        K = np.loadtxt(scene / "intrinsics.txt", dtype=np.float64)[:3, :3]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    if not db.exists():
        if is_epfl:
            _run([
                "colmap", "feature_extractor",
                "--database_path", str(db),
                "--image_path", str(img_dir),
                "--ImageReader.camera_model", "PINHOLE",
                "--SiftExtraction.max_image_size", "3200",
            ])
            _run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(db),
            ])
        else:
            _run([
                "colmap", "feature_extractor",
                "--database_path", str(db),
                "--image_path", str(img_dir),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "PINHOLE",
                "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
            ])
            _run([
                "colmap", "sequential_matcher",
                "--database_path", str(db),
                "--SequentialMatching.overlap", "15",
            ])

    input_model = work / "input_model"
    if is_epfl:
        n = _write_epfl_colmap_input_model(scene, input_model, db)
    else:
        n = _write_colmap_input_model(scene, input_model, db, W, H)
    print(f"  [colmap] wrote {n} images.txt entries")

    _run([
        "colmap", "point_triangulator",
        "--database_path", str(db),
        "--image_path", str(img_dir),
        "--input_path", str(input_model),
        "--output_path", str(out_model),
    ])
    return out_model / "points3D.bin"


# ---------- IGR fit ---------------------------------------------------------

def igr_pretrain(points_unit: torch.Tensor, *, hidden: int = 256, depth: int = 8,
                 steps: int = 5000, lr: float = 5e-4, batch: int = 16384,
                 alpha: float = 100.0, lam_off: float = 0.1,
                 device: str = "cuda") -> torch.nn.Module:
    f = make_model(hidden=hidden, depth=depth, group_size=2,
                   activation="groupsort", input_encoding="pe", multires=6,
                   architecture="cpl").to(device)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    pts = points_unit.to(device)
    n_pts = pts.shape[0]
    for s in range(steps):
        idx = torch.randint(0, n_pts, (min(batch, n_pts),), device=device)
        p = pts[idx]
        q = (2 * torch.rand(batch, 3, device=device) - 1) * 1.1
        fp = f.sdf(p)
        fq = f.sdf(q)
        loss_surface = (fp ** 2).mean()
        loss_off     = torch.exp(-alpha * fq.abs()).mean()
        loss = loss_surface + lam_off * loss_off
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 200 == 0 or s == steps - 1:
            with torch.no_grad():
                mae = fp.abs().mean().item()
            print(f"  igr  step {s:5d}  loss={loss.item():.5f}  "
                  f"surf={loss_surface.item():.5f}  "
                  f"off={loss_off.item():.5f}  |f(p)|={mae:.4f}", flush=True)
    return f


# ---------- mesh + PNG -----------------------------------------------------

def render_init_png(f, scene: Path, out_png: Path, *,
                    bound: float = 1.0, mc_res: int = 256,
                    device: str = "cuda", view_idx: int = 0) -> None:
    import imageio.v2 as imageio
    import trimesh  # noqa: F401  (used inside render_paper_marching)
    from render_paper_marching import (_mesh_from_volume, _make_intersector,
                                       render_normals_only)

    vox = torch.linspace(-bound, bound, mc_res)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"),
                       dim=-1).reshape(-1, 3).to(device)
    vals = []
    with torch.no_grad():
        for i in range(0, grid.shape[0], 65536):
            vals.append(f.sdf(grid[i:i+65536]).cpu().numpy())
    vol = np.concatenate(vals).reshape(mc_res, mc_res, mc_res)
    print(f"  [render] sdf grid: min={vol.min():.3f} max={vol.max():.3f} "
          f"mid={(vol < 0).mean():.1%} inside")

    mesh = _mesh_from_volume(vol, bound=bound, res=mc_res, level=0.0)
    print(f"  [render] mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    intersector = _make_intersector(mesh)

    pose_paths = sorted((scene / "pose").glob("0_*.txt"))
    c2w = np.loadtxt(pose_paths[view_idx], dtype=np.float32).reshape(4, 4)
    bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float32)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale  = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    c2w[:3, 3] = (c2w[:3, 3] - center) / scale
    K = np.loadtxt(scene / "intrinsics.txt", dtype=np.float32)[:3, :3]
    H, W = 1080, 1920
    down = 2
    K2 = K.copy(); K2[0] /= down; K2[1] /= down
    H2, W2 = H // down, W // down

    img = render_normals_only(mesh, intersector, c2w, K2, H2, W2, ss=1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, (img * 255).astype(np.uint8))
    print(f"  [render] -> {out_png}")


# ---------- main ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None,
                    help="COLMAP workdir (default: <scene>/colmap)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--colmap-points", type=Path, default=None,
                    help="use this EXISTING colmap points3D.bin/.txt (already in "
                         "the scene's pose world frame) instead of re-triangulating. "
                         "Required for migrated TnT scenes whose rgb/ are symlinks "
                         "into a raw colmap dir (feature_extractor can't re-register "
                         "them). The raw sparse model is the source of truth.")
    ap.add_argument("--no-igr", action="store_true",
                    help="stop after triangulating + saving sparse_sfm_points.txt "
                         "(skip the IGR pretrain + render); use to produce the SfM "
                         "ROI cloud the MVSFormer sphere-carve consumes")
    ap.add_argument("--save-sfm-points", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="write bbox-normalised inside-cube points to "
                         "<scene>/sparse_sfm_points.txt (default on)")
    args = ap.parse_args()

    scene = args.scene.resolve()
    work  = (args.work or scene / "colmap").resolve()

    if args.colmap_points is not None:
        print(f"  [pts] reading existing colmap points: {args.colmap_points}")
        p_bin = args.colmap_points.resolve()
        pts_world = _read_points3d_any(p_bin)
    else:
        p_bin = run_colmap_triangulation(scene, work)
        pts_world = _read_points3d_any(p_bin)
    print(f"  [pts] {pts_world.shape[0]} sparse points "
          f"(x={pts_world[:,0].min():.2f}..{pts_world[:,0].max():.2f})")

    # overlay PNG so we can eyeball the COLMAP quality before trusting IGR
    overlay_out = args.out_png.with_name("overlay.png")
    try:
        subprocess.run([
            sys.executable, str(ROOT / "tools" / "overlay_colmap_points.py"),
            "--scene", str(scene), "--points", str(p_bin),
            "--out",   str(overlay_out),
        ], check=True)
    except Exception as e:
        print(f"  [overlay] skipped: {e}")

    if _find_epfl_strecha_urd(scene) is not None:
        pts_unit = pts_world
        inside = np.all(np.abs(pts_unit) < 1.5, axis=1)
        print(f"  [pts] {inside.sum()}/{len(pts_unit)} inside ±1.5 EPFL-normalised cube")
    else:
        bbox = np.loadtxt(scene / "bbox.txt", dtype=np.float32)
        center = 0.5 * (bbox[:3] + bbox[3:6])
        scale  = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
        pts_unit = (pts_world - center) / scale
        inside = np.all(np.abs(pts_unit) < 1.1, axis=1)
        print(f"  [pts] {inside.sum()}/{len(pts_unit)} inside ±1.1 bbox-normalised cube")
    pts_unit = pts_unit[inside]

    if args.save_sfm_points:
        sfm_out = scene / "sparse_sfm_points.txt"
        np.savetxt(sfm_out, pts_unit, fmt="%.6f")
        print(f"  [sfm] saved {len(pts_unit)} bbox-normalised points -> {sfm_out}")

    if args.no_igr:
        print("  [igr] skipped (--no-igr); SfM ROI cloud is ready for the carve")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = igr_pretrain(torch.from_numpy(pts_unit), steps=args.steps, device=device)

    if args.ckpt is not None:
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"f": f.state_dict()}, args.ckpt)
        print(f"  [ckpt] -> {args.ckpt}")

    render_init_png(f, scene, args.out_png, device=device)


if __name__ == "__main__":
    main()
