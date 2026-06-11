"""Fuse per-view sphere-traced screen meshes into one PLY.

Each view's mesh is one-sided (the face the camera sees). Concatenated, they
cover the whole surface. To control multi-view overlap we drop, per view, the
faces whose vertices are too grazing to that view's camera — i.e. each view
only "claims" the surface region it sees nearly head-on. No spatial dedup, no
remeshing, no smoothing: the union of the per-view triangulations.

Output PLY is in DTU world frame (input PLYs are too).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_ply_with_normals(path: Path):
    """Minimal binary-LE PLY reader for files we wrote in analysis/sphere_traced_screen_mesh.py.
    Returns (verts (V,3), normals (V,3), faces (F,3))."""
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = fh.readline()
            if not line:
                raise ValueError(f"truncated header in {path}")
            header += line
        body = fh.read()
    h = header.decode("ascii").splitlines()
    nV = int(next(L for L in h if L.startswith("element vertex")).split()[-1])
    nF = int(next(L for L in h if L.startswith("element face")).split()[-1])
    vsz = 6 * 4                     # x y z nx ny nz, float32
    vbuf = np.frombuffer(body[:nV * vsz], dtype="<f4").reshape(nV, 6)
    verts = vbuf[:, :3].copy()
    normals = vbuf[:, 3:].copy()
    face_dtype = np.dtype([("c", "u1"), ("v", "<i4", 3)])
    fb = np.frombuffer(body[nV * vsz:nV * vsz + nF * face_dtype.itemsize], dtype=face_dtype)
    faces = fb["v"].copy()
    return verts, normals, faces


def write_ply(path: Path, verts, normals, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    nV, nF = len(verts), len(faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {nV}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {nF}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vbuf = np.concatenate([verts.astype("<f4"), normals.astype("<f4")], axis=1).tobytes()
    fb = np.empty(nF, dtype=[("c", "u1"), ("v", "<i4", 3)])
    fb["c"] = 3
    fb["v"] = faces.astype("<i4")
    with open(path, "wb") as fh:
        fh.write(header); fh.write(vbuf); fh.write(fb.tobytes())
    print(f"saved {path}  ({nV:,} verts, {nF:,} faces)", flush=True)


def cam_center_world(cam_dict, view_idx):
    """Recover camera centre in DTU world frame from world_mat_i."""
    P = cam_dict[f"world_mat_{view_idx}"][:3, :4].astype(np.float64)
    M = P[:, :3]
    # RQ via flipped QR (same as tools/render_blender.py)
    Pf = np.flipud(np.eye(3))
    Q_, R_ = np.linalg.qr((Pf @ M).T)
    R = Pf @ R_.T @ Pf
    K = Pf @ Q_.T  # not used
    sign = np.sign(np.diag(K))
    sign[sign == 0] = 1.0
    T = np.diag(sign)
    K = K @ T
    R = T @ R
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1.0
        R[2, :] *= -1.0
    K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    return -R.T @ t  # DTU world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_dir", type=Path,
                    help="directory containing sphere_traced_screen_view{NN}.ply")
    ap.add_argument("--scene", type=Path, required=True,
                    help="DTU scene dir for cameras_sphere.npz")
    ap.add_argument("--views", type=str, default="all",
                    help="comma-list or 'all'")
    ap.add_argument("--n-views-fallback", type=int, default=64)
    ap.add_argument("--cos-thresh", type=float, default=0.3,
                    help="per view, drop faces with all-corner |cos(n, -view)| < this")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cam_path = args.scene / "cameras_sphere.npz"
    if not cam_path.exists():
        cam_path = args.scene / "cameras.npz"
    cam_dict = np.load(cam_path)

    if args.views.strip() == "all":
        view_list = list(range(args.n_views_fallback))
    else:
        view_list = [int(s) for s in args.views.split(",")]

    all_v, all_n, all_f = [], [], []
    voff = 0
    for vi in view_list:
        ply = args.ply_dir / f"sphere_traced_screen_view{vi:02d}.ply"
        if not ply.exists():
            print(f"  view {vi}: missing {ply.name} — skip", flush=True)
            continue
        v, n, f = read_ply_with_normals(ply)
        cam_c = cam_center_world(cam_dict, vi)
        view_dir = v - cam_c            # vertex-to-camera vector reversed = "outward"
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True).clip(min=1e-9)
        # cos(outward-normal, vertex→pointing-away-from-cam) = -n·view_dir
        cos_nv = -(n * view_dir).sum(-1)
        keep_v = cos_nv >= args.cos_thresh
        # face passes iff all 3 corners pass
        keep_f = keep_v[f[:, 0]] & keep_v[f[:, 1]] & keep_v[f[:, 2]]
        # don't compact vertices — keep all so face indices stay stable. Renderers
        # don't care about unreferenced vertices and skipping compaction keeps this
        # step branch-free; the final .ply size is dominated by faces anyway.
        fk = f[keep_f] + voff
        print(f"  view {vi:>2d}: kept {keep_f.sum():>8,} / {len(f):>8,} faces"
              f"  ({100*keep_f.sum()/max(1,len(f)):4.1f}%)", flush=True)
        all_v.append(v); all_n.append(n); all_f.append(fk)
        voff += len(v)

    if not all_f:
        raise SystemExit("no PLYs found")
    verts = np.concatenate(all_v)
    normals = np.concatenate(all_n)
    faces = np.concatenate(all_f)
    print(f"\nfused: {len(verts):,} verts  {len(faces):,} faces  from {len(all_f)} views",
          flush=True)
    write_ply(args.out, verts, normals, faces)


if __name__ == "__main__":
    main()
