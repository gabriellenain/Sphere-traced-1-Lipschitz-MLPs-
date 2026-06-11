#!/usr/bin/env python3
"""Rigorous geometry comparison between two f_theta models (1-Lip or NeuS mesh).

Each model slot accepts *either*:
  --a-ckpt  path/to/checkpoint.pt   (FTheta / RegularMLP saved by this repo)
  --a-mesh  path/to/mesh.ply        (any triangulated mesh, e.g. exported from NeuS)

Metrics
-------
  chamfer_L1          bidirectional mean nearest-neighbour distance (lower is better)
  precision           pred to ref mean NN dist
  completeness        ref to pred mean NN dist
  hausdorff           max of all NN distances (lower is better)
  f_score@T           harmonic mean of precision/completeness at threshold T
  normal_consistency  mean |cos theta| between normals at NN pairs (higher is better)

Usage
-----
  python compare_geometry.py \\
      --a-ckpt outputs/run_1lip/checkpoint.pt \\
      --b-mesh outputs/neus/mesh.ply \\
      --res 256 --bound 1.5 --n-pts 100000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def _extract_mesh_from_model(ckpt_path, bound, res, device):
    from skimage.measure import marching_cubes
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["f"]
    architecture = ckpt.get("architecture", "cpl")
    from lip_tracer.model import make_model
    hidden     = ckpt.get("hidden",         256)
    depth      = ckpt.get("depth",          8)
    group_size = ckpt.get("group_size",     2)
    activation = ckpt.get("activation",     "groupsort")
    input_enc  = ckpt.get("input_encoding", "identity")
    multires   = ckpt.get("multires",       6)
    for k, v in state.items():
        if "weight" in k and v.ndim >= 2:
            hidden = v.shape[-1]
            break
    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_enc,
                   multires=multires, architecture=architecture)
    f.load_state_dict(state, strict=False)
    f = f.to(device).eval()
    vox  = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        vals = torch.cat([f(grid[i:i+4096]) for i in range(0, len(grid), 4096)])
    vol = vals.reshape(res, res, res).cpu().numpy()
    if vol.min() > 0 or vol.max() < 0:
        raise ValueError(f"No zero-crossing in SDF grid (min={vol.min():.3f}, max={vol.max():.3f}). Try adjusting --bound.")
    spacing = 2 * bound / (res - 1)
    verts, faces, normals, _ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3, gradient_direction="ascent")
    return (verts - bound).astype("float32"), faces.astype("int32"), normals.astype("float32")


def _load_mesh(mesh_path):
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh found in {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    return (np.asarray(mesh.vertices, dtype="float32"),
            np.asarray(mesh.faces, dtype="int32"),
            np.asarray(mesh.vertex_normals, dtype="float32"))


def _sample_surface(verts, faces, normals, n):
    import trimesh
    mesh    = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, fi = trimesh.sample.sample_surface(mesh, n)
    pts     = pts.astype("float32")
    v0 = verts[faces[fi, 0]]; v1 = verts[faces[fi, 1]]; v2 = verts[faces[fi, 2]]
    n0 = normals[faces[fi, 0]]; n1 = normals[faces[fi, 1]]; n2 = normals[faces[fi, 2]]
    def _area(a, b, c):
        return np.linalg.norm(np.cross(b - a, c - a), axis=-1, keepdims=True)
    w0 = _area(pts, v1, v2); w1 = _area(v0, pts, v2); w2 = _area(v0, v1, pts)
    w   = (w0 + w1 + w2).clip(min=1e-12)
    nrm = (w0 * n0 + w1 * n1 + w2 * n2) / w
    nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True).clip(min=1e-8)
    return pts, nrm.astype("float32")


def _nn(a, b):
    from scipy.spatial import cKDTree
    dists, idxs = cKDTree(b).query(a, k=1, workers=-1)
    return dists.astype("float32"), idxs.astype("int32")


def compute_metrics(pts_a, nrm_a, pts_b, nrm_b, thresholds):
    d_ab, idx_ab = _nn(pts_a, pts_b)
    d_ba, idx_ba = _nn(pts_b, pts_a)
    precision    = float(d_ab.mean())
    completeness = float(d_ba.mean())
    chamfer_l1   = 0.5 * (precision + completeness)
    hausdorff    = float(max(d_ab.max(), d_ba.max()))
    nc = 0.5 * (np.abs((nrm_a * nrm_b[idx_ab]).sum(-1)).mean()
              + np.abs((nrm_b * nrm_a[idx_ba]).sum(-1)).mean())
    out = {"precision": precision, "completeness": completeness,
           "chamfer_L1": chamfer_l1, "hausdorff": hausdorff,
           "normal_consistency": float(nc)}
    for t in thresholds:
        p = float((d_ab < t).mean()); c = float((d_ba < t).mean())
        f = 2 * p * c / (p + c) if (p + c) > 0 else 0.0
        out[f"f_score@{t:.4g}"] = f
        out[f"precision@{t:.4g}"] = p
        out[f"completeness@{t:.4g}"] = c
    return out


def _print_results(results, label_a, label_b, thresholds):
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  Geometry comparison")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(sep)
    print(f"  {'Chamfer-L1':<32s}  {results['chamfer_L1']:.6f}")
    print(f"  {'  precision  (A->B)':<32s}  {results['precision']:.6f}")
    print(f"  {'  completeness (B->A)':<32s}  {results['completeness']:.6f}")
    print(f"  {'Hausdorff':<32s}  {results['hausdorff']:.6f}")
    print(f"  {'Normal consistency':<32s}  {results['normal_consistency']:.4f}  (1=perfect)")
    for t in thresholds:
        p = results[f"precision@{t:.4g}"]; c = results[f"completeness@{t:.4g}"]
        f = results[f"f_score@{t:.4g}"]
        print(f"  {'F-score @ ' + str(t):<32s}  {f:.4f}  (P={p:.4f}  R={c:.4f})")
    print(sep + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ga = ap.add_mutually_exclusive_group(required=True)
    ga.add_argument("--a-ckpt", type=Path, metavar="PT")
    ga.add_argument("--a-mesh", type=Path, metavar="PLY")
    gb = ap.add_mutually_exclusive_group(required=True)
    gb.add_argument("--b-ckpt", type=Path, metavar="PT")
    gb.add_argument("--b-mesh", type=Path, metavar="PLY")
    ap.add_argument("--res",        type=int,   default=256)
    ap.add_argument("--bound",      type=float, default=1.5)
    ap.add_argument("--n-pts",      type=int,   default=100_000)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.005, 0.010, 0.025])
    ap.add_argument("--out",        type=Path,  default=None)
    ap.add_argument("--save-meshes", action="store_true")
    ap.add_argument("--device",     type=str,   default="auto")
    ap.add_argument("--seed",       type=int,   default=0)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    np.random.seed(args.seed)

    t0 = time.time()
    if args.a_ckpt is not None:
        label_a = str(args.a_ckpt)
        print(f"[A] extracting mesh from checkpoint {args.a_ckpt}  (res={args.res}, bound={args.bound})")
        verts_a, faces_a, normals_a = _extract_mesh_from_model(args.a_ckpt, args.bound, args.res, device)
    else:
        label_a = str(args.a_mesh)
        print(f"[A] loading mesh {args.a_mesh}")
        verts_a, faces_a, normals_a = _load_mesh(args.a_mesh)
    print(f"    {len(verts_a):,} verts, {len(faces_a):,} faces  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    if args.b_ckpt is not None:
        label_b = str(args.b_ckpt)
        print(f"[B] extracting mesh from checkpoint {args.b_ckpt}  (res={args.res}, bound={args.bound})")
        verts_b, faces_b, normals_b = _extract_mesh_from_model(args.b_ckpt, args.bound, args.res, device)
    else:
        label_b = str(args.b_mesh)
        print(f"[B] loading mesh {args.b_mesh}")
        verts_b, faces_b, normals_b = _load_mesh(args.b_mesh)
    print(f"    {len(verts_b):,} verts, {len(faces_b):,} faces  ({time.time()-t0:.1f}s)")

    if args.save_meshes and args.out is not None:
        import trimesh
        args.out.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=verts_a, faces=faces_a).export(args.out / "mesh_a.ply")
        trimesh.Trimesh(vertices=verts_b, faces=faces_b).export(args.out / "mesh_b.ply")
        print(f"saved meshes -> {args.out}/mesh_a.ply, mesh_b.ply")

    print(f"sampling {args.n_pts:,} pts per mesh ...")
    pts_a, nrm_a = _sample_surface(verts_a, faces_a, normals_a, args.n_pts)
    pts_b, nrm_b = _sample_surface(verts_b, faces_b, normals_b, args.n_pts)

    print("computing metrics ...")
    t0 = time.time()
    results = compute_metrics(pts_a, nrm_a, pts_b, nrm_b, args.thresholds)
    print(f"  done in {time.time()-t0:.1f}s")
    _print_results(results, label_a, label_b, args.thresholds)

    if args.out is not None:
        import json
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / "geo_comparison.json"
        out_file.write_text(json.dumps({"model_a": label_a, "model_b": label_b,
            "res": args.res, "bound": args.bound, "n_pts": args.n_pts,
            "thresholds": args.thresholds, "metrics": results}, indent=2))
        print(f"results saved -> {out_file}")


if __name__ == "__main__":
    main()
