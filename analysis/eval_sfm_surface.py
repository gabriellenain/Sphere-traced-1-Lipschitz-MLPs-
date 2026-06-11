#!/usr/bin/env python3
"""COLMAP point-to-SDF-surface distance for a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lip_tracer.data import load_colmap_points
from lip_tracer.model import make_model


def load_model(pt: Path, device: str, architecture_override: str | None = None):
    ckpt = torch.load(pt, map_location="cpu")
    state = ckpt["f"]
    architecture = architecture_override or ckpt.get("architecture", "cpl")
    if architecture == "neus":
        hidden = ckpt.get("hidden", state["layers.0.weight"].shape[0])
        depth = ckpt.get("depth", sum(1 for k in state if k.startswith("layers.") and k.endswith(".weight")))
    else:
        hidden = ckpt.get("hidden", state["head_weight"].shape[0])
        depth = ckpt.get(
            "depth",
            sum(1 for k in state if k.startswith("net.") and k.endswith(".weight") and "_u" not in k),
        )
    f = make_model(
        hidden=hidden,
        depth=depth,
        group_size=ckpt.get("group_size", 2),
        activation=ckpt.get("activation", "groupsort"),
        input_encoding=ckpt.get("input_encoding", "identity"),
        multires=ckpt.get("multires", 6),
        architecture=architecture,
    ).to(device)
    f.load_state_dict(state, strict=False)
    f.eval()
    return f, architecture, hidden, depth


def extract_surface(
    f,
    bound: float,
    res: int,
    device: str,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    from skimage.measure import marching_cubes

    vox = torch.linspace(-bound, bound, res, device=device)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    vals = []
    with torch.no_grad():
        for i in range(0, len(grid), chunk):
            vals.append(f(grid[i:i + chunk]).detach().cpu())
    vol = torch.cat(vals).reshape(res, res, res).numpy()
    if vol.min() > 0 or vol.max() < 0:
        return None
    spacing = 2 * bound / (res - 1)
    verts, faces, normals, _ = marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
    return (verts - bound).astype(np.float32), faces.astype(np.int64), normals.astype(np.float32)


def mesh_area_stats(verts: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    tri = verts[faces]
    face_area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    total_area = float(face_area.sum())
    if len(faces) == 0:
        return {"total": 0.0, "largest": 0.0, "largest_pct": 0.0, "components": 0.0}

    parent = np.arange(len(verts), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(int(a), int(b))
        union(int(a), int(c))

    roots = np.fromiter((find(int(v)) for v in faces[:, 0]), dtype=np.int64, count=len(faces))
    _, inv = np.unique(roots, return_inverse=True)
    comp_area = np.bincount(inv, weights=face_area)
    largest_area = float(comp_area.max()) if len(comp_area) else 0.0
    return {
        "total": total_area,
        "largest": largest_area,
        "largest_pct": float(100.0 * largest_area / total_area) if total_area > 0 else 0.0,
        "components": float(len(comp_area)),
    }


def nearest_stats(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(target).query(query, k=1, workers=-1)
    return {
        "mean": float(dist.mean()),
        "p50": float(np.percentile(dist, 50)),
        "p90": float(np.percentile(dist, 90)),
        "p99": float(np.percentile(dist, 99)),
        "max": float(dist.max()),
        "pct_lt_001": float((dist < 0.01).mean() * 100.0),
        "pct_lt_002": float((dist < 0.02).mean() * 100.0),
        "pct_gt_005": float((dist > 0.05).mean() * 100.0),
    }


def _stats_from_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _normal_pairs(surface: np.ndarray, normals: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    pairs = cKDTree(surface).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return pairs, np.empty(0, dtype=np.float32)
    dot = (normals[pairs[:, 0]] * normals[pairs[:, 1]]).sum(axis=1).clip(-1.0, 1.0)
    return pairs, (1.0 - dot).astype(np.float32)


def normal_disagreement_stats(surface: np.ndarray, normals: np.ndarray, radius: float) -> dict[str, float]:
    pairs, err = _normal_pairs(surface, normals, radius)
    if len(err) == 0:
        return {}
    return {"pairs": float(len(pairs)), **_stats_from_values(err)}


def flat_normal_disagreement_stats(
    surface: np.ndarray,
    normals: np.ndarray,
    radius: float,
    keep_percentile: float,
) -> dict[str, float]:
    pairs, err = _normal_pairs(surface, normals, radius)
    if len(err) == 0:
        return {}

    local_sum = np.zeros(len(surface), dtype=np.float64)
    local_count = np.zeros(len(surface), dtype=np.int32)
    np.add.at(local_sum, pairs[:, 0], err)
    np.add.at(local_sum, pairs[:, 1], err)
    np.add.at(local_count, pairs[:, 0], 1)
    np.add.at(local_count, pairs[:, 1], 1)

    local_mean = np.full(len(surface), np.inf, dtype=np.float64)
    has_neighbors = local_count > 0
    local_mean[has_neighbors] = local_sum[has_neighbors] / local_count[has_neighbors]
    threshold = np.percentile(local_mean[has_neighbors], keep_percentile)
    keep = has_neighbors & (local_mean <= threshold)
    flat_pair_mask = keep[pairs[:, 0]] & keep[pairs[:, 1]]
    flat_err = err[flat_pair_mask]
    if len(flat_err) == 0:
        return {
            "pairs": 0.0,
            "vertices": float(keep.sum()),
            "vertex_pct": float(keep.mean() * 100.0),
            "threshold": float(threshold),
        }
    return {
        "pairs": float(len(flat_err)),
        "vertices": float(keep.sum()),
        "vertex_pct": float(keep.mean() * 100.0),
        "threshold": float(threshold),
        **_stats_from_values(flat_err),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--architecture", choices=["cpl", "neus"], default=None)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--normal-radius", type=float, default=None)
    ap.add_argument("--flat-keep-percentile", type=float, default=90.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    f, arch, hidden, depth = load_model(args.pt, args.device, args.architecture)
    sfm = load_colmap_points(args.scene).numpy().astype(np.float32)
    surface = extract_surface(f, args.bound, args.res, args.device, args.chunk)
    if surface is None or len(surface[0]) == 0:
        raise SystemExit(f"no zero crossing found at bound={args.bound} res={args.res}")
    verts, faces, normals = surface

    area_stats = mesh_area_stats(verts, faces)
    sfm_to_surface = nearest_stats(sfm, verts)
    surface_to_sfm = nearest_stats(verts, sfm)
    normal_radius = args.normal_radius
    if normal_radius is None:
        normal_radius = 2.5 * (2 * args.bound / (args.res - 1))
    nstats = normal_disagreement_stats(verts, normals, normal_radius)
    flat_nstats = flat_normal_disagreement_stats(
        verts,
        normals,
        normal_radius,
        args.flat_keep_percentile,
    )
    print(f"checkpoint: {args.pt}")
    print(f"scene:      {args.scene}")
    print(f"model:      arch={arch} hidden={hidden} depth={depth}")
    print(f"surface:    {len(verts):,} MC vertices  {len(faces):,} faces  bound={args.bound} res={args.res}")
    print(f"mesh area total:             {area_stats['total']:.6f}")
    print(f"mesh area largest component: {area_stats['largest']:.6f}")
    print(f"mesh largest component:      {area_stats['largest_pct']:.2f}%  components={int(area_stats['components'])}")
    print(f"sfm points: {len(sfm):,}")
    print("e_i = min_x_surface ||x_sfm_i - x_surface||")
    print(f"mean {sfm_to_surface['mean']:.6f}")
    print(f"p50  {sfm_to_surface['p50']:.6f}")
    print(f"p90  {sfm_to_surface['p90']:.6f}")
    print(f"p99  {sfm_to_surface['p99']:.6f}")
    print(f"max  {sfm_to_surface['max']:.6f}")
    print(f"% e_i < 0.01  {sfm_to_surface['pct_lt_001']:.2f}")
    print(f"% e_i < 0.02  {sfm_to_surface['pct_lt_002']:.2f}")
    print(f"% e_i > 0.05  {sfm_to_surface['pct_gt_005']:.2f}")
    print("d_j = min_x_sfm ||x_surface_j - x_sfm||")
    print(f"mean {surface_to_sfm['mean']:.6f}")
    print(f"p50  {surface_to_sfm['p50']:.6f}")
    print(f"p90  {surface_to_sfm['p90']:.6f}")
    print(f"p99  {surface_to_sfm['p99']:.6f}")
    print(f"max  {surface_to_sfm['max']:.6f}")
    print(f"% d_j < 0.01  {surface_to_sfm['pct_lt_001']:.2f}")
    print(f"% d_j < 0.02  {surface_to_sfm['pct_lt_002']:.2f}")
    print(f"% d_j > 0.05  {surface_to_sfm['pct_gt_005']:.2f}")
    print(f"nearby surface normal pairs: radius={normal_radius:.6f} count={int(nstats.get('pairs', 0))}")
    if nstats:
        print("normal disagreement: 1 - n_i^T n_j")
        print(f"mean {nstats['mean']:.6f}")
        print(f"p50  {nstats['p50']:.6f}")
        print(f"p90  {nstats['p90']:.6f}")
        print(f"p99  {nstats['p99']:.6f}")
        print(f"max  {nstats['max']:.6f}")
    print(
        "flat-zone normal pairs: "
        f"keep={args.flat_keep_percentile:.1f}% "
        f"vertices={int(flat_nstats.get('vertices', 0))} "
        f"({flat_nstats.get('vertex_pct', 0.0):.2f}%) "
        f"local_mean_threshold={flat_nstats.get('threshold', float('nan')):.6f} "
        f"count={int(flat_nstats.get('pairs', 0))}"
    )
    if flat_nstats.get("pairs", 0) > 0:
        print("flat-zone normal disagreement: 1 - n_i^T n_j")
        print(f"mean {flat_nstats['mean']:.6f}")
        print(f"p50  {flat_nstats['p50']:.6f}")
        print(f"p90  {flat_nstats['p90']:.6f}")
        print(f"p99  {flat_nstats['p99']:.6f}")
        print(f"max  {flat_nstats['max']:.6f}")


if __name__ == "__main__":
    main()
