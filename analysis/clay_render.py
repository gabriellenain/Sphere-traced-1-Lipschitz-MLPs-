#!/usr/bin/env python3
"""Dependency-free CPU clay render of a triangle mesh (matplotlib only).

The cluster GPUs are busy and no offscreen GL renderer (open3d / pyrender /
osmesa) is available on the CPU nodes, so this draws a shaded mesh purely with
a painter's-algorithm fill: orthographic project -> z-sort faces back-to-front
-> lambertian shade by face-normal . light. Good enough for compact, mostly
star-convex init blobs, and identical for every mesh in the figure (the point of
using one renderer for both the eps-ball seeds and the MVSFormer++ carves).
"""
from __future__ import annotations

import numpy as np
import trimesh
from matplotlib.collections import PolyCollection


def _rot(az_deg: float, el_deg: float) -> np.ndarray:
    """World->camera rotation: azimuth about world-up (y), then elevation."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ca, sa = np.cos(az), np.sin(az)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    ce, se = np.cos(el), np.sin(el)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return Rx @ Ry


def clay_render(ax, mesh_path, az=25.0, el=18.0,
                base=(0.83, 0.79, 0.73), light_dir=(-0.35, 0.55, 0.75),
                ambient=0.30, edgewidth=0.0):
    """Draw `mesh_path` onto matplotlib `ax` as a shaded clay model. Returns dict
    of basic stats, or None if the mesh is empty."""
    m = trimesh.load(str(mesh_path), force="mesh")
    if m.is_empty or len(m.faces) == 0:
        return None
    v = np.asarray(m.vertices, np.float64)
    f = np.asarray(m.faces, np.int64)

    # center + isotropic normalize so every panel shares a scale
    c = 0.5 * (v.min(0) + v.max(0))
    v = v - c
    v /= max(np.abs(v).max(), 1e-9)

    R = _rot(az, el)
    vc = v @ R.T                                   # camera frame (+z toward viewer)
    tri = vc[f]                                     # (F,3,3)

    # face normals in camera frame (CCW winding from trimesh -> outward)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    # backface cull: on a closed mesh only front faces (normal toward +z viewer)
    # are visible, which makes painter's fill render a solid surface.
    front = n[:, 2] > 0
    tri, n = tri[front], n[front]

    L = np.array(light_dir, float)
    L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0, 1)
    shade = np.clip(ambient + (1 - ambient) * lam, 0, 1)
    colors = shade[:, None] * np.array(base)[None, :]

    # painter's algorithm: draw far faces first (sort by mean camera depth)
    depth = tri[:, :, 2].mean(1)
    order = np.argsort(depth)

    polys = tri[order][:, :, :2]                    # (F,3,2) image-plane coords
    pc = PolyCollection(
        polys, facecolors=colors[order],
        edgecolors=(colors[order] * 0.6) if edgewidth > 0 else "none",
        linewidths=edgewidth, antialiaseds=True)
    ax.add_collection(pc)

    pad = 0.06
    lo, hi = polys.reshape(-1, 2).min(0), polys.reshape(-1, 2).max(0)
    span = (hi - lo).max() * (1 + 2 * pad)
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return {"verts": len(v), "faces": len(f), "components": int(m.body_count)}
