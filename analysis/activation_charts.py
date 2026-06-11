#!/usr/bin/env python3
"""Visualize the partition of an SDF surface into local Fourier charts.

h_theta(x) = F_theta(gamma_L(x)) is piecewise-linear: gamma_L is the fixed
Fourier positional encoding, F_theta a CPL stack with ReLU (inside each
ConvexPotentialLayer) and MaxMin activations. On each activation region R_r the
map is a single affine function -- a "local Fourier chart". The R_r are not
known in closed form; we recover them numerically from activation signatures,
on the *sphere-traced* surface (the true zero-level-set of the SDF).

The two panels show the same partition at two scales, because of a genuine
scale separation:

  * The regions are exact convex polytopes in feature space z = gamma_L(x):
    every ReLU / MaxMin boundary is a hyperplane a.z + b = 0.
  * In surface space x a boundary is the preimage {a.gamma_L(x) + b = 0} --
    a trigonometric level set, genuinely curved.
  * But the shortest PE wavelength is 2*pi / 2^(L-1); a single chart is
    ~1e-3 of it. Within one chart gamma_L is affine to numerical precision,
    so the chart is, to that precision, a flat polytope.

LEFT  -- chart partition at a chart-resolving zoom: faces coloured by
         connected component of equal activation signature, black chart
         boundaries. Charts read as straight-edged polytopes (sub-wavelength).
RIGHT -- the first-CPL ReLU boundary arrangement on a wavelength-scale patch:
         each curve is a boundary {W_i gamma_L(x) + b_i = 0}, drawn as a
         zero-contour. Under the Fourier encoding these are trigonometric
         curves; they would be exactly straight under identity encoding.

Orthographic camera, no axes, light background.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.model import ConvexPotentialLayer, MaxMin, make_model


def load_model(run_dir: Path, ckpt_name: str, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt_path = run_dir / "ckpt" / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = run_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = cfg["model"]
    f = make_model(
        hidden=m["hidden"], depth=m["depth"], group_size=m.get("group_size", 2),
        activation=m.get("activation", "groupsort"),
        input_encoding=m.get("input_encoding", "pe"),
        multires=m.get("multires", 6),
        architecture=m.get("architecture", "cpl"),
    ).to(device)
    f.load_state_dict(ckpt.get("f", ckpt), strict=False)
    with torch.enable_grad():               # prime power-iteration / head buffers
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f, cfg


def _encode(f, x: torch.Tensor) -> torch.Tensor:
    """gamma_L(x), padded to the network width -- the input to F_theta's stack."""
    if f.encoder is not None:
        return F.pad(f.encoder(x), (0, f.hidden - f.encoder.out_dim))
    return F.pad(x, (0, f.hidden - x.shape[-1]))


@torch.no_grad()
def activation_signature(f, x: torch.Tensor) -> torch.Tensor:
    """Boolean activation signature of F_theta(gamma_L(x)) for each row of x.

    Bits, in order: per ConvexPotentialLayer the ReLU sign mask (Wz+b > 0),
    per MaxMin the per-pair swap mask (even >= odd). Identical signature ==
    same affine chart. Only group_size=2 (MaxMin) activations are supported.
    """
    h = _encode(f, x)
    bits: list[torch.Tensor] = []
    for mod in f.net:
        if isinstance(mod, ConvexPotentialLayer):
            sigma_sq = mod._sigma_sq(update_u=False).clamp(min=1e-12)
            pre = F.linear(h, mod.weight, mod.bias)
            bits.append(pre > 0)
            y = F.linear(F.relu(pre), mod.weight.t())
            h = h - (2.0 / sigma_sq) * y
        elif isinstance(mod, MaxMin):
            pairs = h.view(*h.shape[:-1], -1, 2)
            bits.append(pairs[..., 0] >= pairs[..., 1])
            h = torch.stack([pairs.max(-1).values, pairs.min(-1).values],
                            dim=-1).view(h.shape)
        else:
            raise NotImplementedError(
                f"signature only supports CPL/MaxMin, got {type(mod).__name__}")
    return torch.cat(bits, dim=-1)


@torch.no_grad()
def first_cpl_preact(f, x: torch.Tensor) -> torch.Tensor:
    """Pre-activations W z + b of the first ConvexPotentialLayer, z = gamma_L(x).

    Their zero sets {(W gamma_L(x))_i + b_i = 0} are the first-layer ReLU
    chart boundaries -- trigonometric curves under the Fourier encoding.
    """
    cpl0 = next(m for m in f.net if isinstance(m, ConvexPotentialLayer))
    return F.linear(_encode(f, x), cpl0.weight, cpl0.bias)


@torch.no_grad()
def sphere_trace(f, origins: torch.Tensor, dirs: torch.Tensor, eps: float,
                 t_far: float, iters: int, chunk: int):
    """Plain sphere trace of a 1-Lipschitz SDF. Returns (hit_points, hit_mask)."""
    n = origins.shape[0]
    hit_t = torch.zeros(n)
    hit_ok = torch.zeros(n, dtype=torch.bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        o, d = origins[s:e], dirs[s:e]
        t = torch.zeros(e - s)
        done = torch.zeros(e - s, dtype=torch.bool)
        for _ in range(iters):
            act = ~done
            if not act.any():
                break
            sdf = f(o[act] + t[act, None] * d[act])
            t_act = t[act] + sdf
            conv, esc = sdf.abs() < eps, t_act >= t_far
            t[act] = t_act
            idx = act.nonzero(as_tuple=True)[0]
            done[idx[conv | esc]] = True
            hit_ok[s:e][idx[conv]] = True
        hit_t[s:e] = t
    return origins + hit_t[:, None] * dirs, hit_ok


def orthonormal_frame(n: np.ndarray):
    """An (u, v) basis spanning the plane orthogonal to unit vector n."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def trace_patch(f, p0, u, v, n0, window: float, res: int, eps: float,
                iters: int, chunk: int):
    """Orthographic sphere trace of a square surface patch.

    Returns hits [res*res, 3], ok [res*res], and the (gu, gv) camera-plane
    coordinate grids, each [res, res], in normalised units.
    """
    g = np.linspace(-window / 2, window / 2, res)
    gu, gv = np.meshgrid(g, g, indexing="xy")
    plane = p0[None, :] + gu.reshape(-1, 1) * u + gv.reshape(-1, 1) * v
    origins = plane + 1.0 * n0                       # step back along the normal
    dirs = np.broadcast_to(-n0, origins.shape)
    o_t = torch.from_numpy(origins.astype(np.float32))
    d_t = torch.from_numpy(np.ascontiguousarray(dirs, dtype=np.float32))
    hits, ok = sphere_trace(f, o_t, d_t, eps, 2.0, iters, chunk)
    return hits, ok, gu, gv


def chunked(fn, x: torch.Tensor, chunk: int) -> np.ndarray:
    return np.concatenate([fn(x[s:s + chunk]).numpy()
                           for s in range(0, len(x), chunk)], axis=0)


def pixel_charts(sig_label: np.ndarray, hit_mask: np.ndarray) -> np.ndarray:
    """4-connected components of equal-signature pixels; -1 for background."""
    h, w = sig_label.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols = [], []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        link = hit_mask[a] & hit_mask[b] & (sig_label[a] == sig_label[b])
        rows.append(idx[a][link]); cols.append(idx[b][link])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = csr_matrix((np.ones(len(r) * 2, bool),
                        (np.concatenate([r, c]), np.concatenate([c, r]))),
                       shape=(h * w, h * w))
    _, comp = connected_components(graph, directed=False)
    comp = comp.reshape(h, w)
    comp[~hit_mask] = -1
    return comp


def draw_charts(ax, comp: np.ndarray, hit_mask: np.ndarray, extent_mm: float):
    """LEFT panel: surface coloured by chart, black chart boundaries."""
    h, w = comp.shape
    bnd = np.zeros((h, w), bool)
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        diff = hit_mask[a] & hit_mask[b] & (comp[a] != comp[b])
        bnd[a] |= diff; bnd[b] |= diff

    n_comp = int(comp[hit_mask].max()) + 1 if hit_mask.any() else 1
    rng = np.random.default_rng(0)
    palette = plt.cm.hsv(np.linspace(0, 1, max(n_comp, 1), endpoint=False))[:, :3]
    palette = palette[rng.permutation(len(palette))]
    img = np.ones((h, w, 3), np.float32)
    img[hit_mask] = palette[comp[hit_mask]]
    img[bnd] = 0.0
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"chart partition  |  {extent_mm:.2f} mm patch  |  {n_comp} charts\n"
                 "sub-wavelength: charts are flat polytopes", fontsize=8.5)
    ax.axis("off")
    return n_comp


def draw_boundaries(ax, pre: np.ndarray, ok: np.ndarray, gu: np.ndarray,
                    gv: np.ndarray, extent_mm: float, max_curves: int = 48):
    """RIGHT panel: first-CPL ReLU boundary curves on a wavelength-scale patch.

    Each drawn curve is the zero set of one pre-activation; at most max_curves
    are kept (evenly subsampled) so the trigonometric warping reads clearly.
    """
    res = ok.shape[0]
    pre = pre.reshape(res, res, -1).copy()
    pre[~ok] = np.nan
    crossing = []
    for i in range(pre.shape[-1]):
        z = pre[:, :, i]
        zf = z[np.isfinite(z)]
        if zf.size >= 16 and zf.min() < 0.0 < zf.max():
            crossing.append(i)
    sel = (crossing if len(crossing) <= max_curves
           else [crossing[k] for k in np.linspace(0, len(crossing) - 1,
                                                   max_curves).astype(int)])
    ax.contourf(gu, gv, ok.astype(float), levels=[0.5, 1.5], colors=["#eef0f2"])
    for i in sel:
        ax.contour(gu, gv, pre[:, :, i], levels=[0.0], colors=["#16407a"],
                   linewidths=1.15, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(gu.min(), gu.max()); ax.set_ylim(gv.min(), gv.max())
    ax.set_title(f"first-layer ReLU boundaries  |  {extent_mm:.1f} mm patch\n"
                 f"{len(sel)} of {len(crossing)} trigonometric chart-boundary "
                 "curves", fontsize=8.5)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, nargs="?",
                    default=Path("outputs/run_20260521_014634_scan122_4907463"))
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--mesh", type=Path, default=None,
                    help="mesh .ply for picking the camera target "
                         "(default: <run>/ckpt/pred_world_mesh.ply)")
    ap.add_argument("--res", type=int, default=600, help="image side in pixels")
    ap.add_argument("--window-charts", type=float, default=0.006,
                    help="left-panel patch width, normalised units (charts ~2e-4)")
    ap.add_argument("--window-bounds", type=float, default=0.28,
                    help="right-panel patch width, ~one PE wavelength")
    ap.add_argument("--target-vertex", type=int, default=-1,
                    help="mesh vertex index to centre on (-1 = nearest centroid)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--out", type=Path, default=Path("figs/activation_charts.png"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    f, cfg = load_model(args.run_dir, args.ckpt, args.device)
    eps = float(cfg["trace"]["eps"])
    multires = cfg["model"].get("multires", 6)
    pe_wavelength = 2 * np.pi / (2 ** (multires - 1))

    # The mesh is in DTU world space; F_theta operates in normalised space:
    # verts_world = scale_mat @ [verts_norm, 1].
    mesh_path = args.mesh or args.run_dir / "ckpt" / "pred_world_mesh.ply"
    mesh = trimesh.load(mesh_path, process=False)
    scale_mat = np.load(Path(cfg["scene"]) / "cameras.npz")["scale_mat_0"].astype(np.float64)
    inv = np.linalg.inv(scale_mat)
    v_h = np.concatenate([mesh.vertices, np.ones((len(mesh.vertices), 1))], axis=1)
    verts_norm = (inv @ v_h.T).T[:, :3]

    # camera target: a surface point and its outward normal
    vi = (args.target_vertex if args.target_vertex >= 0
          else int(np.argmin(np.linalg.norm(verts_norm - verts_norm.mean(0), axis=1))))
    p0 = verts_norm[vi]
    n0 = inv[:3, :3] @ mesh.vertex_normals[vi]
    n0 /= np.linalg.norm(n0)
    u, v = orthonormal_frame(n0)
    mm = scale_mat[0, 0]
    print(f"target vertex {vi}  |  PE shortest wavelength {pe_wavelength:.3f} units "
          f"({pe_wavelength * mm:.1f} mm)")
    print(f"left patch {args.window_charts:g} units ({args.window_charts / pe_wavelength:.1%} "
          f"of a wavelength)  |  right patch {args.window_bounds:g} units "
          f"({args.window_bounds / pe_wavelength:.1%})")

    # --- LEFT: chart partition on a chart-resolving patch -------------------
    hits_c, ok_c, _, _ = trace_patch(f, p0, u, v, n0, args.window_charts,
                                     args.res, eps, args.iters, args.chunk)
    sig = np.full(args.res * args.res, -1, np.int64)
    hpts = hits_c[ok_c]
    if len(hpts):
        packed = chunked(lambda z: torch.from_numpy(
            np.packbits(activation_signature(f, z).numpy(), axis=1)),
            hpts, args.chunk)
        _, lab = np.unique(packed, axis=0, return_inverse=True)
        sig[ok_c.numpy()] = lab
    sig = sig.reshape(args.res, args.res)
    ok_grid = ok_c.numpy().reshape(args.res, args.res)
    print(f"left: {ok_grid.sum()} hits, "
          f"{len(np.unique(sig[ok_grid]))} unique signatures")
    comp = pixel_charts(sig, ok_grid)

    # --- RIGHT: first-layer ReLU boundary arrangement, wavelength scale -----
    hits_b, ok_b, gu, gv = trace_patch(f, p0, u, v, n0, args.window_bounds,
                                       args.res, eps, args.iters, args.chunk)
    pre = np.full((args.res * args.res, f.hidden), np.nan, np.float32)
    if ok_b.any():
        pre[ok_b.numpy()] = chunked(lambda z: first_cpl_preact(f, z),
                                    hits_b[ok_b], args.chunk)
    print(f"right: {int(ok_b.sum())} hits")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.7))
    fig.patch.set_facecolor("white")
    n_comp = draw_charts(axes[0], comp, ok_grid, args.window_charts * mm)
    draw_boundaries(axes[1], pre, ok_b.numpy().reshape(args.res, args.res),
                    gu, gv, args.window_bounds * mm)
    fig.suptitle(
        r"Local Fourier charts of $h_\theta(x)=F_\theta(\gamma_L(x))$ on the "
        "sphere-traced surface", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}  ({n_comp} charts left)")


if __name__ == "__main__":
    main()
