#!/usr/bin/env python3
"""Fit the 1-Lipschitz FTheta to a COLMAP sparse cloud — closed surface guaranteed.

"Can the sparse SfM points alone seed a usable surface?" — yes, but only if the
SIGN of f is injected somewhere: unsigned point samples cannot decide inside vs
outside (a closed surface IS a sign change), and the textbook IGR/SAL objective

    L = mean_p |f(p)|^2 + lam_off * mean_q exp(-alpha |f(q)|)        (--legacy-igr)

fragments on sparse clouds: the push-away term punches through the zero set in
the gaps between points, and random init settles into specks around clusters.

Default method instead builds the sign by construction, from the points alone
(no normals, no Poisson, no eikonal):

  1. unsigned distance d(x, cloud) on a regular grid over the MC cube;
  2. solid = {d <= eps}, eps ~ 2x median point spacing — the union of eps-balls;
     its boundary is watertight by construction;
  3. flood-fill from the cube border: voxels unreachable from outside are
     interior -> filled (this closes the object and decides the sign);
  4. signed grid = EDT(outside) - EDT(inside): an EXACT SDF of a closed solid —
     and an exact SDF is 1-Lipschitz, i.e. the natural target for FTheta;
  5. regress f onto the grid (MSE, trilinear targets, surface-biased sampling).

The zero set sits eps OUTSIDE the points (inflated like a hull init; training
carves it back). --shift deflates it toward the cloud at the risk of re-opening
thin-coverage gaps. target_mesh.ply is the MC of the grid itself — what the net
is asked to fit. Sibling of fit_gt_sdf.py; reuses its MC / render helpers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lip_tracer.config import ModelConfig
from lip_tracer.model import make_model
from fit_gt_sdf import save_mc_mesh, save_render_png, save_hq_renders, save_loss_plot


def load_points(path: Path) -> np.ndarray:
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"expected (N,>=3) point file, got {pts.shape} from {path}")
    return pts[:, :3]


def sfm_roi(pts: np.ndarray, pad: float) -> tuple[np.ndarray, np.ndarray]:
    """Padded axis-aligned bbox of the SfM cloud. pad is a fraction of extent."""
    lo, hi = pts.min(0), pts.max(0)
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def remove_outliers(pts: np.ndarray, k: int = 8, factor: float = 3.0) -> np.ndarray:
    """Statistical outlier removal: drop points whose k-th NN distance exceeds
    factor x the median. COLMAP sparse clouds always carry stray triangulations;
    each one would otherwise mint its own floating eps-ball speck."""
    from scipy.spatial import cKDTree
    dk = cKDTree(pts).query(pts, k=k + 1, workers=-1)[0][:, -1]
    keep = dk <= factor * np.median(dk)
    print(f"  outlier filter (k={k}, x{factor:g} median): kept {keep.sum():,}/{len(pts):,} "
          f"(dropped {(~keep).sum():,})", flush=True)
    return pts[keep]


def distance_grid(pts: np.ndarray, bound: float, grid_res: int) -> tuple[np.ndarray, float]:
    """Unsigned distance to the cloud on a regular grid over [-bound,bound]^3."""
    from scipy.spatial import cKDTree
    G = grid_res
    xs = np.linspace(-bound, bound, G, dtype=np.float32)
    voxel = float(xs[1] - xs[0])
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    grid_pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    d = cKDTree(pts).query(grid_pts, workers=-1)[0].reshape(G, G, G)
    return d.astype(np.float32), voxel


def _fill_and_count(d: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Flood-fill the eps-ball solid from the cube border; return it + #components."""
    from scipy.ndimage import label
    solid = d <= eps
    lab, _ = label(~solid)
    border = np.unique(np.concatenate([
        lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
        lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    border = border[border != 0]
    solid_filled = ~np.isin(lab, border)
    _, n_comp = label(solid_filled)
    return solid_filled, n_comp


def auto_eps_connect(d: np.ndarray, eps_min: float, voxel: float,
                     max_components: int = 1) -> float:
    """Smallest eps (to ~1 voxel) whose filled solid has <= max_components.

    Median NN spacing says nothing about the LARGEST gaps in a sparse cloud, so
    a fixed eps either fragments or over-inflates. Re-thresholding the same
    distance grid is cheap, so search for the connectivity transition instead.
    """
    hi = eps_min
    for _ in range(20):
        _, n = _fill_and_count(d, hi)
        print(f"    eps={hi:.4g} -> {n} components", flush=True)
        if n <= max_components:
            break
        hi *= 1.4
    else:
        print(f"  [warn] no eps <= {hi:.4g} reaches {max_components} components; "
              f"using it anyway", flush=True)
        return hi
    lo = hi / 1.4 if hi > eps_min else eps_min
    while hi - lo > voxel:
        mid = 0.5 * (lo + hi)
        _, n = _fill_and_count(d, mid)
        if n <= max_components:
            hi = mid
        else:
            lo = mid
    return hi


def signed_grid(d: np.ndarray, eps: float, voxel: float) -> np.ndarray:
    """Exact SDF grid (positive outside) of the flood-filled union of eps-balls.
    Its zero level set is watertight by construction (boundary of a union of
    closed balls, interior cavities filled by the border flood-fill)."""
    from scipy.ndimage import distance_transform_edt
    solid_filled, n_comp = _fill_and_count(d, eps)
    fill_frac = solid_filled.mean()
    print(f"  eps-ball solid: filled={fill_frac:.4f} of cube  components={n_comp}  "
          f"(eps={eps:.4g}, voxel={voxel:.4g})", flush=True)
    if fill_frac > 0.9:
        print("  [warn] solid fills >90% of the cube — eps likely too large or "
              "flood-fill leaked; check bound/eps", flush=True)
    inside_d = distance_transform_edt(solid_filled, sampling=voxel)
    outside_d = distance_transform_edt(~solid_filled, sampling=voxel)
    return (outside_d - inside_d).astype(np.float32)


class GridSDF:
    """Trilinear sampler of the target grid on the training device."""

    def __init__(self, sdf_grid: np.ndarray, bound: float, device: str):
        self.vol = torch.from_numpy(sdf_grid)[None, None].to(device)  # (1,1,Dx,Hy,Wz)
        self.bound = bound

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        u = (x / self.bound).clamp(-1.0, 1.0)
        # grid_sample coord order is (w,h,d) = our (z,y,x) — flip the last dim.
        g = u.flip(-1).view(1, -1, 1, 1, 3)
        out = torch.nn.functional.grid_sample(
            self.vol, g, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(-1)

    def self_test(self, sdf_grid: np.ndarray, n: int = 1000) -> float:
        """Max |grid_sample - scipy trilinear| on random points (axis-order check)."""
        from scipy.ndimage import map_coordinates
        G = sdf_grid.shape[0]
        rs = np.random.default_rng(0)
        x = (rs.random((n, 3), dtype=np.float32) * 2 - 1) * self.bound
        idx = (x + self.bound) / (2 * self.bound) * (G - 1)
        ref = map_coordinates(sdf_grid, idx.T, order=1)
        got = self(torch.from_numpy(x).to(self.vol.device)).cpu().numpy()
        return float(np.abs(got - ref).max())


def main() -> None:
    mc = ModelConfig()
    ap = argparse.ArgumentParser(description="Fit FTheta to a COLMAP sparse cloud (closed by construction).")
    ap.add_argument("--points", type=Path, required=True,
                    help="sparse_sfm_points.txt (already in the normalized training frame)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/points_sdf_fit"))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eps", type=float, default=0.0,
                    help="ball radius for the solid; 0 = auto (smallest eps whose "
                         "filled solid has <= max-components components)")
    ap.add_argument("--max-components", type=int, default=1,
                    help="auto-eps target: allowed connected components of the solid")
    ap.add_argument("--sor-k", type=int, default=8,
                    help="outlier filter: k-th nearest neighbour used for the test")
    ap.add_argument("--sor-factor", type=float, default=3.0,
                    help="outlier filter: drop points with kNN dist > factor x median")
    ap.add_argument("--no-sor", action="store_true",
                    help="disable the statistical outlier filter on the cloud")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="deflate the fitted level set toward the cloud by this distance "
                         "(target+shift). 0 keeps the guaranteed-closed eps-offset surface")
    ap.add_argument("--grid-res", type=int, default=256,
                    help="target SDF grid resolution over the MC cube")
    ap.add_argument("--legacy-igr", action="store_true",
                    help="old objective: |f(p)|^2 + lam_off exp(-alpha|f(q)|) (fragments on sparse clouds)")
    ap.add_argument("--alpha", type=float, default=100.0,
                    help="(legacy) SAL off-surface sharpness: exp(-alpha|f(q)|)")
    ap.add_argument("--lam-off", type=float, default=0.1,
                    help="(legacy) weight on the off-surface term")
    ap.add_argument("--roi-pad", type=float, default=0.1,
                    help="(legacy) off-surface samples drawn inside the SfM AABB padded by this "
                         "fraction of its extent; set <0 to sample the full [-bound,bound] cube")
    ap.add_argument("--bound", type=float, default=1.0, help="marching-cubes half-extent")
    ap.add_argument("--mc-res", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=mc.hidden)
    ap.add_argument("--depth", type=int, default=mc.depth)
    ap.add_argument("--group-size", type=int, default=mc.group_size)
    ap.add_argument("--activation", choices=["groupsort", "nact"], default=mc.activation)
    ap.add_argument("--input-encoding", choices=["identity", "pe"], default=mc.input_encoding)
    ap.add_argument("--multires", type=int, default=mc.multires)
    ap.add_argument("--architecture", choices=["cpl", "neus"], default=mc.architecture)
    ap.add_argument("--lipschitz-mode", choices=["none", "uniform", "per_band"],
                    default=mc.lipschitz_mode)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="skip the slow HQ render")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts_np = load_points(args.points)
    print(f"points: {args.points}")
    print(f"  N={len(pts_np):,}  r_mean={np.linalg.norm(pts_np, axis=1).mean():.3f}  "
          f"aabb=[{pts_np.min(0).round(3)} .. {pts_np.max(0).round(3)}]")

    pts = torch.from_numpy(pts_np).to(device)
    n_pts = pts.shape[0]

    f = make_model(hidden=args.hidden, depth=args.depth, group_size=args.group_size,
                   activation=args.activation, input_encoding=args.input_encoding,
                   multires=args.multires, architecture=args.architecture,
                   lipschitz_mode=args.lipschitz_mode).to(device)
    opt = torch.optim.Adam(f.parameters(), lr=args.lr)

    history: list[tuple[int, float, float, float]] = []

    if args.legacy_igr:
        if args.roi_pad >= 0:
            lo, hi = sfm_roi(pts_np, args.roi_pad)
            print(f"  SFM ROI (pad={args.roi_pad}): [{lo.round(3)} .. {hi.round(3)}]")
        else:
            lo = np.full(3, -args.bound, np.float32)
            hi = np.full(3,  args.bound, np.float32)
            print(f"  off-surface sampling: full cube ±{args.bound}")
        lo_t = torch.from_numpy(lo.astype(np.float32)).to(device)
        hi_t = torch.from_numpy(hi.astype(np.float32)).to(device)
        for s in range(args.steps + 1):
            idx = torch.randint(0, n_pts, (min(args.batch, n_pts),), device=device)
            p = pts[idx]
            q = lo_t + (hi_t - lo_t) * torch.rand(args.batch, 3, device=device)
            fp = f.sdf(p)
            loss_surface = (fp ** 2).mean()
            loss_off = torch.exp(-args.alpha * f.sdf(q).abs()).mean()
            loss = loss_surface + args.lam_off * loss_off
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s % 200 == 0 or s == args.steps:
                history.append((s, loss.item(), loss_surface.item(),
                                args.lam_off * loss_off.item()))
                print(f"  step {s:5d}  loss={loss.item():.5f}  surf={loss_surface.item():.6f}  "
                      f"off={loss_off.item():.5f}  |f(p)|={fp.detach().abs().mean().item():.4f}",
                      flush=True)
        eps = float("nan")
    else:
        if not args.no_sor:
            pts_np = remove_outliers(pts_np, k=args.sor_k, factor=args.sor_factor)
            pts = torch.from_numpy(pts_np).to(device)
            n_pts = pts.shape[0]

        d_grid, voxel = distance_grid(pts_np, args.bound, args.grid_res)
        eps_min = 1.5 * voxel                          # must be resolvable on the grid
        if args.eps > 0:
            eps = max(args.eps, eps_min)
        else:
            # balls must overlap across the LARGEST surface gaps, not the median
            # spacing — search the connectivity transition of the filled solid.
            print(f"  auto-eps: smallest radius with <= {args.max_components} "
                  f"component(s)", flush=True)
            eps = auto_eps_connect(d_grid, eps_min, voxel, args.max_components)
        print(f"  eps={eps:.4g}  shift={args.shift:g}  grid={args.grid_res}^3", flush=True)

        sdf_grid = signed_grid(d_grid, eps, voxel)
        target = GridSDF(sdf_grid, args.bound, device)
        err = target.self_test(sdf_grid)
        assert err < 1e-3 * args.bound, f"grid_sample axis-order self-test failed: {err}"

        # MC of the target itself — the guaranteed-closed surface the net must fit.
        try:
            from skimage import measure
            import trimesh
            v, fc, _, _ = measure.marching_cubes(sdf_grid, level=-args.shift, spacing=(voxel,) * 3)
            v += -args.bound
            tm = trimesh.Trimesh(v, fc, process=False)
            tm.export(args.out_dir / "target_mesh.ply")
            print(f"  target mesh: {len(v):,} verts  {len(fc):,} faces  "
                  f"components={tm.body_count}  watertight={tm.is_watertight}", flush=True)
        except Exception as e:                         # diagnostic artifact only
            print(f"  [warn] target mesh export failed: {e}", flush=True)

        sigma = 3.0 * eps                               # surface-biased half-batch
        for s in range(args.steps + 1):
            nb = args.batch // 2
            q_uni = (torch.rand(nb, 3, device=device) * 2 - 1) * args.bound
            idx = torch.randint(0, n_pts, (args.batch - nb,), device=device)
            q_srf = (pts[idx] + sigma * torch.randn(args.batch - nb, 3, device=device)
                     ).clamp(-args.bound, args.bound)
            q = torch.cat([q_uni, q_srf], dim=0)
            t = target(q) + args.shift
            fq = f.sdf(q)
            loss = ((fq - t) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s % 200 == 0 or s == args.steps:
                with torch.no_grad():
                    # at the cloud points the target is ~ -eps+shift (inside the shell)
                    fp = f.sdf(pts[:min(n_pts, 16384)])
                    drift = (fp - (args.shift - eps)).abs().mean().item()
                history.append((s, loss.item(), loss.item(), drift))
                print(f"  step {s:5d}  mse={loss.item():.6f}  "
                      f"|f(p)-({args.shift - eps:+.3g})|={drift:.4f}", flush=True)

    torch.save({
        "f": f.state_dict(),
        "architecture": f.architecture, "group_size": f.group_size, "depth": f.depth,
        "activation": f.activation, "input_encoding": f.input_encoding,
        "multires": f.multires, "lipschitz_mode": f.lipschitz_mode,
        "points": str(args.points), "bound": args.bound, "eps": eps,
        "shift": args.shift, "legacy_igr": args.legacy_igr,
    }, args.out_dir / "checkpoint_points_sdf.pt")
    save_loss_plot(history, args.out_dir / "loss.png")

    pred_mesh = args.out_dir / "pred_mesh.ply"
    n_v, n_f = save_mc_mesh(f, pred_mesh, args.bound, args.mc_res, device)

    # How faithfully does the extracted surface pass by the cloud? (the default
    # method sits eps-shift OUTSIDE the points by design.) Plus closedness stats.
    if n_f > 0 and n_v > 0:
        import trimesh
        from scipy.spatial import cKDTree
        m = trimesh.load(str(pred_mesh), force="mesh")
        surf, _ = m.sample(min(200_000, max(1, 50 * len(m.faces))), return_index=True)
        d, _ = cKDTree(surf).query(pts_np, workers=-1)   # point -> nearest surface
        fit = {"point_to_surface_mean": float(d.mean()),
               "point_to_surface_p90": float(np.percentile(d, 90)),
               "point_to_surface_max": float(d.max()),
               "expected_offset_eps_minus_shift": (None if args.legacy_igr
                                                   else float(eps - args.shift)),
               "n_verts": int(n_v), "n_faces": int(n_f),
               "n_components": int(m.body_count),
               "watertight": bool(m.is_watertight)}
        (args.out_dir / "fit.json").write_text(json.dumps(fit, indent=2))
        print(f"fit: point->surface mean={fit['point_to_surface_mean']:.5f}  "
              f"p90={fit['point_to_surface_p90']:.5f}  max={fit['point_to_surface_max']:.5f}  "
              f"components={fit['n_components']}  watertight={fit['watertight']}")

    if not args.sweep and n_f > 0:
        save_render_png(pred_mesh, args.out_dir / "pred_mesh_render.png")
        save_hq_renders(pred_mesh, args.out_dir / "pred_mesh_hq.png")
        print("saved renders -> pred_mesh_render.png, pred_mesh_hq.png")


if __name__ == "__main__":
    main()
