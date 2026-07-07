"""Build a screen-space triangle mesh from one view's sphere-traced hits.

For an ICLR figure where we want a Blender render of the SDF *as the tracer
actually sees it* (no marching cubes). Per pixel we sphere-trace to the surface,
build a vertex at the hit point with its analytic ∇f normal, and stitch
adjacent pixels into a face quad — except where the depth difference signals a
silhouette gap. The resulting mesh is open (one side, this camera's side); load
it into Blender from the same view and you see exactly what the tracer renders.

Output PLY is in DTU world coordinates so it matches the marching-cubes
pred_world_mesh.ply convention and can be fed straight into tools/render_blender.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))            # analysis/ siblings (plot_sphere_trace_steps)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `lip_tracer`
from dataclasses import replace

from lip_tracer.config import TraceConfig
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd
import lip_tracer.data as data_mod
from plot_sphere_trace_steps import rays_for_view


class _NeuSSDF(torch.nn.Module):
    """Thin wrapper so the project sphere tracer can call a NeuS SDFNetwork."""

    architecture = "neus"
    input_encoding = "pe"

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.sdf(x).reshape(-1)

    def sdf(self, x):
        return self.forward(x)


def _resolve_ckpt_path(run_dir: Path, ckpt_name: str) -> Path:
    ckpt_path = Path(ckpt_name)
    if ckpt_path.is_absolute() and ckpt_path.exists():
        return ckpt_path
    candidates = [
        run_dir / "ckpt" / ckpt_name,
        run_dir / "checkpoints" / ckpt_name,
        run_dir / ckpt_name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def load_neus_run(run_dir: Path, ckpt_name: str, device: str):
    from pyhocon import ConfigFactory

    neus_dir = Path(__file__).resolve().parent.parent / "baselines" / "NeuS"
    sys.path.insert(0, str(neus_dir))
    from models.fields import SDFNetwork

    conf_path = run_dir / "run.conf"
    conf = ConfigFactory.parse_file(str(conf_path))
    net = SDFNetwork(**conf["model.sdf_network"]).to(device).eval()
    ckpt_path = _resolve_ckpt_path(run_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["sdf_network_fine"])
    f = _NeuSSDF(net).to(device).eval()
    scene = Path(str(conf["dataset.data_dir"]))
    cfg = TraceConfig(iters=256, eps=1e-5, t_far=10.0, newton_steps=8)
    return f, cfg, scene


def load_run(run_dir: Path, ckpt_name: str, device: str):
    if not (run_dir / "config.json").exists() and (run_dir / "run.conf").exists():
        return load_neus_run(run_dir, ckpt_name, device)

    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt_path = _resolve_ckpt_path(run_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = cfg["model"]
    f = make_model(
        hidden=m["hidden"], depth=m["depth"],
        group_size=m.get("group_size", 2),
        activation=m.get("activation", "groupsort"),
        input_encoding=m.get("input_encoding", "pe"),
        multires=m.get("multires", 6),
        architecture=m.get("architecture", "cpl"),
    ).to(device)
    f.load_state_dict(ckpt.get("f", ckpt), strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f, TraceConfig(**cfg["trace"]), Path(cfg["scene"])


def rays_for_view_expanded(views: dict, view_idx: int, device: str, pad: int,
                           ss: int = 1):
    """Like rays_for_view but trace a frustum padded by `pad` pixels on every
    side. Rays are built from K/c2w alone (no image lookup), so out-of-image
    pixels are valid — this captures parts of the subject (e.g. the crown of a
    tall bust) that fall outside the original landscape frame, which is otherwise
    sliced flat at the image border. Still one view, one trace; foreground comes
    from the trace hits, not the image mask.

    `ss` supersamples the padded grid ss× per axis (pixel centres on a 1/ss
    sub-grid). ss=1 reproduces the native integer-pixel grid exactly; ss>1 makes
    the screen-mesh gaps at depth discontinuities sub-pixel so the white backdrop
    no longer bleeds through silhouette/occlusion seams — AND keeps the full
    expanded frustum, so the crown stays whole. Combine expand-px + ss for both.
    """
    K = views["K"][view_idx].numpy()
    c2w = views["c2w"][view_idx].numpy()
    H0, W0 = views["images"][view_idx].shape[:2]
    # continuous pixel-centre coords over the padded frame, ss× denser per axis
    rows = -pad + (np.arange((H0 + 2 * pad) * ss) + 0.5) / ss
    cols = -pad + (np.arange((W0 + 2 * pad) * ss) + 0.5) / ss
    H, W = len(rows), len(cols)
    ys, xs = np.meshgrid(rows, cols, indexing="ij")
    d_cam = np.stack([
        (xs - K[0, 2]) / K[0, 0],
        (ys - K[1, 2]) / K[1, 1],
        np.ones_like(xs, dtype=np.float64),
    ], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).copy()
    return (
        torch.from_numpy(origins.astype(np.float32)).to(device),
        torch.from_numpy(dirs.reshape(-1, 3).astype(np.float32)).to(device),
        H, W,
    )


@torch.no_grad()
def trace(f, origins, dirs, cfg: TraceConfig, chunk: int):
    """Run the project's compacted no-grad sphere tracer chunk by chunk."""
    n = origins.shape[0]
    t_all = torch.zeros(n, device=origins.device)
    hit_all = torch.zeros(n, dtype=torch.bool, device=origins.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, t_c, hit_c = trace_nograd(f, origins[s:e], dirs[s:e], cfg)
        t_all[s:e] = t_c
        hit_all[s:e] = hit_c
        print(f"  traced {e}/{n}", flush=True)
    return t_all.cpu().numpy(), hit_all.cpu().numpy()


def grad_at(f, pts, chunk: int):
    n = pts.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x = pts[s:e].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(x).sum(), x)[0]
        out[s:e] = g.detach().cpu().numpy()
    return out


def eval_f(f, pts, chunk: int):
    """f(x) at each point (no grad) — for the |f(x_hit)| convergence buffer."""
    n = pts.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        with torch.no_grad():
            out[s:e] = f(pts[s:e]).reshape(-1).cpu().numpy()
    return out


def dump_diag(stem: Path, hit: np.ndarray, pts: np.ndarray, n_analytic: np.ndarray,
              grad_norm: np.ndarray, fval: np.ndarray, dirs: np.ndarray,
              H: int, W: int):
    """One 2×3 diagnostic figure for a sphere-traced view (ICLR-minimal).

    Panels: analytic normals (∇f/‖∇f‖) vs screen-space geometric normals (from
    finite-difference of the hit-point position buffer) and their angular error,
    then ‖∇f‖ (1-Lipschitz target = 1), |f(x_hit)| (trace convergence), and
    |n·d| (grazing). All maps share the per-pixel hit/valid mask; misses white.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hit2 = hit.reshape(H, W)
    pos = pts.reshape(H, W, 3).astype(np.float64)
    na = n_analytic.reshape(H, W, 3).astype(np.float64)
    d = dirs.reshape(H, W, 3).astype(np.float64)
    gn = grad_norm.reshape(H, W)
    fv = np.abs(fval).reshape(H, W)
    nd = np.abs((na * d).sum(-1))                       # |n·d|, 1=head-on 0=grazing

    # --- screen-space geometric normal: cross of position derivatives ---
    gx = np.zeros_like(pos); gy = np.zeros_like(pos)
    gx[:, 1:-1] = pos[:, 2:] - pos[:, :-2]             # ∂pos/∂x (image right)
    gy[1:-1, :] = pos[2:, :] - pos[:-2, :]             # ∂pos/∂y (image down)
    n_geo = np.cross(gx, gy)
    n_geo /= np.clip(np.linalg.norm(n_geo, axis=-1, keepdims=True), 1e-12, None)
    n_geo[(n_geo * d).sum(-1) > 0] *= -1               # orient toward camera
    # geometric normal only valid where the 4-neighbourhood are all hits
    valid = np.zeros((H, W), bool)
    valid[1:-1, 1:-1] = (hit2[1:-1, 1:-1] & hit2[1:-1, :-2] & hit2[1:-1, 2:]
                         & hit2[:-2, 1:-1] & hit2[2:, 1:-1])
    ang_err = np.degrees(np.arccos(np.clip(np.abs((na * n_geo).sum(-1)), 0, 1)))

    na_rgb = np.clip(0.5 * (na + 1), 0, 1); na_rgb[~hit2] = 1.0
    ng_rgb = np.clip(0.5 * (n_geo + 1), 0, 1); ng_rgb[~valid] = 1.0

    def _stats(name, s, m):
        v = s[m]
        if v.size:
            print(f"  {name:34s} min {v.min():.4g}  median {np.median(v):.4g}  "
                  f"p90 {np.percentile(v, 90):.4g}  max {v.max():.4g}", flush=True)

    fmax = max(float(np.percentile(fv[hit2], 99)), 1e-6) if hit2.any() else 1.0
    _stats("||grad f|| (1-Lip target=1)", gn, hit2)
    _stats("|f(x_hit)| (convergence)", fv, hit2)
    _stats("|n.d| (grazing)", nd, hit2)
    _stats("normal ang err deg (anal vs geom)", ang_err, valid)

    fig, ax = plt.subplots(2, 3, figsize=(16, 11))

    def _scalar(a, s, m, vmin, vmax, cmap, title):
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
        im = a.imshow(np.ma.array(s, mask=~m), cmap=cm, vmin=vmin, vmax=vmax)
        a.set_title(title); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    ax[0, 0].imshow(na_rgb); ax[0, 0].set_title("analytic normals  ∇f/‖∇f‖"); ax[0, 0].axis("off")
    ax[0, 1].imshow(ng_rgb); ax[0, 1].set_title("geometric normals  (screen-space)"); ax[0, 1].axis("off")
    _scalar(ax[0, 2], ang_err, valid, 0, 30, "magma", "normal angular error (deg)")
    _scalar(ax[1, 0], gn, hit2, 0, 2, "coolwarm", "‖∇f‖  (1-Lipschitz target = 1)")
    _scalar(ax[1, 1], fv, hit2, 0, fmax, "magma", f"|f(x_hit)|  (vmax=p99={fmax:.2g})")
    _scalar(ax[1, 2], nd, hit2, 0, 1, "viridis", "|n·d|  (1=head-on, 0=grazing)")
    fig.tight_layout()

    stem.parent.mkdir(parents=True, exist_ok=True)
    out = stem.with_name(f"{stem.name}_diag.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved diagnostics {out}", flush=True)


def _fg_bbox(views: dict, view_idx: int, pad: float = 0.06):
    """Fixed foreground-mask bbox (r0,r1,c0,c1) — identical regardless of ss, so
    every supersample factor traces the SAME framing. Mirrors rays_for_view_super."""
    mask = views["masks"][view_idx].numpy().astype(bool)
    H, W = mask.shape[:2]
    ys, xs = np.where(mask)
    if ys.size:
        dr = int((ys.max() - ys.min()) * pad) + 1
        dc = int((xs.max() - xs.min()) * pad) + 1
        r0 = max(int(ys.min()) - dr, 0); r1 = min(int(ys.max()) + dr, H - 1) + 1
        c0 = max(int(xs.min()) - dc, 0); c1 = min(int(xs.max()) + dc, W - 1) + 1
    else:
        r0, r1, c0, c1 = 0, H, 0, W
    return r0, r1, c0, c1


def _bbox_rays(views: dict, view_idx: int, device: str, bbox,
               ss: int = 1, jitter=(0.0, 0.0)):
    """Rays over a FIXED bbox. Output pixel grid is (r1-r0)×(c1-c0); each pixel is
    split into ss×ss sub-rays at centres (i+0.5)/ss, plus a constant sub-pixel
    (jx,jy) offset in pixel units. Row-major so reshape (Ho,ss,Wo,ss) recovers the
    sub-ray axes. Returns (origins, dirs, Hs, Ws)."""
    r0, r1, c0, c1 = bbox
    K = views["K"][view_idx].numpy()
    c2w = views["c2w"][view_idx].numpy()
    jx, jy = jitter
    vs = r0 + (np.arange((r1 - r0) * ss) + 0.5) / ss + jy
    us = c0 + (np.arange((c1 - c0) * ss) + 0.5) / ss + jx
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    Hs, Ws = vv.shape
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                      (vv - K[1, 2]) / K[1, 1],
                      np.ones_like(uu)], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).copy()
    return (
        torch.from_numpy(origins.astype(np.float32)).to(device),
        torch.from_numpy(dirs.reshape(-1, 3).astype(np.float32)).to(device),
        Hs, Ws,
    )


@torch.no_grad()
def _chart_code(f, x, ncpl):
    """Activation signature truncated to the first `ncpl` ConvexPotentialLayers
    (+ the MaxMins among them), post-PE. ncpl=1 = first-CPL ReLU partition (256
    boundaries, coarse/readable); ncpl=depth = full sub-pixel partition. The full
    bits are layer-ordered so a smaller ncpl is a strictly coarser chart."""
    from lip_tracer.model import ConvexPotentialLayer, MaxMin
    if f.encoder is not None:
        h = F.pad(f.encoder(x), (0, f.hidden - f.encoder.out_dim))
    else:
        h = F.pad(x, (0, f.hidden - x.shape[-1]))
    bits = []
    seen = 0
    for mod in f.net:
        if isinstance(mod, ConvexPotentialLayer):
            if seen >= ncpl:
                break
            sig = mod._sigma_sq(update_u=False).clamp(min=1e-12)
            pre = F.linear(h, mod.weight, mod.bias)
            bits.append(pre > 0)
            h = h - (2.0 / sig) * F.linear(F.relu(pre), mod.weight.t())
            seen += 1
        elif isinstance(mod, MaxMin):
            if seen >= ncpl:
                break
            pairs = h.view(*h.shape[:-1], -1, 2)
            bits.append(pairs[..., 0] >= pairs[..., 1])
            h = torch.stack([pairs.max(-1).values, pairs.min(-1).values],
                            dim=-1).view(h.shape)
    return torch.cat(bits, dim=-1)


def _save_scalar_png(path: Path, scalar, valid, vmax, cmap, title):
    """Scalar map → colour PNG with colorbar; invalid/miss pixels white."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    fig, ax = plt.subplots(figsize=(6, 9))
    im = ax.imshow(np.ma.array(scalar, mask=~valid), cmap=cm, vmin=0, vmax=vmax)
    ax.set_title(title); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def aliasing_diag(f, views, vi, cfg_trace, args, out_dir: Path, ss_list):
    """Aliasing diagnostic for ONE view, all over the same fixed fg-bbox framing:
      1) SSAA normal maps for each ss, box-averaged to the native bbox resolution.
      2) (ss>1) intra-pixel normal angular std map (deg) — surface-normal dispersion
         among the ss² sub-rays of a pixel; high at silhouettes / normal cliffs.
      3) (ss>1) relative depth std map: std(t_subrays)/mean(t_subrays).
      4) Jitter mode: ss=1 traces at sub-pixel offsets (0,0),(.25,0),(0,.25),(.5,.5),
         each saved as a normal map — shows single-sample normal sensitivity to the
         sampling position (the signature of geometric aliasing).
    """
    import imageio.v2 as imageio

    bbox = _fg_bbox(views, vi)
    r0, r1, c0, c1 = bbox
    Ho, Wo = r1 - r0, c1 - c0
    print(f"\n=== aliasing-diag view {vi} ===  bbox {bbox}  output {Ho}×{Wo}", flush=True)

    def _trace_grid(ss, jitter=(0.0, 0.0)):
        o, d, Hs, Ws = _bbox_rays(views, vi, args.device, bbox, ss=ss, jitter=jitter)
        t, hit = trace(f, o, d, cfg_trace, args.chunk)
        pts = o + torch.from_numpy(t).to(o.device).unsqueeze(-1) * d
        nraw = grad_at(f, pts, args.chunk)
        nn = np.linalg.norm(nraw, axis=1, keepdims=True).clip(min=1e-9)
        nrm = (nraw / nn).astype(np.float32)
        if args.flip_normals:
            nrm = -nrm
        return t, hit, nrm

    def _normals_png(path, nrm_hw3, hit_hw):
        rgb = np.clip(0.5 * (nrm_hw3 + 1.0), 0, 1)
        rgb[~hit_hw] = 1.0
        imageio.imwrite(path, np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8))
        print(f"saved {path}  hit_rate={hit_hw.mean():.2%}", flush=True)

    # ---- 1)+2)+3) SSAA sweep ----
    for ss in ss_list:
        nray = Ho * Wo * ss * ss
        print(f"-- ss={ss}: {nray:,} rays ({ss*ss}×/pixel)", flush=True)
        t, hit, nrm = _trace_grid(ss)
        hit_g = hit.reshape(Ho, ss, Wo, ss)
        nrm_g = nrm.reshape(Ho, ss, Wo, ss, 3)
        t_g = t.reshape(Ho, ss, Wo, ss)

        # SSAA normal map: per-subray normal RGB (miss=white), box-averaged
        rgb_sub = np.where(hit_g[..., None], np.clip(0.5 * (nrm_g + 1.0), 0, 1), 1.0)
        nmap = rgb_sub.mean(axis=(1, 3))
        imageio.imwrite(out_dir / f"view{vi}_ss{ss}_normals.png",
                        np.clip(nmap * 255.0 + 0.5, 0, 255).astype(np.uint8))
        print(f"saved {out_dir / f'view{vi}_ss{ss}_normals.png'}  "
              f"hit_rate={hit_g.mean():.2%}", flush=True)

        if ss > 1:
            S = ss * ss
            hsub = hit_g.transpose(0, 2, 1, 3).reshape(Ho, Wo, S)
            nsub = nrm_g.transpose(0, 2, 1, 3, 4).reshape(Ho, Wo, S, 3)
            tsub = t_g.transpose(0, 2, 1, 3).reshape(Ho, Wo, S)
            w = hsub.astype(np.float64)
            cnt = w.sum(-1)
            valid = cnt >= 2

            # 2) intra-pixel normal angular std (deg) about the per-pixel mean dir
            msum = (nsub * w[..., None]).sum(2)
            mdir = msum / np.clip(np.linalg.norm(msum, axis=-1, keepdims=True), 1e-12, None)
            cosang = np.clip((nsub * mdir[:, :, None, :]).sum(-1), -1, 1)
            ang = np.degrees(np.arccos(cosang))                      # (Ho,Wo,S)
            ang_mean = (ang * w).sum(-1) / np.clip(cnt, 1, None)
            ang_std = np.sqrt(((ang - ang_mean[..., None]) ** 2 * w).sum(-1)
                              / np.clip(cnt, 1, None))
            vmax = float(np.percentile(ang_std[valid], 99)) if valid.any() else 1.0
            _save_scalar_png(out_dir / f"view{vi}_ss{ss}_normal_angstd.png",
                             ang_std, valid, max(vmax, 1e-3), "magma",
                             f"intra-pixel normal ang std (deg)  ss={ss}  (vmax=p99)")

            # 3) relative depth std: std(t)/mean(t) over hit sub-rays
            t_mean = (tsub * w).sum(-1) / np.clip(cnt, 1, None)
            t_std = np.sqrt(((tsub - t_mean[..., None]) ** 2 * w).sum(-1)
                            / np.clip(cnt, 1, None))
            rel = t_std / np.clip(np.abs(t_mean), 1e-9, None)
            vmaxr = float(np.percentile(rel[valid], 99)) if valid.any() else 1.0
            _save_scalar_png(out_dir / f"view{vi}_ss{ss}_reldepth_std.png",
                             rel, valid, max(vmaxr, 1e-6), "viridis",
                             f"rel depth std  std(t)/mean(t)  ss={ss}  (vmax=p99)")

    # ---- 4) sub-pixel jitter mode (ss=1, shifted sample position) ----
    for jx, jy in [(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (0.5, 0.5)]:
        t, hit, nrm = _trace_grid(1, jitter=(jx, jy))
        tag = f"j{jx}_{jy}".replace(".", "p")
        _normals_png(out_dir / f"view{vi}_jitter_{tag}_normals.png",
                     nrm.reshape(Ho, Wo, 3), hit.reshape(Ho, Wo))
    print("aliasing-diag done", flush=True)


def chart_diag(f, views, vi, cfg_trace, args, out_dir: Path):
    """CPL-partition hypothesis test for ONE view (same fixed fg-bbox framing).

    For each sphere-traced hit x: the discrete activation signature of
    F_theta(gamma_L(x)) -- per-CPL ReLU sign mask + per-MaxMin swap mask, all
    layers, computed AFTER the positional encoding -- is hashed to a partition
    ID (= which local affine 'Fourier chart' the point lives in). Writes:
      view<vi>_chart_id.png        partition-ID image (random colour per chart)
      view<vi>_chart_boundary.png  boundary mask (4-neighbour different chart)
      view<vi>_chart_overlay_normals.png   boundary (red) over the normal map
      view<vi>_chart_overlay_angerr.png    boundary (red) over the angular-error map
    Prints chart density (charts per hit pixel): if ~1 chart/pixel the full
    partition is sub-pixel and the boundary map saturates -- charts then can't be
    the *specific* cause of the speckle (they are everywhere); a fraction well
    below 1 with boundaries tracking the speckle would support the hypothesis.
    """
    import imageio.v2 as imageio
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from activation_charts import chunked

    bbox = _fg_bbox(views, vi)
    r0, r1, c0, c1 = bbox
    Ho, Wo = r1 - r0, c1 - c0
    levels = [int(s) for s in args.chart_levels.split(",") if s.strip()]
    levels = [min(L, f.depth) for L in levels]
    print(f"\n=== chart-diag view {vi} ===  bbox {bbox}  output {Ho}×{Wo}  "
          f"levels(ncpl)={levels} (depth={f.depth})", flush=True)

    o, d, Hs, Ws = _bbox_rays(views, vi, args.device, bbox, ss=1)
    t, hit = trace(f, o, d, cfg_trace, args.chunk)
    pts = o + torch.from_numpy(t).to(o.device).unsqueeze(-1) * d
    nraw = grad_at(f, pts, args.chunk)
    nn = np.linalg.norm(nraw, axis=1, keepdims=True).clip(min=1e-9)
    normals = (nraw / nn).astype(np.float32)
    if args.flip_normals:
        normals = -normals

    hit2 = hit.reshape(Ho, Wo)
    na = normals.reshape(Ho, Wo, 3).astype(np.float64)
    dirs_hw = d.detach().cpu().numpy().reshape(Ho, Wo, 3).astype(np.float64)
    nhit = int(hit2.sum())
    okm = hit.reshape(-1)
    hpts = pts.reshape(-1, 3)[torch.from_numpy(okm).to(pts.device)]

    # base normal map (for boundary overlays) computed once
    nmap_base = np.clip(0.5 * (na + 1.0), 0, 1)
    nmap_base[~hit2] = 1.0

    def _labels(ncpl):
        label = np.full(Ho * Wo, -1, np.int64)
        if hpts.shape[0]:
            code = chunked(
                lambda z: torch.from_numpy(
                    np.packbits(_chart_code(f, z, ncpl).cpu().numpy(), axis=1)),
                hpts, args.chunk)
            _, lab = np.unique(code, axis=0, return_inverse=True)
            label[okm] = lab
        return label.reshape(Ho, Wo)

    # --- coarseness sweep: truncate the layer-ordered signature ---
    for ncpl in levels:
        label = _labels(ncpl)
        n_charts = int(label[hit2].max() + 1) if hit2.any() else 0
        bnd = np.zeros((Ho, Wo), bool)
        for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
            diff = hit2[a] & hit2[b] & (label[a] != label[b])
            bnd[a] |= diff; bnd[b] |= diff
        frac_bnd = bnd[hit2].mean() if nhit else 0.0
        cpp = n_charts / nhit if nhit else 0.0
        print(f"  ncpl={ncpl:>2}: charts={n_charts:>8,}  charts/hit-px={cpp:6.3f}  "
              f"boundary px={frac_bnd:6.1%}", flush=True)

        tag = f"L{ncpl}"
        rng = np.random.default_rng(0)
        lut = rng.random((max(n_charts, 1), 3))
        chart_rgb = np.ones((Ho, Wo, 3)); chart_rgb[hit2] = lut[label[hit2]]
        imageio.imwrite(out_dir / f"view{vi}_chart_{tag}_id.png",
                        np.clip(chart_rgb * 255 + 0.5, 0, 255).astype(np.uint8))
        bnd_img = np.ones((Ho, Wo, 3)); bnd_img[bnd] = 0.0
        imageio.imwrite(out_dir / f"view{vi}_chart_{tag}_boundary.png",
                        np.clip(bnd_img * 255 + 0.5, 0, 255).astype(np.uint8))
        nmap = nmap_base.copy(); nmap[bnd] = (1.0, 0.0, 0.0)
        imageio.imwrite(out_dir / f"view{vi}_chart_{tag}_overlay_normals.png",
                        np.clip(nmap * 255 + 0.5, 0, 255).astype(np.uint8))
        if ncpl == levels[0]:
            bnd_coarse = bnd                                   # for ang-err overlay
    print("  → charts/px≈1 = sub-pixel partition (aliased, not resolvable); "
          "a coarse level (ncpl=1) tracking the speckle would support CPL creases",
          flush=True)

    # --- angular-error map (analytic vs screen-space geometric) + coarsest overlay ---
    pos = pts.reshape(-1, 3).detach().cpu().numpy().reshape(Ho, Wo, 3).astype(np.float64)
    gx = np.zeros_like(pos); gy = np.zeros_like(pos)
    gx[:, 1:-1] = pos[:, 2:] - pos[:, :-2]
    gy[1:-1, :] = pos[2:, :] - pos[:-2, :]
    n_geo = np.cross(gx, gy)
    n_geo /= np.clip(np.linalg.norm(n_geo, axis=-1, keepdims=True), 1e-12, None)
    n_geo[(n_geo * dirs_hw).sum(-1) > 0] *= -1
    valid = np.zeros((Ho, Wo), bool)
    valid[1:-1, 1:-1] = (hit2[1:-1, 1:-1] & hit2[1:-1, :-2] & hit2[1:-1, 2:]
                         & hit2[:-2, 1:-1] & hit2[2:, 1:-1])
    ang_err = np.degrees(np.arccos(np.clip(np.abs((na * n_geo).sum(-1)), 0, 1)))
    vmax = float(np.percentile(ang_err[valid], 99)) if valid.any() else 1.0
    aer = cm.get_cmap("magma")(mcolors.Normalize(0, max(vmax, 1e-3))(ang_err))[..., :3]
    aer[~valid] = 1.0
    aer[bnd_coarse] = (0.0, 1.0, 0.0)                   # coarsest-level boundary, green
    imageio.imwrite(out_dir / f"view{vi}_chart_L{levels[0]}_overlay_angerr.png",
                    np.clip(aer * 255 + 0.5, 0, 255).astype(np.uint8))
    print(f"chart-diag done → {out_dir}/view{vi}_chart_*", flush=True)


def largest_depth_component(hit: np.ndarray, t: np.ndarray, H: int, W: int,
                            rel_depth_gap: float) -> np.ndarray:
    """(H,W) bool mask of the largest depth-connected component of hit pixels.

    Two 4-neighbour hits are linked only when their depth spread stays within
    rel_depth_gap·min(t) — the SAME silhouette test build_screen_mesh uses on
    quads. So a background plane seen past the object's silhouette (a depth jump
    away) becomes a separate component and is dropped when it is not the biggest,
    and disconnected floaters/speckles are removed. Keeps the single largest
    component by pixel count.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    hit2 = hit.reshape(H, W)
    t2 = t.reshape(H, W)
    ids = np.arange(H * W).reshape(H, W)

    rows, cols = [], []
    # horizontal neighbours (c, c+1)
    hh = hit2[:, :-1] & hit2[:, 1:]
    ta, tb = t2[:, :-1], t2[:, 1:]
    ok = hh & (np.abs(ta - tb) < rel_depth_gap * np.minimum(ta, tb))
    rows.append(ids[:, :-1][ok]); cols.append(ids[:, 1:][ok])
    # vertical neighbours (r, r+1)
    vv = hit2[:-1, :] & hit2[1:, :]
    ta, tb = t2[:-1, :], t2[1:, :]
    ok = vv & (np.abs(ta - tb) < rel_depth_gap * np.minimum(ta, tb))
    rows.append(ids[:-1, :][ok]); cols.append(ids[1:, :][ok])

    r = np.concatenate(rows); c = np.concatenate(cols)
    data = np.ones(len(r), dtype=np.uint8)
    g = coo_matrix((data, (r, c)), shape=(H * W, H * W))
    n_comp, labels = connected_components(g, directed=False)

    flat_hit = hit2.reshape(-1)
    lab_hit = labels[flat_hit]
    if lab_hit.size == 0:
        return np.zeros((H, W), dtype=bool)
    counts = np.bincount(lab_hit)
    biggest = counts.argmax()
    keep = (labels == biggest) & flat_hit
    print(f"  [largest-component] {n_comp} comps, kept {keep.sum():,}/"
          f"{flat_hit.sum():,} hit px", flush=True)
    return keep.reshape(H, W)


def build_screen_mesh(hit: np.ndarray, pos: np.ndarray, t: np.ndarray,
                      normals: np.ndarray, H: int, W: int,
                      rel_depth_gap: float, grazing_cos: float, dirs_hw3: np.ndarray):
    """Stitch a triangle mesh over the pixel grid.

    hit, t: (H*W,). pos, normals, dirs_hw3: (H*W, 3) in normalized frame.
    A quad's four corners must all be hits, their depth spread must stay
    within rel_depth_gap·min(t), and at least one corner must not be grazing
    (|cos∠(n,d)| ≥ grazing_cos) — that drops both silhouette gaps and
    tangent-grazing faces that would render as black slivers.
    """
    hit2 = hit.reshape(H, W)
    t2 = t.reshape(H, W)
    cos_nd = np.abs((normals * dirs_hw3).sum(-1)).reshape(H, W)

    # 2×2 windows: (i,j) is the top-left corner
    h = hit2[:-1, :-1] & hit2[1:, :-1] & hit2[:-1, 1:] & hit2[1:, 1:]
    tw = np.stack([t2[:-1, :-1], t2[1:, :-1], t2[:-1, 1:], t2[1:, 1:]], axis=-1)
    tmin = tw.min(-1); tmax = tw.max(-1)
    ok_depth = (tmax - tmin) < rel_depth_gap * tmin
    cw = np.stack([cos_nd[:-1, :-1], cos_nd[1:, :-1], cos_nd[:-1, 1:], cos_nd[1:, 1:]], -1)
    ok_graze = cw.max(-1) >= grazing_cos
    keep_quad = h & ok_depth & ok_graze
    print(f"  candidate quads: {h.sum():,}  kept: {keep_quad.sum():,}", flush=True)

    # vertex compaction: only emit vertices that are referenced by some kept face
    referenced = np.zeros((H, W), dtype=bool)
    Y, X = np.where(keep_quad)
    referenced[Y, X] = True
    referenced[Y + 1, X] = True
    referenced[Y, X + 1] = True
    referenced[Y + 1, X + 1] = True
    idx_map = -np.ones((H, W), dtype=np.int64)
    yy, xx = np.where(referenced)
    idx_map[yy, xx] = np.arange(len(yy))
    verts = pos.reshape(H, W, 3)[yy, xx]
    vnorm = normals.reshape(H, W, 3)[yy, xx]

    # two triangles per quad, diagonal chosen to minimize max edge length
    v00 = idx_map[Y, X]
    v10 = idx_map[Y + 1, X]
    v01 = idx_map[Y, X + 1]
    v11 = idx_map[Y + 1, X + 1]
    p00, p10, p01, p11 = verts[v00], verts[v10], verts[v01], verts[v11]
    d_a = np.linalg.norm(p00 - p11, axis=1)  # diagonal 00–11
    d_b = np.linalg.norm(p10 - p01, axis=1)  # diagonal 10–01
    use_a = d_a <= d_b
    tri1 = np.where(use_a[:, None],
                    np.stack([v00, v10, v11], 1),
                    np.stack([v00, v10, v01], 1))
    tri2 = np.where(use_a[:, None],
                    np.stack([v00, v11, v01], 1),
                    np.stack([v10, v11, v01], 1))
    faces = np.concatenate([tri1, tri2], axis=0)
    return verts, vnorm, faces


def dump_buffers(out_png_stem: Path, hit: np.ndarray, normals: np.ndarray,
                 pts: np.ndarray, c2w: np.ndarray, H: int, W: int, ss: int = 1,
                 fg_mask: np.ndarray = None, smooth_sigma: float = 0.0,
                 fill_holes: bool = True, fill_mask_misses: bool = False):
    """Write the raw sphere-trace buffers as PNGs — the surface exactly as the
    tracer sees it from this camera, BEFORE any mesh stitching / culling / Blender.

    Diagnostic for "is geometry absent in f_theta or only in the render?":
    if a structure is missing from view*_shaded.png (a Blender render of the
    stitched mesh) but present here, the loss is in meshing/Blender, not the SDF.
    Reuses the 3-point phong recipe from lip_tracer.render_views so the shaded
    look matches. Misses render white, same as the mesh-render background.
    """
    import imageio.v2 as imageio

    hit2 = hit.reshape(H, W)
    if fg_mask is not None:
        # Drop hits outside the dataset foreground mask (pedestal / floaters /
        # background surface) so the buffers show only the masked object.
        hit2 = hit2 & fg_mask.reshape(H, W).astype(bool)
    n = normals.reshape(H, W, 3).astype(np.float64)
    p = pts.reshape(H, W, 3).astype(np.float64)

    # Fill interior miss pixels (grazing-angle silhouette/thin-feature slivers the
    # sphere trace slips past — |f| never drops to eps, so raising iters does NOT
    # recover them: diagnosed at ~0.25% of the fg). Left alone they punch white
    # holes THROUGH the subject. Fill each hole with the nearest surface pixel's
    # normal + position and mark it a hit, so shading closes it seamlessly. The
    # hole set is restricted to the dataset fg silhouette (fg_mask), so genuine
    # background gaps (between legs, arm↔torso) are NOT filled; without a mask,
    # fall back to fully ENCLOSED holes only (binary_fill_holes), never the open
    # silhouette. Done at the ss grid, before smoothing/SSAA.
    if fill_holes and hit2.any():
        from scipy.ndimage import distance_transform_edt, binary_fill_holes
        if fg_mask is not None:
            # binary_fill_holes on the mask first, so tiny 0-holes IN the dataset
            # eval mask itself (mask defects on the body) also count as interior
            # and get filled — only ENCLOSED holes close, the open leg gap stays.
            silo = binary_fill_holes(fg_mask.reshape(H, W).astype(bool))
            holes = silo & ~hit2
        else:
            holes = binary_fill_holes(hit2) & ~hit2
        if holes.any():
            ir, ic = distance_transform_edt(
                ~hit2, return_distances=False, return_indices=True)
            n[holes] = n[ir[holes], ic[holes]]
            p[holes] = p[ir[holes], ic[holes]]
            hit2[holes] = True
            print(f"[fill-holes] filled {int(holes.sum())} interior miss px "
                  f"from nearest surface", flush=True)

    # Display-only repair for strict masks / subpixel silhouettes: if a cleaned
    # foreground mask says a pixel belongs to the subject but the tracer misses
    # it, copy the nearest traced surface sample. Unlike fill_holes, this also
    # repairs open boundary bites (hands/shoes), so keep it explicit.
    if fill_mask_misses and fg_mask is not None and hit2.any():
        from scipy.ndimage import distance_transform_edt
        holes = fg_mask.reshape(H, W).astype(bool) & ~hit2
        if holes.any():
            ir, ic = distance_transform_edt(
                ~hit2, return_distances=False, return_indices=True)
            n[holes] = n[ir[holes], ic[holes]]
            p[holes] = p[ir[holes], ic[holes]]
            hit2[holes] = True
            print(f"[fill-mask-misses] filled {int(holes.sum())} fg miss px "
                  f"from nearest surface", flush=True)

    # Optional screen-space smoothing of the per-pixel ∇f normal field. The raw
    # 1-Lipschitz gradient carries high-frequency wobble (fine speckle on the
    # body, grain over genuine ridges/terracing) that shading amplifies — visible
    # as striations that are NOT edge aliasing. A small masked Gaussian on the
    # normal VECTORS (then renormalize) removes that grain while preserving the
    # silhouette and low-frequency shape. Masked + normalized-convolution so the
    # white background never bleeds across the silhouette; only hit pixels
    # contribute and edges keep their own normals. σ≈1.0-1.2 px is the sweet spot
    # (σ≳2 starts softening real detail like wing feathers / facial features).
    # Applied at the supersampled grid, BEFORE the SSAA box-average below.
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        m_hit = hit2.astype(np.float64)
        num = np.stack(
            [gaussian_filter(n[..., c] * m_hit, smooth_sigma) for c in range(3)],
            axis=-1,
        )
        den = gaussian_filter(m_hit, smooth_sigma)[..., None]
        n = num / np.clip(den, 1e-6, None)
        n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
        print(f"[smooth] masked Gaussian on normals  sigma={smooth_sigma}px",
              flush=True)

    cam_right = c2w[:3, 0]; cam_up = c2w[:3, 1]; cam_fwd = -c2w[:3, 2]
    cam_pos = c2w[:3, 3]

    def _norm(v):
        return v / (np.linalg.norm(v) + 1e-12)

    key_light = _norm(cam_fwd + 0.6 * cam_up + 0.3 * cam_right)
    fill_light = _norm(cam_fwd - 0.2 * cam_up - 0.8 * cam_right)
    back_light = _norm(-cam_fwd + 0.4 * cam_up)

    view = p - cam_pos[None, None]            # surface->cam is -view; build view dir
    view = -view / (np.linalg.norm(view, axis=-1, keepdims=True) + 1e-12)

    key_d = np.clip((n * key_light).sum(-1, keepdims=True), 0, None)
    fill_d = np.clip((n * fill_light).sum(-1, keepdims=True), 0, None)
    back_d = np.clip((n * back_light).sum(-1, keepdims=True), 0, None)
    half_key = key_light[None, None] + view
    half_key = half_key / (np.linalg.norm(half_key, axis=-1, keepdims=True) + 1e-12)
    spec = np.clip((n * half_key).sum(-1, keepdims=True), 0, None) ** 40

    albedo = np.array([0.92, 0.90, 0.88])
    shading = 0.08 + 0.70 * key_d + 0.18 * fill_d + 0.10 * back_d + 0.25 * spec
    phong = np.clip(shading * albedo, 0.0, 1.0)
    normals_rgb = np.clip(0.5 * (n + 1.0), 0.0, 1.0)

    white = np.ones((H, W, 3))
    phong = np.where(hit2[..., None], phong, white)
    normals_rgb = np.where(hit2[..., None], normals_rgb, white)

    # SSAA: the buffers above are at the ss× supersampled grid. Box-average each
    # ss×ss block of sub-pixels into one output pixel so the hard hit/miss
    # silhouette becomes soft (misses are white, so the mean is a clean alpha
    # blend over edges). ss=1 is a no-op.
    if ss > 1:
        assert H % ss == 0 and W % ss == 0, f"grid {H}x{W} not divisible by ss={ss}"
        Ho, Wo = H // ss, W // ss
        phong = phong.reshape(Ho, ss, Wo, ss, 3).mean(axis=(1, 3))
        normals_rgb = normals_rgb.reshape(Ho, ss, Wo, ss, 3).mean(axis=(1, 3))

    out_png_stem.parent.mkdir(parents=True, exist_ok=True)
    for tag, img in (("phong", phong), ("normals", normals_rgb)):
        path = out_png_stem.with_name(f"{out_png_stem.name}_{tag}.png")
        imageio.imwrite(path, np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
        print(f"saved buffer {path}  hit_rate={hit2.mean():.2%}", flush=True)


def clean_display_mask(mask: np.ndarray, *, fill_holes: bool = True,
                       dilate_px: int = 0) -> np.ndarray:
    """Heal a foreground mask used only for RGB/phong preview compositing."""
    out = mask.astype(bool)
    if fill_holes:
        from scipy.ndimage import binary_fill_holes
        before = int(out.sum())
        out = binary_fill_holes(out)
        added = int(out.sum()) - before
        if added:
            print(f"[mask-clean] filled {added:,} enclosed mask hole px", flush=True)
    if dilate_px > 0:
        from scipy.ndimage import binary_dilation
        before = int(out.sum())
        out = binary_dilation(out, iterations=int(dilate_px))
        added = int(out.sum()) - before
        if added:
            print(f"[mask-clean] dilated mask by {dilate_px}px "
                  f"(+{added:,} px)", flush=True)
    return out


def write_ply(path: Path, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray):
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
        fh.write(header)
        fh.write(vbuf)
        fh.write(fb.tobytes())
    print(f"saved {path}  ({nV:,} verts, {nF:,} faces)", flush=True)


def scene_normalized_to_world(scene: Path) -> np.ndarray:
    """Return 4x4 matrix mapping the training-normalized frame to scene world."""
    if (scene / "transforms_train.json").exists():
        return np.eye(4, dtype=np.float64)

    cam_npz = scene / ("cameras_sphere.npz" if (scene / "cameras_sphere.npz").exists()
                       else "cameras.npz")
    if cam_npz.exists():
        cam_dict = np.load(cam_npz)
        return cam_dict["scale_mat_0"].astype(np.float64)

    bbox_path = scene / "bbox.txt"
    if not bbox_path.exists():
        bbox_path = scene / "pose" / "bbox.txt"
    if bbox_path.exists():
        bbox = np.loadtxt(bbox_path, dtype=np.float64)
        bb_min, bb_max = bbox[:3], bbox[3:6]
        center = 0.5 * (bb_min + bb_max)
        scale = float(np.max(0.5 * (bb_max - bb_min)))
        S = np.eye(4, dtype=np.float64)
        S[:3, :3] *= scale
        S[:3, 3] = center
        return S

    raise FileNotFoundError(
        f"could not find DTU cameras*.npz or TnT bbox.txt under {scene}"
    )


def rays_for_view_super(views: dict, view_idx: int, device: str, ss: int,
                        pad: float = 0.06):
    """Supersampled rays over the foreground-mask bbox (ss× denser per axis).

    The subject can fill ~1% of a far-camera frame (MVMannequin), so a native
    full-frame trace lands few rays on it → a sparse, grainy screen mesh. This
    restricts a *regular* grid to the fg bbox and samples it ss× per axis, so
    nearly every ray hits the subject — matching DTU-level vertex density for
    roughly the same ray budget. Regular grid → build_screen_mesh still stitches
    quads. Returns (origins, dirs, Hs, Ws, bbox), same dtype/device as
    rays_for_view's first two outputs.
    """
    mask = views["masks"][view_idx].numpy().astype(bool)
    H, W = mask.shape[:2]
    K = views["K"][view_idx].numpy()
    c2w = views["c2w"][view_idx].numpy()
    ys_fg, xs_fg = np.where(mask)
    if ys_fg.size:
        dr = int((ys_fg.max() - ys_fg.min()) * pad) + 1
        dc = int((xs_fg.max() - xs_fg.min()) * pad) + 1
        r0 = max(int(ys_fg.min()) - dr, 0); r1 = min(int(ys_fg.max()) + dr, H - 1) + 1
        c0 = max(int(xs_fg.min()) - dc, 0); c1 = min(int(xs_fg.max()) + dc, W - 1) + 1
    else:
        r0, r1, c0, c1 = 0, H, 0, W
    # dense pixel-center coords (original-image pixel space) over the bbox.
    # ss=1 reduces to integer pixel centres (xs+0.5), matching rays_for_view.
    vs = r0 + (np.arange((r1 - r0) * ss) + 0.5) / ss   # rows  (y)
    us = c0 + (np.arange((c1 - c0) * ss) + 0.5) / ss   # cols  (x)
    vv, uu = np.meshgrid(vs, us, indexing="ij")        # (Hs, Ws), row-major
    Hs, Ws = vv.shape
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                      (vv - K[1, 2]) / K[1, 1],
                      np.ones_like(uu)], axis=-1)
    dirs = d_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).reshape(-1, 3).copy()
    return (
        torch.from_numpy(origins.astype(np.float32)).to(device),
        torch.from_numpy(dirs.reshape(-1, 3).astype(np.float32)).to(device),
        Hs, Ws, (r0, r1, c0, c1),
    )


def _look_at(cam, target):
    """world->cam pose (OpenCV-style c2w) looking from cam toward target."""
    f = target - cam; f /= np.linalg.norm(f)
    up = np.array([0, 1, 0], np.float32)
    if abs(np.dot(f, up)) > 0.95:
        up = np.array([0, 0, 1], np.float32)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = r; c2w[:3, 1] = -u; c2w[:3, 2] = f; c2w[:3, 3] = cam
    return c2w


def run_synth_cam(args):
    """Trace a synthesized orbit camera (normalized frame) and dump phong+normal
    buffers + the screen mesh. No dataset view / GT — pose is arbitrary."""
    f, cfg, scene = load_run(args.run_dir, args.ckpt, args.device)
    dev = args.device
    iters = args.max_iters if args.max_iters is not None else cfg.iters
    cfg_trace = replace(cfg, iters=iters)

    H = W = args.synth_res
    fpix = 0.5 * W / np.tan(np.deg2rad(args.synth_fov) / 2)
    K = np.array([[fpix, 0, W / 2], [0, fpix, H / 2], [0, 0, 1]], np.float64)
    center = np.asarray(args.synth_center, np.float32)
    e = np.deg2rad(args.synth_elev); az = np.deg2rad(args.synth_az)
    cam = center + args.synth_radius * np.array(
        [np.cos(az) * np.cos(e), np.sin(e), np.sin(az) * np.cos(e)], np.float32)
    c2w = _look_at(cam, center)
    if args.synth_roll:
        th = np.deg2rad(args.synth_roll)
        R, Dn = c2w[:3, 0].copy(), c2w[:3, 1].copy()   # right, image-down
        c2w[:3, 0] = np.cos(th) * R + np.sin(th) * Dn
        c2w[:3, 1] = -np.sin(th) * R + np.cos(th) * Dn
    print(f"\n=== synth cam ===  elev={args.synth_elev} az={args.synth_az} "
          f"r={args.synth_radius} fov={args.synth_fov} {H}×{W}  eps={cfg.eps:g}", flush=True)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dcam = np.stack([(xs + .5 - K[0, 2]) / K[0, 0],
                     (ys + .5 - K[1, 2]) / K[1, 1], np.ones_like(xs, np.float64)], -1)
    dw = dcam @ c2w[:3, :3].T.astype(np.float64)
    dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
    origins = torch.from_numpy(np.broadcast_to(c2w[:3, 3], dw.shape).reshape(-1, 3).copy()).float().to(dev)
    dirs = torch.from_numpy(dw.reshape(-1, 3)).float().to(dev)

    t, hit = trace(f, origins, dirs, cfg_trace, args.chunk)
    print(f"hits: {int(hit.sum()):,} / {len(hit):,}", flush=True)
    pts = origins + torch.from_numpy(t).to(dev).unsqueeze(-1) * dirs
    nraw = grad_at(f, pts, args.chunk)
    normals = (nraw / np.linalg.norm(nraw, axis=1, keepdims=True).clip(min=1e-9)).astype(np.float32)
    if args.flip_normals:
        normals = -normals

    buf_stem = args.out.with_suffix("") if args.out else \
        args.run_dir / "ckpt" / "st_buffers_synth"
    dump_buffers(buf_stem, hit, normals,
                 pts.detach().cpu().numpy().astype(np.float32),
                 c2w.astype(np.float64), H, W)

    verts_n, vnorm, faces = build_screen_mesh(
        hit, pts.detach().cpu().numpy().astype(np.float32), t.astype(np.float32),
        normals, H, W, args.rel_depth_gap, args.grazing_cos,
        dw.reshape(-1, 3).astype(np.float32))
    out = args.out if args.out else args.run_dir / "ckpt" / "sphere_traced_screen_synth.ply"
    write_ply(out, verts_n.astype(np.float32),
              (vnorm / np.linalg.norm(vnorm, axis=1, keepdims=True).clip(min=1e-9)).astype(np.float32),
              faces)
    print(f"wrote {out} (+ {buf_stem.name}_phong.png / _normals.png)", flush=True)

    # ---- GT mesh from the SAME camera (phong + normals), for side-by-side ----
    if args.gt_mesh is not None:
        import open3d as o3d
        S = scene_normalized_to_world(scene)          # normalized -> world
        gt = o3d.io.read_triangle_mesh(str(args.gt_mesh))
        V = np.asarray(gt.vertices, np.float64)
        # Some repo GT meshes are stored already in the normalized frame, others
        # in scene world (DTU mm). Detect: a world mesh has coords far outside the
        # unit cube, so only map world->normalized when it doesn't already fit.
        already_norm = np.abs(V).max() < 1.6
        if not already_norm:
            V = (np.concatenate([V, np.ones((len(V), 1))], 1) @ np.linalg.inv(S).T)[:, :3]
        print(f"  GT mesh frame: {'normalized (no transform)' if already_norm else 'world -> inv(scale_mat)'}"
              f"  bbox {V.min(0).round(3)}..{V.max(0).round(3)}", flush=True)
        gt.vertices = o3d.utility.Vector3dVector(V)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
        rays = np.concatenate([
            origins.cpu().numpy().astype(np.float32),
            dirs.cpu().numpy().astype(np.float32)], axis=1)
        res = rc.cast_rays(o3d.core.Tensor(rays))
        t_gt = res["t_hit"].numpy()
        hit_gt = np.isfinite(t_gt)
        n_gt = res["primitive_normals"].numpy().astype(np.float32)
        d_np = dirs.cpu().numpy()
        flip = (n_gt * d_np).sum(1) > 0          # orient toward camera (oppose ray)
        n_gt[flip] *= -1
        nn = np.linalg.norm(n_gt, axis=1, keepdims=True).clip(min=1e-9)
        n_gt = (n_gt / nn).astype(np.float32)
        pts_gt = origins.cpu().numpy() + np.nan_to_num(t_gt, nan=0.0)[:, None] * d_np
        gt_stem = buf_stem.with_name(buf_stem.name + "_gt")
        dump_buffers(gt_stem, hit_gt, n_gt, pts_gt.astype(np.float32),
                     c2w.astype(np.float64), H, W)
        print(f"wrote GT-mesh buffers: {gt_stem.name}_phong.png / _normals.png "
              f"(hits {int(hit_gt.sum()):,}/{hit_gt.size:,})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--view", type=int, default=16)
    ap.add_argument("--views", type=str, default=None,
                    help="comma-list or 'all' — overrides --view")
    ap.add_argument("--down", type=int, default=1)
    ap.add_argument("--expand-px", type=int, default=0,
                    help="pad the traced frustum by N pixels per side so a subject "
                         "clipped by the landscape frame (e.g. a tall bust's crown) "
                         "is captured whole. Single view, single trace.")
    ap.add_argument("--supersample", type=int, default=1,
                    help="ss× denser rays over the fg-mask bbox (regular grid). "
                         "For a small/far subject (MVMannequin ~1%% of frame) this "
                         "lifts the screen mesh to DTU-level density. 1 = native.")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="sphere-trace iteration budget. Default = the run config's "
                         "own `iters`, i.e. EXACTLY what was used to render this .pt. "
                         "Only set this to deliberately deviate from the run.")
    ap.add_argument("--eps", type=float, default=None,
                    help="sphere-trace hit threshold |f(x)|<eps. Default = run cfg "
                         "(NeuS hardcodes 1e-5). Override to match another run's trace.")
    ap.add_argument("--newton-steps", type=int, default=None,
                    help="Newton refinement steps after the sphere trace. Default = "
                         "run cfg (NeuS hardcodes 8). Override to match another run.")
    ap.add_argument("--t-far", type=float, default=None,
                    help="sphere-trace far cutoff (ray length). Default = run cfg. "
                         "Some rigs (e.g. MVMannequin) place cameras farther from the "
                         "normalized origin than the run's t_far, so the ray is cut off "
                         "before the surface (0 hits). Raise it to reach the object.")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--rel-depth-gap", type=float, default=0.02,
                    help="reject quads where max(Δt) > rel·min(t)")
    ap.add_argument("--grazing-cos", type=float, default=0.05,
                    help="reject quads with all corners more grazing than this")
    ap.add_argument("--flip-normals", action="store_true",
                    help="flip normals to point toward the camera")
    ap.add_argument("--gt-photo", action="store_true",
                    help="Also write <stem>_gt.png: the masked dataset photo for "
                         "this view, cropped to the SAME fg-bbox as the supersample "
                         "trace and upsampled to the buffer grid (GT reference for "
                         "side-by-side with the _phong/_normals buffers). Requires "
                         "--dump-buffers and --supersample>1.")
    ap.add_argument("--largest-component", action="store_true",
                    help="(--dump-buffers only) keep only the largest "
                         "depth-connected component of hit pixels (same "
                         "silhouette depth-gap test as the mesh), dropping "
                         "background planes seen past the silhouette and "
                         "detached floaters. Combined with --mask-bg.")
    ap.add_argument("--dump-buffers", action="store_true",
                    help="also write raw per-pixel phong + normal PNGs from the "
                         "trace (pre-meshing, pre-Blender) next to --out, to tell "
                         "whether missing geometry is in f_theta or the render.")
    ap.add_argument("--no-fill-holes", action="store_true",
                    help="(--dump-buffers only) DON'T fill interior miss pixels "
                         "(grazing-angle slivers the trace slips past) from the "
                         "nearest surface pixel. Filling is on by default so the "
                         "subject has no white holes punched through it; pass this "
                         "to see the raw hit/miss buffer instead.")
    ap.add_argument("--fill-mask-misses", action="store_true",
                    help="(--dump-buffers/--mask-bg only) fill every traced miss "
                         "inside the cleaned display mask from the nearest hit. "
                         "Repairs open silhouette bites at hands/shoes; display "
                         "only, not a raw geometry diagnostic.")
    ap.add_argument("--smooth-sigma", type=float, default=0.0,
                    help="(--dump-buffers only) masked Gaussian sigma (px) applied "
                         "to the per-pixel normal field before shading, to remove "
                         "the high-frequency ∇f striations/grain without softening "
                         "the silhouette or genuine detail. 0=off; 1.0-1.2 is a "
                         "good slight smoothing (>~2 starts losing fine detail).")
    ap.add_argument("--mask-bg", action="store_true",
                    help="(--dump-buffers only) composite the dataset foreground "
                         "mask onto the phong/normal buffers: any hit outside the "
                         "GT mask (pedestal/floaters/background surface) is set to "
                         "white. Minimal DTU preview that shows only the masked "
                         "object the way the dataset masks define it.")
    ap.add_argument("--no-mask-fill-holes", action="store_true",
                    help="(--dump-buffers/--mask-bg only) keep enclosed 0-islands "
                         "inside the display mask. By default they are filled so "
                         "mask defects do not punch white pixels through GT/phong.")
    ap.add_argument("--mask-dilate-px", type=int, default=0,
                    help="(--dump-buffers/--mask-bg only) dilate the display mask "
                         "by this many pixels after hole filling. Useful for "
                         "strict eval masks that trim hands/shoes at the boundary.")
    ap.add_argument("--diag", action="store_true",
                    help="write <stem>_diag.png: a 2x3 figure comparing analytic vs "
                         "screen-space geometric normals (+ angular error) and "
                         "mapping ||grad f||, |f(x_hit)| and |n.d| on the surface. "
                         "Traces the fg-bbox crop (like --supersample) so it is "
                         "object-tight even at ss=1.")
    ap.add_argument("--alias-diag", action="store_true",
                    help="aliasing study for one view over a FIXED fg-bbox framing: "
                         "SSAA normal maps for each --alias-ss (box-averaged to the "
                         "native bbox res), plus (ss>1) intra-pixel normal angular "
                         "std + relative-depth std maps, plus a sub-pixel jitter "
                         "sweep of ss=1 normal maps. Writes view<vi>_* PNGs next to "
                         "--out (or run_dir/ckpt). Skips meshing.")
    ap.add_argument("--alias-ss", type=str, default="1,2,4,8",
                    help="comma list of supersample factors for --alias-diag. "
                         "WARNING ss=8 over a full bbox is 64x rays — slow on CPU.")
    ap.add_argument("--chart-diag", action="store_true",
                    help="CPL-partition test: hash each hit's full activation "
                         "signature (per-CPL ReLU + per-MaxMin swap, post-PE) into a "
                         "chart ID; write partition-ID + boundary maps and overlay "
                         "the boundary on the normal and angular-error maps. Reports "
                         "chart density (charts per hit pixel). Fixed fg-bbox, ss=1.")
    ap.add_argument("--chart-levels", type=str, default="1,2,4,8",
                    help="coarseness sweep for --chart-diag: comma list of CPL counts "
                         "(ncpl). 1 = first-CPL ReLU partition (256 boundaries, "
                         "readable); depth = full sub-pixel partition. One trace, "
                         "bits truncated per level.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ba-poses", type=Path, default=None,
                    help="a bundle_adjustment ba_final.pt: override each dataset "
                         "view's c2w with the BA-adjusted pose "
                         "(c2w = rodrigues(log_rot)@R_base, t = t_base+dt). Renders "
                         "the run from the cameras BA co-optimised. Pair with "
                         "--ckpt <ba_final.pt> to also use the BA-updated f.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # --- synthesized look-at camera (not a dataset view) ---
    # Trace + dump buffers from an arbitrary orbit camera in the model's
    # normalized frame, so a subject can be shown from a pose no dataset view
    # captured (e.g. the clock rotated so its mechanism faces the camera). There
    # is no GT image for a synthesized pose, so this is phong+normals only.
    ap.add_argument("--synth-cam", action="store_true",
                    help="render a synthesized look-at camera instead of a dataset view")
    ap.add_argument("--synth-elev", type=float, default=-85.0, help="elevation deg")
    ap.add_argument("--synth-az", type=float, default=0.0, help="azimuth deg")
    ap.add_argument("--synth-radius", type=float, default=2.9)
    ap.add_argument("--synth-center", type=float, nargs=3, default=[-0.098, 0.025, 0.051],
                    help="orbit target (object centroid) in normalized frame")
    ap.add_argument("--synth-fov", type=float, default=22.0, help="vertical FOV deg")
    ap.add_argument("--synth-res", type=int, default=1100)
    ap.add_argument("--synth-roll", type=float, default=0.0,
                    help="in-plane camera roll deg (straighten a subject that "
                         "looks tilted at the chosen elev/az). + rotates image CCW.")
    ap.add_argument("--gt-mesh", type=Path, default=None,
                    help="(synth-cam only) also render this GT mesh's phong+normals "
                         "from the SAME camera, written as <stem>_gt_phong/_normals.png. "
                         "Mesh is assumed in scene world frame and mapped to the "
                         "normalized camera frame via inv(scale_mat).")
    args = ap.parse_args()

    if args.synth_cam:
        return run_synth_cam(args)

    f, cfg, scene = load_run(args.run_dir, args.ckpt, args.device)
    data_mod.SCENE = scene
    if (scene / "transforms_train.json").exists():
        views = data_mod.load_blender_views(scene, split="train", down=args.down)
    else:
        views = data_mod.load_views(scene, down=args.down)
    nV = int(views["images"].shape[0])

    if args.ba_poses is not None:
        # Replace calibrated c2w with the BA-adjusted extrinsics. R_base/t_base in
        # cam_params ARE views["c2w"]'s rotation/centre (prepare_context sets
        # c2w_base = views["c2w"]), so this stays in the model's normalized frame
        # and is resolution-independent (pose doesn't depend on --down).
        from lip_tracer.bundle_adjustment import rodrigues
        ba = torch.load(args.ba_poses, map_location="cpu", weights_only=False)
        cp = ba["cam_params"]
        assert cp["R_base"].shape[0] == nV, \
            f"ba cam_params V={cp['R_base'].shape[0]} != scene V={nV}"
        log_rot = cp["log_rot"] * cp["free_mask"]
        dt      = cp["dt"]      * cp["free_mask"]
        R = rodrigues(log_rot) @ cp["R_base"]              # (V,3,3)
        t = cp["t_base"] + dt                              # (V,3)
        c2w = torch.zeros(nV, 4, 4, dtype=views["c2w"].dtype)
        c2w[:, :3, :3] = R.to(c2w.dtype)
        c2w[:, :3, 3]  = t.to(c2w.dtype)
        c2w[:,  3, 3]  = 1.0
        d0 = (views["c2w"][:, :3, 3] - t).norm(dim=-1)
        views["c2w"] = c2w
        print(f"[ba-poses] overrode {nV} c2w from {args.ba_poses}  "
              f"(mean |Δcentre|={d0.mean():.4e} max={d0.max():.4e} norm units)",
              flush=True)
        # If BA also refined intrinsics (opt_intrinsics), apply the SAME
        # dimensionless delta CameraParams.intrinsics() uses so the render
        # reflects the cameras BA actually optimised, not just the extrinsics.
        # K_base in cam_params == views["K"]; dK is (V,4) per-camera or (1,4)
        # shared (broadcasts either way). Absent dK → extrinsics-only (legacy).
        if "dK" in cp:
            Kb = views["K"].clone().float()
            fx0, fy0 = Kb[:, 0, 0].clone(), Kb[:, 1, 1].clone()
            dK = cp["dK"].float()
            Kb[:, 0, 0] = fx0 * dK[:, 0].exp()
            Kb[:, 1, 1] = fy0 * dK[:, 1].exp()
            Kb[:, 0, 2] = Kb[:, 0, 2] + dK[:, 2] * fx0
            Kb[:, 1, 2] = Kb[:, 1, 2] + dK[:, 3] * fy0
            views["K"] = Kb.to(views["K"].dtype)
            print(f"[ba-poses] also applied BA intrinsics: "
                  f"{'shared' if dK.shape[0] == 1 else 'per-camera'}  "
                  f"mean focal scale={dK[:, :2].exp().mean():.5f}  "
                  f"mean |Δcx,cy|={(dK[:, 2:].abs() * torch.stack([fx0, fy0], -1)).mean():.3f}px",
                  flush=True)
    if args.views is not None:
        view_list = list(range(nV)) if args.views.strip() == "all" else \
                    [int(s) % nV for s in args.views.split(",")]
    else:
        view_list = [args.view % nV]

    S = scene_normalized_to_world(scene)
    # Faithful by default: render the .pt with the SAME trace iters its run used
    # (cfg.iters loaded from config.json). eps / newton_steps / bracketing are
    # already inherited from cfg, so the whole trace matches the run unless
    # --max-iters is explicitly passed to deviate.
    iters = args.max_iters if args.max_iters is not None else cfg.iters
    eps = args.eps if args.eps is not None else cfg.eps
    newton = args.newton_steps if args.newton_steps is not None else cfg.newton_steps
    t_far = args.t_far if args.t_far is not None else cfg.t_far
    cfg_trace = replace(cfg, iters=iters, eps=eps, newton_steps=newton, t_far=t_far)
    over = [n for n, v in (("iters", args.max_iters), ("eps", args.eps),
                           ("newton", args.newton_steps), ("t_far", args.t_far)) if v is not None]
    print(f"trace cfg: iters={cfg_trace.iters} eps={cfg_trace.eps:g} "
          f"newton={cfg_trace.newton_steps} t_far={cfg_trace.t_far:g}"
          f"{'  [DEFAULT = run value]' if not over else '  [OVERRIDDEN: ' + ','.join(over) + ']'}",
          flush=True)

    for vi in view_list:
        bbox = None
        if args.alias_diag:
            out_dir = (args.out.parent if args.out else args.run_dir / "ckpt")
            out_dir.mkdir(parents=True, exist_ok=True)
            ss_list = [int(s) for s in args.alias_ss.split(",") if s.strip()]
            aliasing_diag(f, views, vi, cfg_trace, args, out_dir, ss_list)
            continue
        if args.chart_diag:
            out_dir = (args.out.parent if args.out else args.run_dir / "ckpt")
            out_dir.mkdir(parents=True, exist_ok=True)
            chart_diag(f, views, vi, cfg_trace, args, out_dir)
            continue
        if args.expand_px > 0:
            origins, dirs, H, W = rays_for_view_expanded(
                views, vi, args.device, args.expand_px, ss=args.supersample)
            print(f"\n=== view {vi} ===  expanded +{args.expand_px}px/side "
                  f"ss×{args.supersample} → grid {H}×{W} = {H*W:,} rays  "
                  f"eps={cfg.eps:g}", flush=True)
        elif args.supersample > 1 or args.diag:
            origins, dirs, H, W, bbox = rays_for_view_super(
                views, vi, args.device, max(args.supersample, 1))
            print(f"\n=== view {vi} ===  supersample×{max(args.supersample, 1)} over fg bbox "
                  f"{bbox} → grid {H}×{W} = {H*W:,} rays  eps={cfg.eps:g}", flush=True)
        else:
            origins, dirs, _, _, H, W = rays_for_view(views, vi, args.device)
            print(f"\n=== view {vi} ===  H×W={H}×{W} eps={cfg.eps:g}", flush=True)
        t, hit = trace(f, origins, dirs, cfg_trace, args.chunk)
        print(f"hits: {int(hit.sum()):,} / {len(hit):,}", flush=True)
        pts = (origins + torch.from_numpy(t).to(origins.device).unsqueeze(-1) * dirs)
        normals_raw = grad_at(f, pts, args.chunk)
        grad_norm = np.linalg.norm(normals_raw, axis=1).astype(np.float32)  # ‖∇f‖
        n_norm = np.linalg.norm(normals_raw, axis=1, keepdims=True).clip(min=1e-9)
        normals = (normals_raw / n_norm).astype(np.float32)
        if args.flip_normals:
            normals = -normals

        if args.out and len(view_list) == 1:
            buf_stem = args.out.with_suffix("")
        else:
            buf_stem = args.run_dir / "ckpt" / f"st_buffers_view{vi:02d}"

        if args.diag:
            dump_diag(
                buf_stem, hit,
                pts.detach().cpu().numpy().astype(np.float32),
                normals, grad_norm, eval_f(f, pts, args.chunk),
                dirs.detach().cpu().numpy().astype(np.float32), H, W,
            )

        if args.dump_buffers:
            fg_mask = None
            if args.mask_bg:
                # Sample the dataset fg mask onto the trace grid (full frame at
                # ss=1, or the fg-bbox crop when --supersample>1), nearest so it
                # stays a hard 0/1 silhouette.
                from PIL import Image
                m = views["masks"][vi].numpy().astype(bool)      # (Himg, Wimg)
                if args.expand_px > 0:
                    # Expanded full-frame grid: rows/cols run -pad..(H0/W0+pad) in
                    # native pixel coords, ss× denser (see rays_for_view_expanded).
                    # Sample the native mask at each sub-pixel's native index; any
                    # sub-pixel outside the image (the pad margin) is background.
                    H0, W0 = m.shape
                    pad, ss = args.expand_px, max(args.supersample, 1)
                    rr = np.floor(-pad + (np.arange(H) + 0.5) / ss).astype(int)
                    cc = np.floor(-pad + (np.arange(W) + 0.5) / ss).astype(int)
                    rv, cv = (rr >= 0) & (rr < H0), (cc >= 0) & (cc < W0)
                    fg = m[np.clip(rr, 0, H0 - 1)[:, None],
                           np.clip(cc, 0, W0 - 1)[None, :]]
                    fg_mask = fg & rv[:, None] & cv[None, :]
                else:
                    if bbox is not None:
                        r0, r1, c0, c1 = bbox
                        m = m[r0:r1, c0:c1]
                    if m.shape != (H, W):   # ss× fg-bbox crop: linear NEAREST is exact
                        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                                       .resize((W, H), Image.NEAREST)) > 127
                    fg_mask = m
                raw_mean = fg_mask.mean()
                fg_mask = clean_display_mask(
                    fg_mask,
                    fill_holes=not args.no_mask_fill_holes,
                    dilate_px=max(0, args.mask_dilate_px),
                )
                print(f"[mask-bg] fg mask {fg_mask.mean():.2%} of grid", flush=True)
                if fg_mask.mean() != raw_mean:
                    print(f"[mask-bg] raw fg mask was {raw_mean:.2%}", flush=True)
            if args.largest_component:
                comp = largest_depth_component(
                    hit.reshape(-1) if hit.ndim > 1 else hit,
                    t.astype(np.float32), H, W, args.rel_depth_gap)
                fg_mask = comp if fg_mask is None else (fg_mask & comp)
            dump_buffers(
                buf_stem, hit, normals,
                pts.detach().cpu().numpy().astype(np.float32),
                views["c2w"][vi].numpy().astype(np.float64), H, W,
                ss=args.supersample, fg_mask=fg_mask,
                smooth_sigma=args.smooth_sigma,
                fill_holes=not args.no_fill_holes,
                fill_mask_misses=args.fill_mask_misses,
            )
            if args.gt_photo:
                if bbox is None:
                    print("[gt-photo] skipped: needs --supersample>1 (fg-bbox "
                          "crop is only defined for the supersample grid)",
                          flush=True)
                else:
                    import imageio.v2 as imageio
                    from PIL import Image
                    r0, r1, c0, c1 = bbox
                    rgb = views["images"][vi].numpy()            # (Himg,Wimg,3) float 0-1
                    rgb_crop = rgb[r0:r1, c0:c1]
                    # Match the (now box-averaged) phong/normals output size so the
                    # GT photo stays pixel-aligned for side-by-side comparison.
                    gw, gh = W // args.supersample, H // args.supersample
                    if fg_mask is not None and fg_mask.shape == (H, W):
                        if args.supersample > 1:
                            msk = fg_mask.reshape(
                                gh, args.supersample, gw, args.supersample
                            ).any(axis=(1, 3))
                        else:
                            msk = fg_mask
                    else:
                        msk = views["masks"][vi].numpy().astype(bool)[r0:r1, c0:c1]
                        msk = clean_display_mask(
                            msk,
                            fill_holes=not args.no_mask_fill_holes,
                            dilate_px=max(0, args.mask_dilate_px),
                        )
                    masked = np.where(msk[..., None], rgb_crop, 1.0)
                    gt_img = Image.fromarray(
                        np.clip(masked * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    ).resize((gw, gh), Image.Resampling.LANCZOS)
                    gt_path = buf_stem.with_name(buf_stem.name + "_gt.png")
                    imageio.imwrite(gt_path, np.asarray(gt_img))
                    print(f"saved GT photo {gt_path}  (masked, fg-bbox "
                          f"{bbox}, →{gw}×{gh})", flush=True)

        verts_n, vnorm, faces = build_screen_mesh(
            hit, pts.detach().cpu().numpy().astype(np.float32),
            t.astype(np.float32), normals, H, W,
            args.rel_depth_gap, args.grazing_cos,
            dirs.detach().cpu().numpy().astype(np.float32),
        )

        # normalized → DTU world frame so tools/render_blender.py's S_inv lands us back
        v_h = np.concatenate([verts_n.astype(np.float64), np.ones((len(verts_n), 1))], axis=1)
        verts_world = (v_h @ S.T)[:, :3].astype(np.float32)
        vnorm_world = vnorm / np.linalg.norm(vnorm, axis=1, keepdims=True).clip(min=1e-9)

        if args.out and len(view_list) == 1:
            out = args.out
        else:
            out = args.run_dir / "ckpt" / f"sphere_traced_screen_view{vi:02d}.ply"
        write_ply(out, verts_world, vnorm_world.astype(np.float32), faces)


if __name__ == "__main__":
    main()
