#!/usr/bin/env python3
"""Diagnose why sphere-traced SDF has holes that the visual hull does not.

Produces a multi-panel PNG for one training view showing:
  A) Hit map (green=hit, red=miss)
  B) sdf_min map — closest the tracer got along each ray (log scale)
  C) |∇f| map at the ray origin — near-zero means flat SDF region
  D) SDF profiles f(o + t*d) along ~10 sampled hole rays — do zero crossings exist?
  E) SDF profiles along ~5 hit rays for reference
  F) Histogram of sdf_min values, split by hit/miss

Usage:
    python diagnose_holes.py <run_dir> [--view 0] [--down 4] [--t-steps 400] [--out diagnose.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model(run_dir: Path, device: str, ckpt_name: str | None = None):
    from lip_tracer.train import load_config_json
    cfg = load_config_json(run_dir / "config.json")
    # checkpoints may live at run root or under ckpt/
    cands = []
    if ckpt_name:
        cands = [run_dir / ckpt_name, run_dir / "ckpt" / ckpt_name]
    else:
        for nm in ("checkpoint_best_loss.pt", "checkpoint_final.pt",
                   "checkpoint.pt", "checkpoint_latest.pt"):
            cands += [run_dir / nm, run_dir / "ckpt" / nm]
    ckpt_path = next((p for p in cands if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint found under {run_dir} (tried {[str(c) for c in cands]})")
    print(f"[diagnose] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from lip_tracer.model import make_model
    mc = cfg.model
    f = make_model(hidden=mc.hidden, depth=mc.depth, group_size=mc.group_size,
                   activation=mc.activation, input_encoding=mc.input_encoding,
                   multires=mc.multires, architecture=getattr(mc, "architecture", "cpl"))
    f.load_state_dict(ckpt["f"], strict=False)
    f = f.to(device).eval()
    return f, cfg


def _make_rays(c2w: np.ndarray, K: np.ndarray, H: int, W: int, down: int, device: str):
    """Return ray origins (H*W, 3) and directions (H*W, 3) for a camera."""
    Hd, Wd = H // down, W // down
    ys, xs = np.meshgrid(np.arange(Hd) * down + down / 2 - 0.5,
                         np.arange(Wd) * down + down / 2 - 0.5, indexing="ij")
    dirs_cam = np.stack([(xs - K[0, 2]) / K[0, 0],
                         (ys - K[1, 2]) / K[1, 1],
                         np.ones_like(xs)], axis=-1)  # (Hd, Wd, 3)
    R = c2w[:3, :3]
    dirs_world = (dirs_cam.reshape(-1, 3) @ R.T).astype(np.float32)
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8
    origin = np.broadcast_to(c2w[:3, 3].astype(np.float32), (Hd * Wd, 3)).copy()
    o = torch.from_numpy(origin).to(device)
    d = torch.from_numpy(dirs_world).to(device)
    return o, d, Hd, Wd


def _trace_with_diagnostics(f, o, d, cfg_trace, t_steps: int, device: str, chunk: int = 4096):
    """Run sphere tracing + collect sdf_min and grad-norm at origin."""
    from lip_tracer.config import TraceConfig
    N = o.shape[0]
    hit     = torch.zeros(N, dtype=torch.bool, device=device)
    t_out   = torch.zeros(N, device=device)
    sdf_min = torch.full((N,), float("inf"), device=device)

    with torch.no_grad():
        for i in range(0, N, chunk):
            o_b = o[i:i + chunk]; d_b = d[i:i + chunk]
            B = o_b.shape[0]
            t  = torch.zeros(B, device=device)
            sm = torch.full((B,), float("inf"), device=device)
            cv = torch.zeros(B, dtype=torch.bool, device=device)
            sdf_last = torch.zeros(B, device=device)
            for _ in range(cfg_trace.iters):
                esc = t >= cfg_trace.t_far
                if not (~(cv | esc)).any(): break
                sdf_last = f(o_b + t.unsqueeze(-1) * d_b)
                sm = torch.minimum(sm, sdf_last.abs())
                cv  = sdf_last.abs() < cfg_trace.eps
                t   = t + torch.where(cv | esc, torch.zeros_like(t), sdf_last)
            sdf_last = f(o_b + t.unsqueeze(-1) * d_b)
            sm = torch.minimum(sm, sdf_last.abs())
            h  = (sdf_last.abs() < cfg_trace.eps) & (t < cfg_trace.t_far) & (t >= 0)
            hit[i:i + chunk]     = h
            t_out[i:i + chunk]   = t
            sdf_min[i:i + chunk] = sm

    # gradient norm at ray origin — chunked to avoid keeping all activations in RAM
    grad_chunks = []
    for i in range(0, N, chunk):
        o_b = o[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(o_b).sum(), o_b)[0].detach()
        grad_chunks.append(g.norm(dim=-1).cpu())
    grad_norm = torch.cat(grad_chunks)

    return hit.cpu(), t_out.cpu(), sdf_min.detach().cpu(), grad_norm


def _ray_sdf_profiles(f, o, d, t_far: float, t_steps: int, indices, device: str):
    """Sample f along selected rays at t_steps uniform steps. Returns (len(indices), t_steps)."""
    ts = torch.linspace(0.0, t_far, t_steps, device=device)  # (T,)
    profiles = []
    with torch.no_grad():
        for idx in indices:
            o_r = o[idx:idx + 1].to(device)
            d_r = d[idx:idx + 1].to(device)
            pts = o_r + ts.unsqueeze(-1) * d_r  # (T, 3)
            profiles.append(f(pts).cpu().numpy())
    return np.stack(profiles), ts.cpu().numpy()  # (R, T), (T,)


# ── ICLR hypothesis figure ────────────────────────────────────────────────────
# Two falsifiable hypotheses for the persistent back/top holes:
#   H1 (textureless): hole pixels sit on low-image-gradient regions of the GT.
#   H2 (grazing / low-coverage): the surface there is seen front-facing by few
#       training cameras.
# We quantify each per foreground pixel and split the distribution by
# hole vs. filled, reporting a rank-AUC (Mann-Whitney) as the effect size.

def _box_mean(a: np.ndarray, win: int) -> np.ndarray:
    """Mean over a (2*win+1) square window via an integral image (no deps)."""
    if win <= 0:
        return a
    H, W = a.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a.astype(np.float64), 0), 1)
    out = np.empty_like(a, dtype=np.float64)
    for y in range(H):
        y0, y1 = max(0, y - win), min(H, y + win + 1)
        for x in range(W):
            x0, x1 = max(0, x - win), min(W, x + win + 1)
            s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            out[y, x] = s / ((y1 - y0) * (x1 - x0))
    return out.astype(np.float32)


def _texture_energy(gray: np.ndarray, win: int) -> np.ndarray:
    """Local mean gradient magnitude of a grayscale image — high=textured."""
    gy, gx = np.gradient(gray.astype(np.float32))
    return _box_mean(np.hypot(gx, gy), win)


def _dense_argmin_t(f, o, d, t_far: float, steps: int, device: str, chunk: int = 8192):
    """For every ray: t and |f| at closest approach to the zero set, plus whether
    the SDF changes sign along the ray (a zero-crossing ⇒ a surface really exists
    there — a 'miss' on such a ray is a tracer overshoot, not deleted geometry)."""
    ts = torch.linspace(0.0, t_far, steps, device=device)
    N = o.shape[0]
    t_best = torch.zeros(N); f_best = torch.full((N,), float("inf"))
    cross = torch.zeros(N, dtype=torch.bool)
    with torch.no_grad():
        for i in range(0, N, chunk):
            o_b = o[i:i + chunk]; d_b = d[i:i + chunk]
            B = o_b.shape[0]
            bt = torch.zeros(B, device=device)
            ba = torch.full((B,), float("inf"), device=device)
            has_neg = torch.zeros(B, dtype=torch.bool, device=device)
            has_pos = torch.zeros(B, dtype=torch.bool, device=device)
            for s in range(steps):
                fv = f(o_b + ts[s] * d_b)
                a = fv.abs()
                better = a < ba
                ba = torch.where(better, a, ba)
                bt = torch.where(better, torch.full_like(bt, ts[s].item()), bt)
                has_neg |= fv < 0
                has_pos |= fv > 0
            t_best[i:i + chunk] = bt.cpu(); f_best[i:i + chunk] = ba.cpu()
            cross[i:i + chunk] = (has_neg & has_pos).cpu()
    return t_best, f_best, cross


def _normals_at(f, pts: torch.Tensor, device: str, chunk: int = 8192) -> torch.Tensor:
    outs = []
    for i in range(0, pts.shape[0], chunk):
        p = pts[i:i + chunk].to(device).detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(p).sum(), p)[0].detach()
        n = g / (g.norm(dim=-1, keepdim=True) + 1e-9)
        outs.append(n.cpu())
    return torch.cat(outs)


def _angular_coverage(pts: torch.Tensor, normals: torch.Tensor,
                      c2ws_all, Ks_all, H: int, W: int, cos_thr: float):
    """Per point, over all cameras that see it in-frame & in-front:
      cnt    = # cameras with view·normal > cos_thr  (front-facing coverage)
      best   = max view·normal  (1=head-on, →0=only grazing views available)
    `best` is the grazing-severity measure; low best ⇒ surface only ever skimmed.
    """
    M = pts.shape[0]
    cnt = torch.zeros(M)
    best = torch.full((M,), -1.0)
    for c2w_i, K_i in zip(c2ws_all, Ks_all):
        c2w = c2w_i if torch.is_tensor(c2w_i) else torch.tensor(c2w_i, dtype=torch.float32)
        K = torch.tensor(K_i, dtype=torch.float32) if not torch.is_tensor(K_i) else K_i.float()
        w2c = torch.linalg.inv(c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        xc = (R @ pts.T).T + t
        z = xc[:, 2]
        uv = (K @ xc.T).T
        uv = uv[:, :2] / uv[:, 2:3].clamp(min=1e-6)
        infrust = (z > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        vdir = c2w[:3, 3] - pts
        vdir = vdir / (vdir.norm(dim=-1, keepdim=True) + 1e-9)
        cosv = (vdir * normals).sum(-1)
        cnt += (infrust & (cosv > cos_thr)).float()
        best = torch.where(infrust, torch.maximum(best, cosv), best)
    return cnt.numpy(), best.numpy()


def _erode_mask(fg2d: np.ndarray, k: int) -> np.ndarray:
    """Binary erosion by a (2k+1) box — interior pixels only (drops silhouette edge)."""
    if k <= 0:
        return fg2d
    frac = _box_mean(fg2d.astype(np.float32), k)
    return frac >= 1.0 - 1e-6


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC that `pos` scores higher than `neg` (Mann-Whitney U / n_pos n_neg)."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnts)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnts)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _median(a):
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def _verdict(auc):
    return ("SUPPORTS" if auc >= 0.62 else "REFUTES" if auc <= 0.38 else "no signal")


def _view_metrics(f, cfg, v, vi: int, steps: int, cos_thr: float, tex_win: int, device: str):
    """All per-view hole diagnostics for one camera. Returns a dict of flat
    (Hd*Wd) maps + scalar AUC/median summaries. Shared by single- and multi-view."""
    c2w = v["c2w"][vi].numpy(); K = v["K"][vi].numpy()
    H, W = v["H"], v["W"]
    img = v["images"][vi].numpy()
    mask = v["masks"][vi].numpy() > 0.5
    c2ws_all = [v["c2w"][i] for i in range(v["c2w"].shape[0])]
    Ks_all   = [v["K"][i].numpy() for i in range(v["K"].shape[0])]

    o, d, Hd, Wd = _make_rays(c2w, K, H, W, 1, device)
    fg = mask[:Hd, :Wd].reshape(-1)

    hit, _, _, _ = _trace_with_diagnostics(f, o, d, cfg.trace, steps, device)
    hit = hit.numpy()
    hole = fg & (~hit); filled = fg & hit

    fg_t = torch.from_numpy(np.where(fg)[0])
    o_fg, d_fg = o[fg_t], d[fg_t]
    t_proxy, _, cross_fg = _dense_argmin_t(f, o_fg, d_fg, cfg.trace.t_far, steps, device)
    pts = (o_fg.cpu() + t_proxy.unsqueeze(-1) * d_fg.cpu())
    normals = _normals_at(f, pts, device)
    cam_dir = torch.tensor(c2w[:3, 3], dtype=torch.float32) - pts
    normals[(cam_dir * normals).sum(-1) < 0] *= -1
    inc_fg = (-(d_fg.cpu() * normals).sum(-1)).clamp(-1, 1).numpy()
    cov_fg, graze_fg = _angular_coverage(pts, normals, c2ws_all, Ks_all, H, W, cos_thr)

    tex = _texture_energy(img[:Hd, :Wd].mean(-1), tex_win).reshape(-1)
    cov = np.full(Hd * Wd, np.nan); cov[fg] = cov_fg
    graze = np.full(Hd * Wd, np.nan); graze[fg] = graze_fg
    inc = np.full(Hd * Wd, np.nan); inc[fg] = inc_fg
    cross = np.zeros(Hd * Wd, dtype=bool); cross[fg] = cross_fg.numpy()

    interior = _erode_mask(fg.reshape(Hd, Wd), tex_win + 2).reshape(-1)
    hole_in = hole & interior; filled_in = filled & interior

    return dict(
        vi=vi, Hd=Hd, Wd=Wd, img=img[:Hd, :Wd], fg=fg,
        hole_in=hole_in, filled_in=filled_in,
        tex=tex, inc=inc, graze=graze, cov=cov, cross=cross,
        auc_tex=_auc(-tex[hole_in], -tex[filled_in]),
        auc_grz=_auc(-graze[hole_in], -graze[filled_in]),
        auc_inc=_auc(-inc[hole_in], -inc[filled_in]),
        auc_cov=_auc(-cov[hole_in], -cov[filled_in]),
        cross_hole_frac=(float(cross[hole_in].mean()) if hole_in.sum() else float("nan")),
        med_tex_h=_median(tex[hole_in]), med_tex_f=_median(tex[filled_in]),
        med_inc_h=_median(inc[hole_in]), med_inc_f=_median(inc[filled_in]),
    )


def _plot_columns(cfg, run_dir, mets, col_titles, out, suptitle):
    """Tile per-column hole diagnostics (rows: GT+holes / render-grazing / verdict).
    Columns may be different views OR the same view at different checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(mets)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(3, n, figsize=(3.3 * n, 9.4), facecolor="white", squeeze=False)
    for j, (m, ctitle) in enumerate(zip(mets, col_titles)):
        Hd, Wd = m["Hd"], m["Wd"]
        holec = m["hole_in"].reshape(Hd, Wd).astype(float)
        fg_img = m["fg"].reshape(Hd, Wd)

        a = ax[0, j]
        a.imshow(m["img"])
        a.imshow(np.dstack([np.ones_like(holec), np.zeros_like(holec), np.zeros_like(holec), holec * 0.6]))
        a.set_title(f"{ctitle}\n{m['hole_in'].sum():,} hole px"); a.axis("off")

        a = ax[1, j]
        incmap = np.where(fg_img, m["inc"].reshape(Hd, Wd), np.nan)
        im = a.imshow(incmap, cmap="magma", vmin=0, vmax=1)
        a.contour(holec, levels=[0.5], colors="cyan", linewidths=0.6)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        a.set_title("render incidence -d·n (0=grazing)"); a.axis("off")

        a = ax[2, j]; a.axis("off")
        txt = (
            f"interior holes: {m['hole_in'].sum():,}\n"
            f"surface exists: {m['cross_hole_frac']:.0%} of holes\n"
            f"  (zero-crossing along ray)\n\n"
            f"H1 textureless   AUC {m['auc_tex']:.2f}\n"
            f"   [{_verdict(m['auc_tex'])}]  tex h/f "
            f"{m['med_tex_h']:.3f}/{m['med_tex_f']:.3f}\n"
            f"H2 cam-grazing   AUC {m['auc_grz']:.2f}\n"
            f"   [{_verdict(m['auc_grz'])}]\n"
            f"H2 cam-coverage  AUC {m['auc_cov']:.2f}\n"
            f"H2 render-graze  AUC {m['auc_inc']:.2f}\n"
            f"   [{_verdict(m['auc_inc'])}]  inc h/f "
            f"{m['med_inc_h']:.2f}/{m['med_inc_f']:.2f}"
        )
        a.text(0.02, 0.98, txt, transform=a.transAxes, va="top", ha="left",
               family="monospace", fontsize=7.5, linespacing=1.45)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[iclr] saved → {out}")


def iclr_multiview(run_dir: Path, views, down: int, steps: int,
                   cos_thr: float, tex_win: int, out: Path, device: str,
                   ckpt_name: str | None):
    """Compact figure: one column per view at a single checkpoint."""
    from lip_tracer.data import load_views
    f, cfg = _load_model(run_dir, device, ckpt_name)
    print(f"[iclr] loading views (down={down}) …")
    v = load_views(cfg.scene, down=down)
    nview = v["c2w"].shape[0]
    views = [min(int(x), nview - 1) for x in views]
    mets = []
    for vi in views:
        print(f"[iclr] view {vi} …")
        mets.append(_view_metrics(f, cfg, v, vi, steps, cos_thr, tex_win, device))
    suptitle = (
        f"Back/top holes across views — {run_dir.name}\n"
        f"input_enc={cfg.model.input_encoding}  lip_mode={cfg.model.lipschitz_mode}  "
        f"w_eikonal={cfg.train.w_eikonal}   ·   interior-only, AUC>0.62 supports / <0.38 refutes")
    _plot_columns(cfg, run_dir, mets, [f"view {m['vi']}" for m in mets], out, suptitle)
    for m in mets:
        print(f"[iclr] v{m['vi']:>3}: holes={m['hole_in'].sum():>6,}  surf_exists={m['cross_hole_frac']:.0%}  "
              f"tex={m['auc_tex']:.2f} camgrz={m['auc_grz']:.2f} cov={m['auc_cov']:.2f} rendergrz={m['auc_inc']:.2f}")


def iclr_compare_ckpts(run_dir: Path, ckpts, view: int, down: int, steps: int,
                       cos_thr: float, tex_win: int, out: Path, device: str):
    """Same view at several checkpoints — does 'surface exists' start high
    (tracer overshoot) and collapse (geometry deleted) over training?"""
    from lip_tracer.data import load_views
    v = None; mets = []; labels = []
    for ck in ckpts:
        f, cfg = _load_model(run_dir, device, ck)
        if v is None:
            print(f"[iclr] loading views (down={down}) …")
            v = load_views(cfg.scene, down=down)
            view = min(view, v["c2w"].shape[0] - 1)
        print(f"[iclr] {ck} @ view {view} …")
        mets.append(_view_metrics(f, cfg, v, view, steps, cos_thr, tex_win, device))
        labels.append(ck.replace("checkpoint_", "").replace(".pt", ""))
    suptitle = (
        f"Hole evolution over training — {run_dir.name}  (view {view})\n"
        f"input_enc={cfg.model.input_encoding}  lip_mode={cfg.model.lipschitz_mode}  "
        f"w_eikonal={cfg.train.w_eikonal}   ·   surface-exists ↓ ⇒ geometry being deleted")
    _plot_columns(cfg, run_dir, mets, labels, out, suptitle)
    for ck, m in zip(labels, mets):
        print(f"[iclr] {ck:>10}: holes={m['hole_in'].sum():>6,}  surf_exists={m['cross_hole_frac']:.0%}  "
              f"tex={m['auc_tex']:.2f} camgrz={m['auc_grz']:.2f} cov={m['auc_cov']:.2f} rendergrz={m['auc_inc']:.2f}")


def iclr_lost_zncc(run_dir: Path, before_ckpt: str, after_ckpt: str, view: int,
                   down: int, steps: int, topk: int, out: Path, device: str):
    """For one view: M_lost = H_before · (1 − H_after) (pixels eroded between the
    two checkpoints). At the BEFORE surface, compute raw top-k ZNCC (ncc_min=0)
    and colour each lost pixel by how matchable it WAS before it died:
        red z≤0 · orange 0<z≤.05 · yellow .05<z≤.1 · green z>.1
    Green-dominant ⇒ matchable surface was deleted anyway (free-lunch pathology).
    Red-dominant   ⇒ unmatchable surface correctly removed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lip_tracer.data import load_views, precompute_alt_cameras
    from lip_tracer.loss import pmvs_ncc_loss

    fb, cfg = _load_model(run_dir, device, before_ckpt)
    fa, _   = _load_model(run_dir, device, after_ckpt)
    tc = cfg.train
    print(f"[lost] loading views (down={down}) …")
    v = load_views(cfg.scene, down=down)
    vi = min(view, v["c2w"].shape[0] - 1)
    H, W = v["H"], v["W"]
    c2w = v["c2w"][vi].numpy(); K = v["K"][vi].numpy()
    img = v["images"][vi].numpy()
    mask = v["masks"][vi].numpy() > 0.5 if "masks" in v else np.ones((H, W), bool)

    images = v["images"].to(device).float()
    if images.max() > 1.5:
        images = images / 255.0
    K_all = v["K"].to(device).float()
    w2c_all = torch.linalg.inv(v["c2w"]).to(device).float()
    alt_nn = precompute_alt_cameras(v, tc.n_alt)              # (V, n_alt)

    o, d, Hd, Wd = _make_rays(c2w, K, H, W, 1, device)
    print("[lost] tracing before/after …")
    hb, tb, _, _ = _trace_with_diagnostics(fb, o, d, cfg.trace, steps, device)
    ha, _,  _, _ = _trace_with_diagnostics(fa, o, d, cfg.trace, steps, device)
    fg = torch.from_numpy(mask[:Hd, :Wd].reshape(-1))
    lost = hb & (~ha) & fg                                    # M_lost = H_b(1-H_a)
    print(f"[lost] H_before={int(hb.sum()):,}  H_after={int(ha.sum()):,}  lost={int(lost.sum()):,}")

    lost_idx = torch.where(lost)[0]
    x_lost = (o[lost_idx].cpu() + tb[lost_idx].unsqueeze(-1) * d[lost_idx].cpu())
    n_lost = _normals_at(fb, x_lost, device)
    B = x_lost.shape[0]

    # raw top-k ZNCC at the BEFORE surface across the n_alt nearest views (no rejection)
    alt = alt_nn[vi].tolist()
    zstack = torch.full((len(alt), B), float("nan"))
    x_d = x_lost.to(device); n_d = n_lost.to(device)
    for ai, b in enumerate(alt):
        va = torch.full((B,), vi, dtype=torch.long, device=device)
        vb = torch.full((B,), int(b), dtype=torch.long, device=device)
        _, _, _, zf = pmvs_ncc_loss(
            images, x_d, n_d, va, vb, K_all, w2c_all, H, W,
            patch=tc.ncc_patch, half_pix=tc.ncc_half_pix, sample_mode=tc.sample_mode,
            gaussian_sigma=tc.gaussian_sigma, gaussian_radius=tc.gaussian_radius,
            ncc_min=-2.0, return_full=True, ncc_color=tc.ncc_color)
        zstack[ai] = zf.detach().cpu()
    # top-k over valid views per point
    zs = zstack.clone(); zs[torch.isnan(zs)] = -float("inf")
    zs_sorted, _ = zs.sort(dim=0, descending=True)
    kk = max(1, min(topk, len(alt)))
    topv = zs_sorted[:kk]                                     # (kk, B)
    valid = torch.isfinite(topv)
    z = torch.where(valid.any(0),
                    (topv * valid).sum(0) / valid.sum(0).clamp(min=1),
                    torch.full((B,), float("nan")))
    z = z.numpy()

    has_z = np.isfinite(z)
    zc = z[has_z]
    frac_gt0  = float((zc > 0.0).mean())   if len(zc) else float("nan")
    frac_gt05 = float((zc > 0.05).mean())  if len(zc) else float("nan")
    frac_gt1  = float((zc > 0.10).mean())  if len(zc) else float("nan")
    mean_z    = float(zc.mean())           if len(zc) else float("nan")

    # category per lost pixel: 0 red ≤0, 1 orange, 2 yellow, 3 green, -1 gray(no view)
    cat = np.full(B, -1, np.int64)
    cat[has_z & (z <= 0)]                    = 0
    cat[has_z & (z > 0)   & (z <= 0.05)]     = 1
    cat[has_z & (z > 0.05) & (z <= 0.10)]    = 2
    cat[has_z & (z > 0.10)]                  = 3
    colors = {0: (0.90, 0.10, 0.10), 1: (1.00, 0.55, 0.0),
              2: (0.95, 0.90, 0.15), 3: (0.20, 0.80, 0.20), -1: (0.5, 0.5, 0.5)}

    overlay = np.zeros((Hd * Wd, 4), np.float32)
    li = lost_idx.numpy()
    for c, rgb in colors.items():
        sel = li[cat == c]
        overlay[sel, :3] = rgb; overlay[sel, 3] = 0.9
    overlay = overlay.reshape(Hd, Wd, 4)

    plt.rcParams.update({"font.family": "sans-serif"})
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.2), facecolor="white")
    ax.imshow(img[:Hd, :Wd]); ax.imshow(overlay); ax.axis("off")
    import matplotlib.patches as mpatches
    leg = [mpatches.Patch(color=colors[3], label="z>0.1 (matchable)"),
           mpatches.Patch(color=colors[2], label="0.05<z≤0.1"),
           mpatches.Patch(color=colors[1], label="0<z≤0.05"),
           mpatches.Patch(color=colors[0], label="z≤0 (unmatchable)"),
           mpatches.Patch(color=colors[-1], label="no valid view")]
    ax.legend(handles=leg, loc="lower right", fontsize=7, framealpha=0.85)
    ax.set_title(
        f"Eroded surface, coloured by ZNCC BEFORE deletion — view {vi}\n"
        f"{before_ckpt.replace('checkpoint_','').replace('.pt','')} → "
        f"{after_ckpt.replace('checkpoint_','').replace('.pt','')}   "
        f"lost={B:,} px (top-{kk} ZNCC, ncc_min=0)\n"
        f"frac z>0 = {frac_gt0:.2f}   z>0.05 = {frac_gt05:.2f}   "
        f"z>0.1 = {frac_gt1:.2f}   mean_z = {mean_z:.3f}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[lost] saved → {out}")
    print(f"[lost] frac_lost z>0={frac_gt0:.3f}  z>0.05={frac_gt05:.3f}  "
          f"z>0.1={frac_gt1:.3f}  mean_z_lost={mean_z:.3f}  (n={len(zc):,}/{B:,})")


def iclr_figure(run_dir: Path, view: int, down: int, steps: int,
                cos_thr: float, tex_win: int, out: Path, device: str,
                ckpt_name: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lip_tracer.data import load_views

    f, cfg = _load_model(run_dir, device, ckpt_name)
    print(f"[iclr] loading views (down={down}) …")
    v = load_views(cfg.scene, down=down)
    vi = min(view, v["c2w"].shape[0] - 1)
    m = _view_metrics(f, cfg, v, vi, steps, cos_thr, tex_win, device)
    Hd, Wd = m["Hd"], m["Wd"]
    img = m["img"]; fg = m["fg"]
    hole_in = m["hole_in"]; filled_in = m["filled_in"]
    tex, inc, graze, cov, cross = m["tex"], m["inc"], m["graze"], m["cov"], m["cross"]
    interior = (_erode_mask(fg.reshape(Hd, Wd), tex_win + 2).reshape(-1))
    tex_hole, tex_fill = tex[hole_in], tex[filled_in]
    inc_hole, inc_fill = inc[hole_in], inc[filled_in]
    grz_hole, grz_fill = graze[hole_in], graze[filled_in]
    cov_hole, cov_fill = cov[hole_in], cov[filled_in]
    auc_tex, auc_grz, auc_inc, auc_cov = m["auc_tex"], m["auc_grz"], m["auc_inc"], m["auc_cov"]
    cross_hole_frac = m["cross_hole_frac"]
    med = _median
    verdict = _verdict

    # ── plot: 2 rows × 3 ────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.4), facecolor="white")
    hole_img = (hole_in).reshape(Hd, Wd)
    fg_img = fg.reshape(Hd, Wd)
    holec = hole_img.astype(float)

    # (0,0) GT + hole overlay (interior holes only — what the test scores)
    a = ax[0, 0]
    a.imshow(img[:Hd, :Wd]); a.imshow(np.dstack([np.ones_like(holec), np.zeros_like(holec),
              np.zeros_like(holec), holec * 0.6]))
    a.set_title(f"GT view {vi} — interior holes (red)\n"
                f"{hole_in.sum():,} hole / {filled_in.sum():,} filled px"); a.axis("off")

    # (0,1) texture map
    a = ax[0, 1]
    texmap = np.where(fg_img, tex.reshape(Hd, Wd), np.nan)
    im = a.imshow(texmap, cmap="viridis", vmax=np.nanpercentile(texmap, 98))
    a.contour(holec, levels=[0.5], colors="red", linewidths=0.7)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    a.set_title("H1  GT texture energy (|∇I|)\nred contour = holes"); a.axis("off")

    # (0,2) render-ray grazing-incidence map (the tracer-overshoot test)
    a = ax[0, 2]
    incmap = np.where(fg_img, inc.reshape(Hd, Wd), np.nan)
    im = a.imshow(incmap, cmap="magma", vmin=0, vmax=1)
    a.contour(holec, levels=[0.5], colors="cyan", linewidths=0.7)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    a.set_title("H2  render-ray incidence -d·n (0=grazing)\ncyan contour = holes"); a.axis("off")

    # (1,0) texture distribution
    a = ax[1, 0]
    bmax = max(np.nanpercentile(tex[interior & fg], 99), 1e-3)
    bins = np.linspace(0, bmax, 50)
    a.hist(tex_fill, bins=bins, density=True, color="#2c7fb8", alpha=0.7, label=f"filled (med {med(tex_fill):.3f})")
    a.hist(tex_hole, bins=bins, density=True, color="#d7301f", alpha=0.7, label=f"hole (med {med(tex_hole):.3f})")
    a.set_xlabel("texture energy |∇I|  (interior px)"); a.set_ylabel("density")
    a.set_title(f"H1 textureless: {verdict(auc_tex)}\nAUC(low-tex→hole) = {auc_tex:.3f}")
    a.legend(frameon=False, fontsize=7)

    # (1,1) render-ray incidence distribution — the real discriminator
    a = ax[1, 1]
    gbins = np.linspace(0, 1, 40)
    a.hist(inc_fill[np.isfinite(inc_fill)], bins=gbins, density=True, color="#2c7fb8", alpha=0.7,
           label=f"filled (med {med(inc_fill):.2f})")
    a.hist(inc_hole[np.isfinite(inc_hole)], bins=gbins, density=True, color="#d7301f", alpha=0.7,
           label=f"hole (med {med(inc_hole):.2f})")
    a.set_xlabel("render-ray incidence -d·n  (low=grazing)"); a.set_ylabel("density")
    a.set_title(f"H2 render grazing: {verdict(auc_inc)}\nAUC(grazing→hole) = {auc_inc:.3f}  "
                f"[cam-graze {auc_grz:.2f}, cov {auc_cov:.2f}]")
    a.legend(frameon=False, fontsize=7)

    # (1,2) is the surface actually there? zero-crossing fraction
    a = ax[1, 2]
    fr_h = cross[hole_in].mean() if hole_in.sum() else 0.0
    fr_f = cross[filled_in].mean() if filled_in.sum() else 0.0
    a.bar([0, 1], [fr_f, fr_h], color=["#2c7fb8", "#d7301f"], width=0.6)
    a.set_xticks([0, 1]); a.set_xticklabels(["filled", "hole"])
    a.set_ylim(0, 1.05); a.set_ylabel("fraction of rays with a zero-crossing")
    for x, fr in [(0, fr_f), (1, fr_h)]:
        a.text(x, fr + 0.02, f"{fr:.0%}", ha="center", fontsize=9, fontweight="bold")
    a.set_title(f"Surface EXISTS along {fr_h:.0%} of hole rays\n→ holes are tracer overshoots, not deleted geometry")

    fig.suptitle(
        f"What actually makes the back/top holes?  {run_dir.name}  (view {vi}, {Hd}×{Wd})\n"
        f"input_enc={cfg.model.input_encoding}  lip_mode={cfg.model.lipschitz_mode}  "
        f"w_eikonal={cfg.train.w_eikonal}   ·   interior-only test",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[iclr] saved → {out}")
    print(f"[iclr] H1 texture       AUC={auc_tex:.3f} [{verdict(auc_tex)}]  hole={med(tex_hole):.4f} vs filled={med(tex_fill):.4f}")
    print(f"[iclr] H2 cam-grazing   AUC={auc_grz:.3f} [{verdict(auc_grz)}]  hole={med(grz_hole):.3f} vs filled={med(grz_fill):.3f}")
    print(f"[iclr] H2 cam-coverage  AUC={auc_cov:.3f}  hole={med(cov_hole):.1f} vs filled={med(cov_fill):.1f}")
    print(f"[iclr] H2 render-grazing AUC={auc_inc:.3f} [{verdict(auc_inc)}]  hole={med(inc_hole):.3f} vs filled={med(inc_fill):.3f}")
    print(f"[iclr] surface exists along {cross_hole_frac:.0%} of hole rays (zero-crossing)")


# ── main ─────────────────────────────────────────────────────────────────────

def _color_variance_at_holes(
    o: torch.Tensor, d: torch.Tensor,
    sdf_min: torch.Tensor, hit: torch.Tensor,
    c2ws_all, Ks_all, img_paths,
    H: int, W: int, ref_vi: int, n_views: int,
) -> np.ndarray:
    """Per-hole-pixel colour std across n_views other cameras.

    Back-projects each miss pixel using sdf_min as a proxy depth, then
    projects that 3D point into n_views randomly chosen other cameras and
    samples the RGB colour.  Returns an (N,) float32 array with NaN for
    hit pixels.

    Low std  (<0.05) → colours agree across views → topology/init problem.
                       fg mask loss should be enough.
    High std (>0.15) → colours disagree → view-dependent / specular.
                       fg loss will fight photo loss; depth prior needed.
    """
    from PIL import Image as _PIL

    N = o.shape[0]
    miss_mask = ~hit
    if not miss_mask.any() or not img_paths:
        return np.full(N, np.nan, dtype=np.float32)

    # proxy 3D points for hole rays
    x3d_miss = (o[miss_mask] + sdf_min[miss_mask].unsqueeze(-1) * d[miss_mask]).cpu()  # (M, 3)
    M = x3d_miss.shape[0]

    # pick n_views cameras excluding the reference view
    cands = [i for i in range(len(img_paths)) if i != ref_vi]
    chosen = np.random.default_rng(0).choice(cands, size=min(n_views, len(cands)), replace=False)

    colors_per_view = []
    for vi in chosen:
        c2w = c2ws_all[vi] if isinstance(c2ws_all[vi], torch.Tensor) else torch.tensor(c2ws_all[vi], dtype=torch.float32)
        K   = torch.tensor(Ks_all[vi], dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        R, t_cam = w2c[:3, :3], w2c[:3, 3]

        xc  = (R @ x3d_miss.T).T + t_cam          # (M, 3)
        uv_h = (K @ xc.T).T                        # (M, 3)
        uv   = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)

        valid = ((xc[:, 2] > 0.01)
                 & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < H)).numpy().astype(bool)

        img = np.array(_PIL.open(img_paths[vi]).convert("RGB"), dtype=np.float32) / 255.0
        colors = np.full((M, 3), np.nan, dtype=np.float32)
        if valid.any():
            ix = uv[:, 0].numpy()[valid].clip(0, W - 1).astype(np.int32)
            iy = uv[:, 1].numpy()[valid].clip(0, H - 1).astype(np.int32)
            colors[valid] = img[iy, ix]
        colors_per_view.append(colors)

    stack = np.stack(colors_per_view, axis=0)           # (V, M, 3)
    color_std = np.nanstd(stack, axis=0).mean(axis=-1)  # (M,) — mean over RGB

    result = np.full(N, np.nan, dtype=np.float32)
    result[miss_mask.numpy()] = color_std
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--view",    type=int,   default=0)
    ap.add_argument("--down",    type=int,   default=4)
    ap.add_argument("--t-steps", type=int,   default=400)
    ap.add_argument("--n-hole-rays", type=int, default=12)
    ap.add_argument("--n-hit-rays",  type=int, default=5)
    ap.add_argument("--color-views", type=int, default=6,
                    help="how many other cameras to use for colour-consistency check at hole pixels")
    ap.add_argument("--out",     type=Path,  default=None)
    ap.add_argument("--device",  type=str,   default="auto")
    ap.add_argument("--ckpt",    type=str,   default=None, help="checkpoint filename (searched at run root and ckpt/)")
    ap.add_argument("--iclr",    action="store_true",
                    help="minimal ICLR figure: prove holes are textureless (H1) + low-coverage/grazing (H2)")
    ap.add_argument("--views",   type=str, default=None,
                    help="ICLR multi-view: comma-separated view ids, e.g. '120,200,242,250'")
    ap.add_argument("--ckpts",   type=str, default=None,
                    help="ICLR checkpoint compare (one --view): comma-separated ckpt filenames, "
                         "e.g. 'checkpoint_step_010000.pt,checkpoint_final.pt'")
    ap.add_argument("--before",  type=str, default=None,
                    help="ICLR lost-ZNCC: 'before erosion' ckpt filename (pairs with --after)")
    ap.add_argument("--after",   type=str, default="checkpoint_final.pt",
                    help="ICLR lost-ZNCC: 'after erosion' ckpt filename")
    ap.add_argument("--topk",    type=int, default=4, help="ICLR lost-ZNCC: top-k views for ZNCC")
    ap.add_argument("--steps",      type=int,   default=192, help="ICLR: dense samples / trace iters proxy")
    ap.add_argument("--cov-cos",    type=float, default=0.3, help="ICLR: front-facing cos threshold for coverage")
    ap.add_argument("--tex-win",    type=int,   default=2,   help="ICLR: half-window for texture box-mean")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = args.out or (args.run_dir / "diagnose_holes.png")

    if args.iclr:
        if args.before:
            out = args.out or (args.run_dir / "holes_lost_zncc.png")
            iclr_lost_zncc(args.run_dir, args.before, args.after, args.view,
                           args.down, args.steps, args.topk, out, device)
        elif args.ckpts:
            cklist = [x.strip() for x in args.ckpts.split(",") if x.strip()]
            out = args.out or (args.run_dir / "holes_iclr_ckpts.png")
            iclr_compare_ckpts(args.run_dir, cklist, args.view, args.down, args.steps,
                               args.cov_cos, args.tex_win, out, device)
        elif args.views:
            vlist = [int(x) for x in args.views.split(",") if x.strip() != ""]
            out = args.out or (args.run_dir / "holes_iclr_views.png")
            iclr_multiview(args.run_dir, vlist, args.down, args.steps,
                           args.cov_cos, args.tex_win, out, device, args.ckpt)
        else:
            out = args.out or (args.run_dir / "holes_iclr.png")
            iclr_figure(args.run_dir, args.view, args.down, args.steps,
                        args.cov_cos, args.tex_win, out, device, args.ckpt)
        return

    print(f"[diagnose] loading model from {args.run_dir} …")
    f, cfg = _load_model(args.run_dir, device)

    # load cameras only (avoid loading all images into RAM)
    scene = cfg.scene
    is_blender = (scene / "transforms_train.json").exists()
    if is_blender:
        import json
        from PIL import Image as _PIL
        meta = json.loads((scene / "transforms_train.json").read_text())
        frames = meta["frames"]
        fl = meta.get("fl_x") or meta.get("camera_angle_x")
        # collect c2ws
        c2ws_all, Ks_all, img_paths = [], [], []
        for fr in frames:
            c2w_raw = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
            # blender convention: flip Y and Z
            c2w_bl = c2w_raw.clone()
            c2w_bl[:3, 1] *= -1; c2w_bl[:3, 2] *= -1
            c2ws_all.append(c2w_bl)
            p = scene / (fr["file_path"] + (".png" if not fr["file_path"].endswith(".png") else ""))
            img_paths.append(p)
        # load one image to get H, W and K
        _im0 = _PIL.open(img_paths[0])
        H, W = _im0.height, _im0.width
        if "fl_x" in meta:
            fx, fy = meta["fl_x"], meta.get("fl_y", meta["fl_x"])
            cx, cy = meta.get("cx", W / 2), meta.get("cy", H / 2)
        else:
            fov = meta["camera_angle_x"]
            fx = fy = W / (2 * np.tan(fov / 2))
            cx, cy = W / 2, H / 2
        K_single = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        Ks_all = [K_single] * len(c2ws_all)
        vi = min(args.view, len(c2ws_all) - 1)
        c2w = c2ws_all[vi].numpy()
        K   = Ks_all[vi]
        # load only the one GT image
        img_gt_pil = _PIL.open(img_paths[vi]).convert("RGBA")
        img_gt = np.array(img_gt_pil, dtype=np.float32)[..., :3] / 255.0
    else:
        from lip_tracer.data import load_views
        views = load_views(scene)
        vi = min(args.view, views["c2w"].shape[0] - 1)
        c2w = views["c2w"][vi].numpy()
        K   = views["K"][vi].numpy()
        H, W = views["H"], views["W"]
        img_gt = views["images"][vi].numpy() if "images" in views else None
        c2ws_all = [views["c2w"][i] for i in range(views["c2w"].shape[0])]
        Ks_all   = [views["K"][i].numpy() for i in range(views["K"].shape[0])]
        img_paths = []  # DTU images already in views["images"], handled below

    print(f"[diagnose] building rays for view {vi} at down={args.down} …")
    o, d, Hd, Wd = _make_rays(c2w, K, H, W, args.down, device)
    N = Hd * Wd

    print(f"[diagnose] tracing {N:,} rays ({Hd}×{Wd}) …")
    hit, t_out, sdf_min, grad_norm = _trace_with_diagnostics(
        f, o, d, cfg.trace, args.t_steps, device)

    hit_map      = hit.reshape(Hd, Wd).numpy()
    sdf_min_map  = sdf_min.reshape(Hd, Wd).numpy()
    grad_map     = grad_norm.reshape(Hd, Wd).numpy()

    # pick sample rays: deepest holes (largest sdf_min among misses)
    miss_idx = torch.where(~hit)[0]
    hit_idx  = torch.where(hit)[0]
    rng = np.random.default_rng(42)

    # spread hole rays spatially: pick from different grid quadrants
    if len(miss_idx) > 0:
        # sort by sdf_min descending (worst holes first) then subsample
        order = sdf_min[miss_idx].argsort(descending=True)
        top   = miss_idx[order[:min(len(order), 200)]].numpy()
        chosen_hole = rng.choice(top, size=min(args.n_hole_rays, len(top)), replace=False)
    else:
        chosen_hole = np.array([], dtype=np.int64)

    chosen_hit = (hit_idx[rng.choice(len(hit_idx), size=min(args.n_hit_rays, len(hit_idx)),
                                     replace=False)].numpy()
                  if len(hit_idx) > 0 else np.array([], dtype=np.int64))

    print(f"[diagnose] computing colour variance at hole pixels ({args.color_views} views) …")
    color_std = _color_variance_at_holes(
        o.cpu(), d.cpu(), sdf_min, hit,
        c2ws_all, Ks_all, img_paths,
        H, W, vi, args.color_views,
    )
    color_std_map = color_std.reshape(Hd, Wd)
    hole_color_std = color_std[~hit.numpy()]
    mean_hole_std  = float(np.nanmean(hole_color_std)) if len(hole_color_std) > 0 else float("nan")

    print(f"[diagnose] profiling {len(chosen_hole)} hole rays + {len(chosen_hit)} hit rays …")
    ts_arr = None
    hole_profiles = hit_profiles = None
    if len(chosen_hole) > 0:
        hole_profiles, ts_arr = _ray_sdf_profiles(f, o, d, cfg.trace.t_far, args.t_steps,
                                                   chosen_hole, device)
    if len(chosen_hit) > 0:
        hit_profiles, ts_arr2 = _ray_sdf_profiles(f, o, d, cfg.trace.t_far, args.t_steps,
                                                   chosen_hit, device)
        if ts_arr is None:
            ts_arr = ts_arr2

    # ── plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BG = "#0d0d0d"
    plt.style.use("dark_background")

    n_profile_rows = 2 if (hole_profiles is not None or hit_profiles is not None) else 0
    n_rows = 2 + n_profile_rows
    fig = plt.figure(figsize=(20, 5 * n_rows), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.42, wspace=0.35)

    def _ax(row, col, **kw):
        a = fig.add_subplot(gs[row, col], facecolor=BG, **kw)
        a.tick_params(labelsize=7)
        return a

    # ── Row 0: hit map, sdf_min (log), |∇f|, GT image ────────────────────────
    ax_hit = _ax(0, 0)
    rgb_hit = np.where(hit_map[..., None],
                       np.array([[[0.2, 0.8, 0.2]]]),
                       np.array([[[0.9, 0.2, 0.2]]]))
    ax_hit.imshow(rgb_hit)
    ax_hit.set_title(f"Hit map  ({hit_map.sum():,} hit / {(~hit_map).sum():,} miss)", fontsize=9)
    ax_hit.axis("off")

    ax_smin = _ax(0, 1)
    eps = cfg.trace.eps
    smin_vis = np.log10(np.clip(sdf_min_map, 1e-5, None))
    log_eps = np.log10(eps)
    im_smin = ax_smin.imshow(smin_vis, cmap="plasma_r",
                              vmin=np.log10(1e-5), vmax=np.log10(cfg.trace.t_far))
    cb = fig.colorbar(im_smin, ax=ax_smin, fraction=0.046, pad=0.02)
    cb.set_label("log10(sdf_min)", fontsize=7)
    cb.ax.axhline(log_eps, color="cyan", lw=1.5, label=f"eps={eps}")
    cb.ax.tick_params(labelsize=7)
    ax_smin.set_title("sdf_min along ray  (cyan = eps threshold)", fontsize=9)
    ax_smin.axis("off")
    # overlay hit contour
    ax_smin.contour(hit_map.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)

    ax_grad = _ax(0, 2)
    gp = np.percentile(grad_map, 99)
    im_grad = ax_grad.imshow(grad_map, cmap="inferno", vmin=0, vmax=max(gp, 1e-3))
    fig.colorbar(im_grad, ax=ax_grad, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    ax_grad.set_title("|∇f| at ray origin  (0=flat SDF, 1=proper eikonal)", fontsize=9)
    ax_grad.axis("off")
    ax_grad.contour(hit_map.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)

    ax_img = _ax(0, 3)
    if img_gt is not None:
        ax_img.imshow(img_gt[:Hd * args.down:args.down, :Wd * args.down:args.down])
        ax_img.set_title(f"GT image (view {vi})", fontsize=9)
    else:
        ax_img.text(0.5, 0.5, "no GT image", transform=ax_img.transAxes,
                    ha="center", va="center", color="white")
    ax_img.axis("off")
    ax_img.contour(hit_map.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)

    # ── Row 1: histograms + summary ───────────────────────────────────────────
    ax_h1 = _ax(1, 0)
    sdf_min_d = sdf_min.detach()
    vals_hit  = sdf_min_d[hit].numpy()  if hit.any()  else np.array([0.0])
    vals_miss = sdf_min_d[~hit].numpy() if (~hit).any() else np.array([1.0])
    bins = np.logspace(-5, np.log10(cfg.trace.t_far + 1e-3), 80)
    ax_h1.hist(vals_hit,  bins=bins, color="#00c8ff", alpha=0.7, label=f"hit  ({len(vals_hit):,})",  density=True)
    ax_h1.hist(vals_miss, bins=bins, color="#ff4444", alpha=0.7, label=f"miss ({len(vals_miss):,})", density=True)
    ax_h1.axvline(eps, color="cyan", lw=1.2, ls="--", label=f"eps={eps}")
    ax_h1.set_xscale("log")
    ax_h1.set_xlabel("sdf_min", fontsize=8)
    ax_h1.set_ylabel("density", fontsize=8)
    ax_h1.set_title("sdf_min distribution: hit vs miss", fontsize=9)
    ax_h1.legend(fontsize=7, framealpha=0.2)

    ax_h2 = _ax(1, 1)
    ax_h2.hist(grad_map[hit_map],  bins=50, color="#00c8ff", alpha=0.7, label="hit",  density=True)
    ax_h2.hist(grad_map[~hit_map], bins=50, color="#ff4444", alpha=0.7, label="miss", density=True)
    ax_h2.set_xlabel("|∇f| at origin", fontsize=8)
    ax_h2.set_ylabel("density", fontsize=8)
    ax_h2.set_title("|∇f| distribution: hit vs miss", fontsize=9)
    ax_h2.legend(fontsize=7, framealpha=0.2)

    # summary text
    miss_smin_median = float(np.median(vals_miss))
    miss_above_eps   = float((vals_miss > eps).mean())
    hit_frac         = float(hit.float().mean())
    grad_hit_med     = float(np.median(grad_map[hit_map]))  if hit_map.any()  else 0.0
    grad_miss_med    = float(np.median(grad_map[~hit_map])) if (~hit_map).any() else 0.0

    if mean_hole_std < 0.05:
        color_verdict = "LOW  → topology/init problem  (fg loss should fix)"
        color_verdict_c = "#00ff88"
    elif mean_hole_std > 0.15:
        color_verdict = "HIGH → specular/view-dep.  (fg loss will fight photo)"
        color_verdict_c = "#ff4444"
    else:
        color_verdict = "MED  → mixed causes"
        color_verdict_c = "#ffaa00"

    ax_sum = _ax(1, 2)
    ax_sum.axis("off")
    summary = (
        f"view:            {vi}\n"
        f"resolution:      {Hd}×{Wd}  (down={args.down})\n"
        f"hit rate:        {hit_frac:.1%}\n\n"
        f"miss sdf_min median:  {miss_smin_median:.4f}\n"
        f"miss sdf_min > eps:   {miss_above_eps:.1%}\n"
        f"  → {miss_above_eps:.0%} of misses never got close\n\n"
        f"median |∇f| hit:   {grad_hit_med:.3f}\n"
        f"median |∇f| miss:  {grad_miss_med:.3f}\n"
        f"  (1.0 = perfect eikonal)\n\n"
        f"colour std at holes: {mean_hole_std:.3f}\n"
        f"  ({args.color_views} views, proxy depth=sdf_min)\n\n"
        f"trace iters:     {cfg.trace.iters}\n"
        f"trace eps:       {cfg.trace.eps}\n"
        f"w_eikonal:       {cfg.train.w_eikonal}\n"
        f"w_sil:           {cfg.train.w_sil}\n"
        f"w_free:          {cfg.train.w_free}"
    )
    ax_sum.text(0.05, 0.97, summary, transform=ax_sum.transAxes,
                fontsize=9, va="top", ha="left", family="monospace",
                color="white", linespacing=1.6)

    # colour variance map
    ax_cvar = _ax(1, 3)
    cvar_vis = np.where(~hit.numpy().reshape(Hd, Wd), color_std_map, np.nan)
    im_cvar = ax_cvar.imshow(cvar_vis, cmap="RdYlGn_r", vmin=0.0, vmax=0.3)
    cb_cvar = fig.colorbar(im_cvar, ax=ax_cvar, fraction=0.046, pad=0.02)
    cb_cvar.set_label("colour std (miss pixels only)", fontsize=7)
    cb_cvar.ax.tick_params(labelsize=7)
    ax_cvar.set_title(
        f"Colour consistency at holes  (mean={mean_hole_std:.3f})\n"
        f"green=consistent(topology)  red=specular",
        fontsize=8)
    ax_cvar.axis("off")
    ax_cvar.text(0.05, 0.03, color_verdict, transform=ax_cvar.transAxes,
                 fontsize=8, color=color_verdict_c, fontweight="bold", va="bottom")

    # verdict (text rendered into ax_sum below)
    if miss_above_eps > 0.7:
        v1 = "✗ SDF DELETED / COLLAPSED"
        v1c = "#ff4444"
        v1d = "Most miss rays have sdf_min >> eps.\nThe zero level-set is simply absent\nin those regions — the SDF was pushed\npositive by the photometric loss."
    elif miss_smin_median < eps * 3:
        v1 = "? FLAT SDF NEAR ZERO"
        v1c = "#ffaa00"
        v1d = "sdf_min is near eps but rays still\nmiss. SDF has a very flat gradient\n(|∇f|≈0), making steps tiny and the\ntracer can't converge in budget."
    else:
        v1 = "? ITER BUDGET TOO SMALL"
        v1c = "#ffaa00"
        v1d = "sdf_min is moderate. There may be a\nzero crossing but the tracer can't\nreach it in the allotted iters.\nSee SDF profiles below."

    if grad_miss_med < 0.2:
        v2 = "✗ FLAT GRADIENT (no eikonal)"
        v2c = "#ff4444"
        v2d = "Median |∇f| on miss rays is very low.\nWithout eikonal loss the 1-Lip network\nlearns a nearly-constant SDF in unseen\nregions → tracer stalls."
    elif grad_miss_med > 0.8:
        v2 = "✓ gradient norm OK"
        v2c = "#00ff88"
        v2d = "|∇f| ≈ 1 on miss rays — SDF is well-\nconditioned there."
    else:
        v2 = "~ gradient norm weak"
        v2c = "#ffaa00"
        v2d = f"|∇f| ≈ {grad_miss_med:.2f} on miss rays."

    ax_sum.text(0.05, 0.0, "DIAGNOSIS", transform=ax_sum.transAxes,
                fontsize=10, fontweight="bold", va="bottom", color="white")
    ax_sum.text(0.05, -0.08, v1, transform=ax_sum.transAxes,
                fontsize=9, fontweight="bold", va="top", color=v1c)
    ax_sum.text(0.05, -0.22, v1d, transform=ax_sum.transAxes,
                fontsize=7, va="top", color="white", linespacing=1.4)
    ax_sum.text(0.05, -0.46, v2, transform=ax_sum.transAxes,
                fontsize=9, fontweight="bold", va="top", color=v2c)
    ax_sum.text(0.05, -0.56, v2d, transform=ax_sum.transAxes,
                fontsize=7, va="top", color="white", linespacing=1.4)

    # ── Rows 2–3: SDF profiles ────────────────────────────────────────────────
    if ts_arr is not None:
        ax_hp = fig.add_subplot(gs[2, :2], facecolor=BG)
        ax_hp.tick_params(labelsize=7)
        ax_hp.axhline(0, color="white", lw=0.7, alpha=0.5)
        ax_hp.axhline(cfg.trace.eps, color="cyan", lw=0.8, ls="--", alpha=0.8, label=f"+eps={eps}")
        ax_hp.axhline(-cfg.trace.eps, color="cyan", lw=0.8, ls="--", alpha=0.8)
        if hole_profiles is not None:
            cmap_h = plt.cm.Reds
            for k, prof in enumerate(hole_profiles):
                col = cmap_h(0.4 + 0.5 * k / max(len(hole_profiles) - 1, 1))
                ax_hp.plot(ts_arr, prof, color=col, lw=0.9, alpha=0.85,
                           label=f"hole#{k}" if k < 4 else None)
                # mark any zero crossings
                sign_changes = np.where(np.diff(np.sign(prof)))[0]
                for sc in sign_changes:
                    ax_hp.axvline(ts_arr[sc], color=col, lw=0.5, ls=":", alpha=0.6)
        n_crossings = 0
        if hole_profiles is not None:
            n_crossings = sum(1 for p in hole_profiles if np.any(np.diff(np.sign(p)) != 0))
        ax_hp.set_xlabel("t (ray distance)", fontsize=8)
        ax_hp.set_ylabel("f(o + t·d)", fontsize=8)
        ax_hp.set_title(
            f"SDF profiles: HOLE rays  "
            f"({n_crossings}/{len(hole_profiles) if hole_profiles is not None else 0} have zero crossings — "
            f"vertical dots = crossing location)", fontsize=9)
        ax_hp.legend(fontsize=6, framealpha=0.2, ncol=3)
        ax_hp.set_ylim(-cfg.trace.t_far * 0.3, cfg.trace.t_far * 0.3)

        ax_hitp = fig.add_subplot(gs[2, 2:], facecolor=BG)
        ax_hitp.tick_params(labelsize=7)
        ax_hitp.axhline(0, color="white", lw=0.7, alpha=0.5)
        ax_hitp.axhline(cfg.trace.eps,  color="cyan", lw=0.8, ls="--", alpha=0.8, label=f"+eps")
        ax_hitp.axhline(-cfg.trace.eps, color="cyan", lw=0.8, ls="--", alpha=0.8)
        if hit_profiles is not None:
            cmap_g = plt.cm.Greens
            for k, prof in enumerate(hit_profiles):
                col = cmap_g(0.4 + 0.5 * k / max(len(hit_profiles) - 1, 1))
                ax_hitp.plot(ts_arr2, prof, color=col, lw=0.9, alpha=0.85,
                             label=f"hit#{k}")
        ax_hitp.set_xlabel("t (ray distance)", fontsize=8)
        ax_hitp.set_ylabel("f(o + t·d)", fontsize=8)
        ax_hitp.set_title("SDF profiles: HIT rays (reference)", fontsize=9)
        ax_hitp.legend(fontsize=6, framealpha=0.2)
        ax_hitp.set_ylim(-cfg.trace.t_far * 0.3, cfg.trace.t_far * 0.3)

        # row 3: zoomed SDF profiles for hole rays near t=0 to t_far/2
        if hole_profiles is not None and n_rows >= 4:
            ax_zoom = fig.add_subplot(gs[3, :], facecolor=BG)
            ax_zoom.tick_params(labelsize=7)
            ax_zoom.axhline(0,   color="white", lw=0.8, alpha=0.5)
            ax_zoom.axhline(cfg.trace.eps,  color="cyan", lw=1.0, ls="--", alpha=0.9, label=f"±eps={eps}")
            ax_zoom.axhline(-cfg.trace.eps, color="cyan", lw=1.0, ls="--", alpha=0.9)
            # show min-sdf per hole ray and mark
            for k, (prof, idx) in enumerate(zip(hole_profiles, chosen_hole)):
                col = plt.cm.Reds(0.4 + 0.5 * k / max(len(hole_profiles) - 1, 1))
                ax_zoom.plot(ts_arr, prof, color=col, lw=1.2, alpha=0.9)
                t_min_idx = np.argmin(np.abs(prof))
                ax_zoom.plot(ts_arr[t_min_idx], prof[t_min_idx], "o", color=col,
                             ms=4, label=f"ray{k} sdf_min={abs(prof[t_min_idx]):.4f}")
            ax_zoom.set_xlabel("t", fontsize=8)
            ax_zoom.set_ylabel("f(o + t·d)", fontsize=8)
            ax_zoom.set_title("SDF profiles HOLE rays — marker = minimum |f| along ray", fontsize=9)
            ylim = max(0.5, float(np.abs(hole_profiles).max() * 0.5))
            ax_zoom.set_ylim(-ylim, ylim)
            ax_zoom.legend(fontsize=6, framealpha=0.2, ncol=4)

    plt.style.use("default")
    fig.suptitle(
        f"Hole diagnostic — {args.run_dir.name}  view={vi}  "
        f"iters={cfg.trace.iters}  eps={cfg.trace.eps}  w_eikonal={cfg.train.w_eikonal}",
        fontsize=12, y=1.002, color="white")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[diagnose] saved → {out}")

    # print textual summary
    print("\n=== DIAGNOSIS SUMMARY ===")
    print(f"  hit rate:              {hit_frac:.1%}")
    print(f"  miss sdf_min median:   {miss_smin_median:.5f}  (eps={eps})")
    print(f"  miss sdf_min > eps:    {miss_above_eps:.1%}  of all miss rays")
    print(f"  median |∇f| on hits:   {grad_hit_med:.3f}")
    print(f"  median |∇f| on misses: {grad_miss_med:.3f}")
    if hole_profiles is not None:
        nc = sum(1 for p in hole_profiles if np.any(np.diff(np.sign(p)) != 0))
        print(f"  zero crossings in sampled hole rays: {nc}/{len(hole_profiles)}")
        print(f"  → {'surface EXISTS but tracer misses it' if nc > len(hole_profiles) // 2 else 'surface ABSENT along hole rays'}")
    print(f"  colour std at holes:   {mean_hole_std:.3f}  ({args.color_views} views, proxy=sdf_min depth)")
    print(f"  → {color_verdict}")


if __name__ == "__main__":
    main()
