"""ROI depth-offset sweep: does the trained PMVS-NCC loss carry geometric
signal through x_theta, without retraining?

For every pixel p in a ROI of one view:
  1. sphere-trace f_theta -> t0(p), hit, initial SDF normal n0 (frozen/detached)
  2. for delta in linspace(-0.005, +0.005, 81):
         x_delta(p) = o + (t0(p)+delta) * u
     evaluate the SAME PMVS-NCC patch loss used in training (patch=5,
     half_pix=2.0, bilinear, ncc_min from the run config) -- no smoothness,
     no anchor, no Adam.
  3. delta_star(p) = argmin_delta L_train_p(delta)
     train_gain(p)  = L_train_p(0) - L_train_p(delta_star)
     held_gain(p)   = L_held_p(0)  - L_held_p(delta_star)

The alt-view pool is the training nearest-camera pool (n_alt nearest cams);
it is split interleaved by distance into a train half and a held half.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lip_tracer.config import (Config, ModelConfig, TraceConfig)
from lip_tracer.data import load_views, precompute_alt_cameras
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd


# ---------------------------------------------------------------- ZNCC ----

@torch.no_grad()
def pmvs_zncc_pair(images, x3d, n, vi_a, vi_b, K_all, w2c_all, H, W,
                   patch=5, half_pix=2.0):
    """Faithful copy of lip_tracer.loss.pmvs_ncc_loss math, but returns
    per-pixel ZNCC + validity (no compaction, no gate) so it maps back to
    the ROI grid.

    x3d (B,3), n (B,3) frozen normal, vi_a/vi_b scalar-broadcastable (B,).
    Returns zncc (B,), valid_geom (B,), textured (B,).
    """
    P = patch
    n = torch.nn.functional.normalize(n, dim=-1)
    up = n.new_zeros(n.shape[0], 3); up[:, 1] = 1.0
    swap = n[:, 1].abs() > 0.9
    up[swap, 1] = 0.0; up[swap, 0] = 1.0
    t1 = torch.nn.functional.normalize(torch.cross(n, up, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)

    R_a = w2c_all[vi_a, :3, :3]
    t_a = w2c_all[vi_a, :3, 3]
    xc_a = torch.einsum('bij,bj->bi', R_a, x3d) + t_a
    z_ref = xc_a[:, 2].clamp(min=1e-3)
    f_x = K_all[vi_a, 0, 0]
    step_3d = (2.0 * half_pix / max(P - 1, 1)) * z_ref / f_x

    offs = torch.linspace(-(P - 1) / 2, (P - 1) / 2, P, device=x3d.device)
    oi, oj = torch.meshgrid(offs, offs, indexing='ij')
    oi = oi.reshape(-1); oj = oj.reshape(-1)
    pts3d = (x3d.unsqueeze(1)
             + step_3d[:, None, None] * (oi[None, :, None] * t1.unsqueeze(1)
                                         + oj[None, :, None] * t2.unsqueeze(1)))

    def _project(vi):
        R = w2c_all[vi, :3, :3]
        tv = w2c_all[vi, :3, 3]
        xc = (R.unsqueeze(1) @ pts3d.unsqueeze(-1)).squeeze(-1) + tv.unsqueeze(1)
        ph = (K_all[vi].unsqueeze(1) @ xc.unsqueeze(-1)).squeeze(-1)
        uv = ph[:, :, :2] / ph[:, :, 2:3].clamp(min=1e-6)
        return uv, xc[:, :, 2]

    uv_a, z_pa = _project(vi_a)
    uv_b, z_pb = _project(vi_b)

    all_in_a = (z_pa > 0).all(1) \
        & (uv_a[:, :, 0] >= 0).all(1) & (uv_a[:, :, 0] < W).all(1) \
        & (uv_a[:, :, 1] >= 0).all(1) & (uv_a[:, :, 1] < H).all(1)
    all_in_b = (z_pb > 0).all(1) \
        & (uv_b[:, :, 0] >= 0).all(1) & (uv_b[:, :, 0] < W).all(1) \
        & (uv_b[:, :, 1] >= 0).all(1) & (uv_b[:, :, 1] < H).all(1)
    valid_geom = all_in_a & all_in_b

    def _sample(uv, vi):
        u = uv[:, :, 0].clamp(0, W - 1)
        v = uv[:, :, 1].clamp(0, H - 1)
        u0 = u.long(); u1 = (u0 + 1).clamp(max=W - 1)
        v0 = v.long(); v1 = (v0 + 1).clamp(max=H - 1)
        wu = (u - u0.float()).unsqueeze(-1)
        wv = (v - v0.float()).unsqueeze(-1)
        vie = vi[:, None]
        c00 = images[vie, v0, u0].float(); c10 = images[vie, v1, u0].float()
        c01 = images[vie, v0, u1].float(); c11 = images[vie, v1, u1].float()
        return (c00 * (1 - wu) * (1 - wv) + c01 * wu * (1 - wv)
                + c10 * (1 - wu) * wv + c11 * wu * wv)

    pa = _sample(uv_a, vi_a)
    pb = _sample(uv_b, vi_b)
    pa = pa - pa.mean(dim=1, keepdim=True)
    pb = pb - pb.mean(dim=1, keepdim=True)
    std_a = pa.norm(dim=1)
    std_b = pb.norm(dim=1)
    textured = (std_a > 1e-4).all(1) & (std_b > 1e-4).all(1)
    pa = pa / std_a.unsqueeze(1).clamp(min=1e-6)
    pb = pb / std_b.unsqueeze(1).clamp(min=1e-6)
    zncc = (pa * pb).sum(dim=1).mean(dim=1).clamp(-1.0, 1.0)
    return zncc, valid_geom, textured


# ---------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/run_20260517_114302_scan122")
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--view", type=int, default=16)
    ap.add_argument("--x0", type=int, default=600)   # owl feather body region
    ap.add_argument("--x1", type=int, default=950)
    ap.add_argument("--y0", type=int, default=200)
    ap.add_argument("--y1", type=int, default=430)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=6,
                    help="nearest cams used as TRAIN alt views (training pool)")
    ap.add_argument("--n-held", type=int, default=4,
                    help="next nearest cams used as HELD/validation alt views")
    ap.add_argument("--n-delta", type=int, default=81)
    ap.add_argument("--delta-max", type=float, default=0.005)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default="artifacts/roi_delta_sweep_scan122_v16")
    args = ap.parse_args()

    run = Path(args.run)
    cfg_d = json.loads((run / "config.json").read_text())
    scene = Path(cfg_d["scene"])
    mc = cfg_d["model"]; tc = cfg_d["trace"]; tr = cfg_d["train"]
    ncc_patch = tr["ncc_patch"]; ncc_half = tr["ncc_half_pix"]
    ncc_min = tr["ncc_min"]; n_alt = tr["n_alt"]
    print(f"scene={scene}\n  ncc_patch={ncc_patch} half_pix={ncc_half} "
          f"ncc_min={ncc_min} n_alt={n_alt}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # --- model ---
    f = make_model(hidden=mc["hidden"], depth=mc["depth"],
                    group_size=mc["group_size"], activation=mc["activation"],
                    input_encoding=mc["input_encoding"], multires=mc["multires"],
                    architecture=mc["architecture"]).to(dev)
    ck = torch.load(run / args.ckpt, map_location=dev)
    f.load_state_dict(ck["f"], strict=True)
    f.eval()
    trace_cfg = TraceConfig(iters=tc["iters"], eps=tc["eps"], t_far=tc["t_far"],
                            eik_stride=tc["eik_stride"],
                            newton_steps=tc["newton_steps"],
                            grad_mode=tc.get("grad_mode", "idr"),
                            bsphere_radius=tc.get("bsphere_radius", 0.0),
                            sdf_min_beta=tc.get("sdf_min_beta", 0.0))

    # --- views ---
    views = load_views(scene)
    H, W = views["H"], views["W"]
    images = views["images"].to(dev)               # (V,H,W,3)
    c2w = views["c2w"].to(dev)
    K_all = views["K"].to(dev)
    w2c_all = torch.linalg.inv(c2w)
    V = c2w.shape[0]
    print(f"  V={V} H={H} W={W}")

    # Pool = (n_train + n_held) nearest cams. TRAIN = the nearest n_train (the
    # alt-pair pool the NCC loss actually trained on for this ref view;
    # config n_alt={n_alt}); HELD = the next n_held nearest, OUTSIDE the
    # training alt-pool -> a genuine generalization split.
    n_pool = args.n_train + args.n_held
    alt_nn = precompute_alt_cameras(views, n_pool)  # (V, n_pool) nearest cams
    alt = alt_nn[args.view].to(dev)                 # (n_pool,) nearest ascending
    train_views = alt[:args.n_train]
    held_views = alt[args.n_train:n_pool]
    print(f"  view {args.view}: pool(nearest {n_pool})={alt.tolist()} "
          f"(training n_alt={n_alt})")
    print(f"  train({args.n_train})={train_views.tolist()}  "
          f"held({args.n_held})={held_views.tolist()}")

    # --- ROI rays (deterministic pixel grid, down=1 -> pixel-center == int) ---
    xs = torch.arange(args.x0, args.x1, args.stride)
    ys = torch.arange(args.y0, args.y1, args.stride)
    nx, ny = len(xs), len(ys)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")   # (ny? ) use xy: shape (ny,nx)?
    # torch xy indexing: gx,gy have shape (len(ys), len(xs)) -> (ny, nx)
    px = gx.reshape(-1).float().to(dev)
    py = gy.reshape(-1).float().to(dev)
    Npix = px.numel()

    Kv = K_all[args.view]
    c2wv = c2w[args.view]
    d_cam = torch.stack([(px - Kv[0, 2]) / Kv[0, 0],
                         (py - Kv[1, 2]) / Kv[1, 1],
                         torch.ones_like(px)], dim=-1)
    dirs = d_cam @ c2wv[:3, :3].T
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    o = c2wv[:3, 3].expand(Npix, 3).contiguous()

    rgb_roi = images[args.view, py.long(), px.long()].reshape(ny, nx, 3).cpu().numpy()

    # --- sphere trace -> t0, hit, frozen normal n0 ---
    t0 = torch.empty(Npix, device=dev)
    hit = torch.empty(Npix, dtype=torch.bool, device=dev)
    n0 = torch.empty(Npix, 3, device=dev)
    for i in range(0, Npix, args.chunk):
        sl = slice(i, i + args.chunk)
        xh, t_, h_ = trace_nograd(f, o[sl], dirs[sl], trace_cfg)
        t0[sl] = t_; hit[sl] = h_
        xr = xh.detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(xr).sum(), xr)[0]
        n0[sl] = torch.nn.functional.normalize(g, dim=-1).detach()
    print(f"  hit fraction in ROI: {hit.float().mean().item():.3f}")

    # --- delta sweep ---
    deltas = torch.linspace(-args.delta_max, args.delta_max, args.n_delta,
                            device=dev)
    zero_idx = int(torch.argmin(deltas.abs()).item())
    nD = args.n_delta

    all_alt = alt                                    # (n_alt,)
    nA = all_alt.numel()
    # per-pixel, per-delta, per-alt-view ZNCC + kept-mask
    zncc_all = torch.full((Npix, nD, nA), float("nan"), device=dev)
    kept_all = torch.zeros((Npix, nD, nA), dtype=torch.bool, device=dev)

    for ci in range(0, Npix, args.chunk):
        sl = slice(ci, min(ci + args.chunk, Npix))
        oc, dc, t0c = o[sl], dirs[sl], t0[sl]
        n0c = n0[sl]
        B = oc.shape[0]
        # vectorise over all deltas at once: pseudo-batch (B*nD)
        x3d = (oc.unsqueeze(1)
               + (t0c.unsqueeze(1) + deltas[None, :]).unsqueeze(-1)
               * dc.unsqueeze(1)).reshape(B * nD, 3)          # (B*nD, 3)
        n_rep = n0c.unsqueeze(1).expand(B, nD, 3).reshape(B * nD, 3)
        via = torch.full((B * nD,), args.view, dtype=torch.long, device=dev)
        for ai in range(nA):
            vib = all_alt[ai].expand(B * nD)
            z, vg, tx = pmvs_zncc_pair(images, x3d, n_rep, via, vib,
                                       K_all, w2c_all, H, W,
                                       ncc_patch, ncc_half)
            zncc_all[sl, :, ai] = z.reshape(B, nD)
            kept_all[sl, :, ai] = (vg & tx & (z > ncc_min)).reshape(B, nD)
        if ci // args.chunk % 4 == 0:
            print(f"  swept {sl.stop}/{Npix}", flush=True)

    # masks for train / held subsets within all_alt
    is_train = torch.tensor([bool((all_alt[i] == train_views).any())
                             for i in range(nA)], device=dev)
    is_held = ~is_train

    # Fixed visibility: K_p^0 = views kept at delta=0 (valid_geom & textured &
    # zncc>ncc_min). The SAME set is used for every delta, so the loss change is
    # purely geometric (no confound from views entering/leaving the valid set).
    k0_supp = kept_all[:, zero_idx, :]                       # (Npix, nA) bool

    def subset_loss(zncc, mask):
        """L_p(delta) = (1/|K_p^0 ∩ subset|) Σ_{k∈K_p^0∩subset} (1 - ZNCC_p,k(delta)).
        Support fixed at delta=0. Returns (Npix, nD), nan where support empty."""
        supp = k0_supp & mask[None, :]                       # (Npix, nA)
        cnt = supp.sum(-1)                                   # (Npix,)
        s = torch.where(supp[:, None, :], 1.0 - zncc,
                        torch.zeros_like(zncc)).sum(-1)       # (Npix, nD)
        L = torch.where(cnt[:, None] > 0, s / cnt[:, None].clamp(min=1),
                        torch.full_like(s, float("nan")))
        return L, cnt

    L_train, cnt_train = subset_loss(zncc_all, is_train)
    L_held, cnt_held = subset_loss(zncc_all, is_held)

    # delta_star = argmin over deltas of L_train (ignore nan)
    L_train_filled = torch.nan_to_num(L_train, nan=float("inf"))
    has_train = torch.isfinite(L_train_filled).any(dim=1)
    delta_idx = L_train_filled.argmin(dim=1)            # (Npix,)
    delta_star = deltas[delta_idx]

    ar = torch.arange(Npix, device=dev)
    L_train_0 = L_train[:, zero_idx]
    L_train_ds = L_train[ar, delta_idx]
    L_held_0 = L_held[:, zero_idx]
    L_held_ds = L_held[ar, delta_idx]

    train_gain = L_train_0 - L_train_ds
    held_gain = L_held_0 - L_held_ds

    valid = hit & has_train & torch.isfinite(L_train_0) & torch.isfinite(L_train_ds)
    valid_h = valid & torch.isfinite(L_held_0) & torch.isfinite(L_held_ds)

    # ZNCC-at-delta0 aggregates over all n_alt valid views
    z0 = zncc_all[:, zero_idx, :]
    k0 = kept_all[:, zero_idx, :]
    z0m = torch.where(k0, z0, torch.full_like(z0, float("nan")))
    def topk_mean(z, k):
        zz = z.clone()
        zz[~k] = -1e9
        vals, _ = zz.sort(dim=-1, descending=True)
        kk = k.sum(-1)
        out = torch.full((Npix,), float("nan"), device=dev)
        for kn in (vals.shape[1],):
            pass
        return vals, kk
    vals_sorted, ncnt0 = topk_mean(z0, k0)
    def mean_topn(nn):
        v = vals_sorted[:, :nn]
        ok = ncnt0 >= nn
        m = v.mean(dim=1)
        return torch.where(ok, m, torch.full_like(m, float("nan")))
    zncc_top2 = mean_topn(2)
    zncc_top3 = mean_topn(3)
    zncc_mean6 = torch.where(ncnt0 > 0,
                             torch.nansum(z0m, dim=1) / ncnt0.clamp(min=1),
                             torch.full((Npix,), float("nan"), device=dev))

    # mean ZNCC at delta=0 and best ZNCC after sweep, over TRAIN views
    def train_view_zncc(di_idx):
        z = zncc_all[ar, di_idx][:, :]            # (Npix, nA)
        k = kept_all[ar, di_idx][:, :]
        m = k & is_train[None, :]
        c = m.sum(-1)
        s = torch.where(m, z, torch.zeros_like(z)).sum(-1)
        return torch.where(c > 0, s / c.clamp(min=1),
                           torch.full_like(s, float("nan"))), c
    zt0, _ = train_view_zncc(torch.full((Npix,), zero_idx, device=dev))
    ztds, _ = train_view_zncc(delta_idx)

    def reshape(t):
        return t.detach().cpu().numpy().reshape(ny, nx)

    # --- oracle depth + normal-from-depth (before vs after) ---
    # t_oracle(p) = t0(p) + delta_star(p). Surface points X = o + t*u; the
    # local normal is estimated from the point-map spatial gradients
    # (dX/dpx x dX/dpy), expressed in the reference camera frame.
    t0_np = reshape(t0)
    t_oracle = t0 + torch.where(valid, delta_star, torch.zeros_like(delta_star))
    t_oracle_np = reshape(t_oracle)
    o_g = o.reshape(ny, nx, 3).cpu().numpy()
    u_g = dirs.reshape(ny, nx, 3).cpu().numpy()
    R_wc = w2c_all[args.view, :3, :3].cpu().numpy()      # world -> ref cam

    def normal_from_depth(t_map):
        X = o_g + t_map[..., None] * u_g                 # (ny,nx,3) world
        dXdy, dXdx = np.gradient(X, axis=0), np.gradient(X, axis=1)
        n = np.cross(dXdx, dXdy)                          # (ny,nx,3)
        n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
        n_cam = n @ R_wc.T                                # to ref-cam frame
        # orient toward camera (negative z in OpenCV cam frame faces viewer)
        flip = (n_cam[..., 2] > 0)[..., None]
        n_cam = np.where(flip, -n_cam, n_cam)
        return n_cam

    n_before = normal_from_depth(t0_np)
    n_after = normal_from_depth(t_oracle_np)
    vmask_np = reshape(valid.float()) > 0.5

    def n_rgb(n):
        img = 0.5 * (n + 1.0)
        img[~vmask_np] = 0.0
        return np.clip(img, 0, 1)

    npz = dict(
        rgb_roi=rgb_roi,
        delta_star=reshape(delta_star),
        train_gain=reshape(train_gain),
        held_gain=reshape(held_gain),
        train_view_count=reshape(cnt_train),
        held_view_count=reshape(cnt_held),
        zncc_top2=reshape(zncc_top2),
        zncc_top3=reshape(zncc_top3),
        zncc_mean6=reshape(zncc_mean6),
        valid=reshape(valid.float()),
        t0=t0_np,
        t_oracle=t_oracle_np,
        normal_before=n_before,
        normal_after=n_after,
        deltas=deltas.cpu().numpy(),
        roi=np.array([args.x0, args.x1, args.y0, args.y1]),
        view=args.view,
        train_views=train_views.cpu().numpy(),
        held_views=held_views.cpu().numpy(),
    )
    np.savez(out / "sweep.npz", **npz)

    # --- scalar summary ---
    def m(t, msk):
        v = t[msk]
        return float(v.mean()) if v.numel() else float("nan")

    summary = dict(
        n_valid=int(valid.sum()),
        n_pixels=int(Npix),
        hit_frac=float(hit.float().mean()),
        mean_train_gain=m(train_gain, valid),
        mean_held_gain=m(held_gain, valid_h),
        frac_train_gain_pos=float((train_gain[valid] > 0).float().mean()),
        frac_held_gain_pos=float((held_gain[valid_h] > 0).float().mean()),
        mean_abs_delta_star=m(delta_star.abs(), valid),
        std_delta_star=float(delta_star[valid].std()),
        mean_zncc_delta0_train=m(zt0, valid & torch.isfinite(zt0)),
        mean_best_zncc_after_sweep_train=m(ztds, valid & torch.isfinite(ztds)),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ext = [args.x0, args.x1, args.y1, args.y0]

    def show(a, img, title, cmap=None, vmin=None, vmax=None, sym=False):
        if sym:
            lim = np.nanmax(np.abs(img)) or 1.0
            vmin, vmax = -lim, lim
        im = a.imshow(img, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(title); fig.colorbar(im, ax=a, fraction=0.046)

    vmask = npz["valid"] > 0.5
    def masked(arr):
        b = arr.copy().astype(float); b[~vmask] = np.nan; return b

    ax[0, 0].imshow(np.clip(rgb_roi, 0, 1), extent=ext)
    ax[0, 0].set_title(f"RGB ROI  view {args.view}")
    show(ax[0, 1], masked(npz["delta_star"]), "delta_star (m)",
         cmap="coolwarm", sym=True)
    show(ax[0, 2], masked(npz["train_gain"]), "train_gain",
         cmap="viridis")
    show(ax[0, 3], masked(npz["held_gain"]), "held_gain",
         cmap="viridis")
    show(ax[1, 0], npz["train_view_count"], "valid train-view count @0",
         cmap="magma")
    show(ax[1, 1], masked(npz["zncc_top2"]), "mean top-2 ZNCC @0",
         cmap="viridis", vmin=-1, vmax=1)
    show(ax[1, 2], masked(npz["zncc_top3"]), "mean top-3 ZNCC @0",
         cmap="viridis", vmin=-1, vmax=1)
    show(ax[1, 3], masked(npz["zncc_mean6"]), f"mean {nA}-view ZNCC @0",
         cmap="viridis", vmin=-1, vmax=1)
    fig.suptitle(
        f"ROI delta-sweep  scan122 v{args.view}  "
        f"mean train_gain={summary['mean_train_gain']:.4f}  "
        f"held_gain={summary['mean_held_gain']:.4f}  "
        f"frac held>0={summary['frac_held_gain_pos']:.2f}",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=110)

    # --- oracle depth + normal-from-depth figure ---
    fig2, bx = plt.subplots(2, 3, figsize=(18, 10))
    bx[0, 0].imshow(np.clip(rgb_roi, 0, 1), extent=ext)
    bx[0, 0].set_title(f"RGB ROI  view {args.view}")
    tb = masked(t0_np)
    dlo = np.nanpercentile(tb, 2); dhi = np.nanpercentile(tb, 98)
    im = bx[0, 1].imshow(tb, extent=ext, cmap="turbo", vmin=dlo, vmax=dhi)
    bx[0, 1].set_title("t0 (traced depth)"); fig2.colorbar(im, ax=bx[0, 1], fraction=.046)
    im = bx[0, 2].imshow(masked(t_oracle_np), extent=ext, cmap="turbo",
                         vmin=dlo, vmax=dhi)
    bx[0, 2].set_title("t_oracle = t0 + delta*"); fig2.colorbar(im, ax=bx[0, 2], fraction=.046)
    bx[1, 0].imshow(n_rgb(n_before), extent=ext)
    bx[1, 0].set_title("normal-from-depth: before (t0)")
    bx[1, 1].imshow(n_rgb(n_after), extent=ext)
    bx[1, 1].set_title("normal-from-depth: after (t_oracle)")
    show(bx[1, 2], masked(npz["delta_star"]), "delta_star (m)",
         cmap="coolwarm", sym=True)
    fig2.suptitle(f"Oracle depth/normal  scan122 v{args.view}  "
                  f"mean|delta*|={summary['mean_abs_delta_star']:.4f} m",
                  fontsize=14)
    fig2.tight_layout()
    fig2.savefig(out / "oracle.png", dpi=110)
    print(f"wrote {out/'sweep.png'}, {out/'oracle.png'}, {out/'sweep.npz'}")


if __name__ == "__main__":
    main()
