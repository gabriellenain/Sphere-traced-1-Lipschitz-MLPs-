#!/usr/bin/env python3
"""Test 0 — 1D NCC-vs-depth profile + roughness/coverage attribution.

Diagnose the "parasite surfaces" on flat walls by asking, per pixel: does the
*training* NCC objective itself prefer the bump (data-driven) or does the
network park a bump where NCC actually prefers flat (PE / capacity-driven)?

The probe loads a run's checkpoint with its EXACT config (model + trace +
train NCC params, including the bilateral-gamma value *at the checkpoint's
step*), traces a reference view, auto-selects bump vs clean *textured*
foreground pixels by normal-map roughness, then marches each pixel's camera
ray through depth and recomputes the same `pmvs_ncc_loss` against the same
`alt_nn` views at every depth.

Read the profiles:
  * several periodic maxima, rendered surface sitting in a side-lobe
        -> texture aliasing (brick), NCC genuinely multi-modal
  * single clean max at the true depth but surface rendered off it
        -> network/PE-driven; NCC prefers flat, the model ignored it
  * flat / ambiguous ZNCC over a wide depth band, few valid alts
        -> low coverage; the constraint is just weak there

Clean pixels are required to be *textured* (high image-gradient) so the
contrast against bump pixels is geometry roughness, not texture presence.

Usage:
  python diag_ncc_profile.py --run-dir outputs/run_..._scan24_4938150 \
         --step 150000 --view 16
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.model import make_model
from lip_tracer.data import load_views, precompute_alt_cameras
from lip_tracer.sphere_tracing import trace_nograd, TraceConfig
from lip_tracer.loss import pmvs_ncc_loss


# ---------------------------------------------------------------- helpers ----
def trace_chunked(f, o, d, cfg, chunk=131072):
    """Full-view sphere trace in pixel chunks — the whole image at full res is
    too many rays for one forward+Newton pass on a shared GPU."""
    xs, ts, hs = [], [], []
    for i in range(0, o.shape[0], chunk):
        with torch.no_grad():
            x, t, h = trace_nograd(f, o[i:i + chunk], d[i:i + chunk], cfg)
        xs.append(x); ts.append(t); hs.append(h)
    return torch.cat(xs), torch.cat(ts), torch.cat(hs)


def alt_gate(f, x, n_t, a, ref_o, K_all, w2c_all, c2w_all, masks, H, W,
             cos_thresh, trace_cfg, occ_mode, eps_occ=1e-2):
    """Replicate the *training* per-(point,alt) NCC gate exactly (loss.py:399-462),
    plus the triangulation/parallax angle to the reference camera.

    Returns (over the T depth samples): gate(bool), |cos(n,dp)|, parallax-rad,
    occluded(bool), in_frame(bool), fg(bool).
    """
    cam_a = c2w_all[a, :3, 3]
    dirv  = cam_a[None] - x                                    # hit -> alt cam
    dist  = dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dp    = dirv / dist;  dist = dist.squeeze(-1)
    with torch.no_grad():
        if occ_mode == "from_hit":
            _, tp, hitp = trace_nograd(f, x + eps_occ * dp, dp, trace_cfg)
            not_occl = (~hitp) | (tp > dist - 0.1)
        else:                                                  # pinhole
            _, tp, hitp = trace_nograd(f, cam_a[None].expand_as(x).contiguous(), -dp, trace_cfg)
            not_occl = hitp & (dist <= tp + 0.1)
    cos    = (n_t * dp).sum(-1)
    cos_ok = cos.abs() > cos_thresh
    # reprojection into the alt view
    R = w2c_all[a, :3, :3]; tt = w2c_all[a, :3, 3]
    xc  = x @ R.T + tt
    uvh = xc @ K_all[a].T
    uv  = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)
    inb = (xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) \
          & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    if masks is not None:
        uvc = uv.long().clamp(min=0)
        uvc[:, 0].clamp_(max=W - 1); uvc[:, 1].clamp_(max=H - 1)
        fg = masks[a, uvc[:, 1], uvc[:, 0]].bool()
    else:
        fg = torch.ones_like(inb)
    gate = inb & not_occl & cos_ok & fg
    dref = ref_o[None] - x
    dref = dref / dref.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    parallax = torch.acos((dp * dref).sum(-1).clamp(-1, 1))    # rad
    return gate, cos.abs(), parallax, ~not_occl, inb, fg


def grad_normals(f, x, chunk=4096):
    """Unit ∇f at each x (B,3), chunked, with grad enabled."""
    out = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(xc).sum(), xc)[0].detach()
        out.append(g)
    g = torch.cat(out)
    return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def greedy_pick(score, valid, k, min_dist, H, W, largest):
    """Pick k pixels (flat indices) optimising `score` under `valid`, spaced
    at least `min_dist` px apart so the set isn't one clustered blob."""
    idx = np.where(valid.reshape(-1))[0]
    order = idx[np.argsort(score.reshape(-1)[idx])]
    if largest:
        order = order[::-1]
    picks, ys, xs = [], [], []
    for p in order:
        y, x = divmod(int(p), W)
        if all((y - yy) ** 2 + (x - xx) ** 2 >= min_dist ** 2 for yy, xx in zip(ys, xs)):
            picks.append(p); ys.append(y); xs.append(x)
            if len(picks) == k:
                break
    return picks


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--step", type=int, default=150000,
                    help="checkpoint step to load (ckpt/checkpoint_step_<step>.pt)")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="explicit checkpoint path (overrides --step)")
    ap.add_argument("--view", type=int, default=16, help="reference view index")
    ap.add_argument("--n-bump", type=int, default=4)
    ap.add_argument("--n-clean", type=int, default=4)
    ap.add_argument("--n-depth", type=int, default=161)
    ap.add_argument("--band", type=float, default=0.04,
                    help="half-width of the depth sweep, world units")
    ap.add_argument("--min-dist", type=int, default=20, help="px spacing of picks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = args.device
    cfg = json.loads((args.run_dir / "config.json").read_text())
    mcfg, tcfg = cfg["model"], cfg["train"]
    out_dir = args.out or (args.run_dir / "diag")
    out_dir.mkdir(exist_ok=True, parents=True)

    # --- model + checkpoint (exact run config) ------------------------------
    f = make_model(**mcfg).to(dev).eval()
    ckpt_path = args.ckpt or (args.run_dir / "ckpt" / f"checkpoint_step_{args.step:06d}.pt")
    sd = torch.load(ckpt_path, map_location="cpu")
    f.load_state_dict(sd["f"] if "f" in sd else sd, strict=False)
    step = int(sd.get("step", args.step)) if isinstance(sd, dict) else args.step
    print(f"[load] {ckpt_path.name}  step={step}")

    # --- scene / cameras ----------------------------------------------------
    down = tcfg.get("down", 1)
    views = load_views(Path(cfg["scene"]), down=down)
    H, W = views["H"], views["W"]
    K_all   = views["K"].float().to(dev)
    c2w_all = views["c2w"].float().to(dev)
    w2c_all = torch.linalg.inv(c2w_all)
    images  = views["images"].float().to(dev)                 # (V,H,W,3)
    masks   = views["masks"].to(dev) if "masks" in views else None

    n_alt = tcfg["n_alt"]
    alt_nn = precompute_alt_cameras(views, n_alt).to(dev)
    alts = [int(a) for a in alt_nn[args.view].tolist()]
    print(f"[view {args.view}] alt views: {alts}")

    trace_cfg = TraceConfig(**cfg["trace"])

    # NCC params, exactly as training would set them at this step ------------
    P       = tcfg["ncc_patch"]
    half_pix = tcfg["ncc_half_pix"]
    ncc_min  = tcfg.get("ncc_min", 0.0)
    ncc_clr  = tcfg.get("ncc_color", "gray")
    smode    = tcfg.get("sample_mode", "bilinear")
    g_sigma  = tcfg.get("gaussian_sigma", 2.0)
    g_rad    = tcfg.get("gaussian_radius", 1)
    bg0, bg1 = tcfg.get("ncc_bilateral_gamma", 0.0), tcfg.get("ncc_bilateral_gamma_end", 0.0)
    steps    = tcfg.get("steps", 300000)
    if bg0 > 0 and bg1 > 0 and bg1 != bg0:
        eff_bgamma = bg0 * (bg1 / bg0) ** (step / max(steps - 1, 1))
    else:
        eff_bgamma = bg0
    # gaussian sigma anneal only active with sample_mode=="gaussian"
    eff_sigma = g_sigma
    if smode == "gaussian":
        s1 = tcfg.get("gaussian_sigma_end", g_sigma)
        if s1 > 0 and s1 != g_sigma:
            eff_sigma = g_sigma * (s1 / g_sigma) ** (step / max(steps - 1, 1))
    print(f"[ncc] patch={P} half_pix={half_pix} bilateral_gamma={eff_bgamma:.4f} "
          f"sample={smode} sigma={eff_sigma:.3f}")

    def ncc(x3d, nrm, vi_b):
        """Per-point ZNCC of ref view vs one alt (NaN where patch out-of-frame)."""
        B = x3d.shape[0]
        va = torch.full((B,), args.view, device=dev, dtype=torch.long)
        vb = torch.full((B,), vi_b, device=dev, dtype=torch.long)
        _, _, _, zf = pmvs_ncc_loss(
            images, x3d, nrm, va, vb, K_all, w2c_all, H, W,
            P, half_pix, smode, eff_sigma, g_rad, ncc_min,
            return_full=True, ncc_color=ncc_clr,
            patch_bilateral_gamma=eff_bgamma)
        return zf                                              # (B,) NaN-padded

    # --- trace the reference view, normals, roughness, texture --------------
    K = views["K"][args.view].numpy(); c2w = views["c2w"][args.view].numpy()
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xf = (xs + .5) * down - .5; yf = (ys + .5) * down - .5
    d_cam = np.stack([(xf - K[0, 2]) / K[0, 0], (yf - K[1, 2]) / K[1, 1],
                      np.ones_like(xf)], -1)
    d_w = d_cam @ c2w[:3, :3].T
    d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
    o_t = torch.from_numpy(np.broadcast_to(c2w[:3, 3], d_w.shape).copy()
                           .reshape(-1, 3)).float().to(dev)
    d_t = torch.from_numpy(d_w.reshape(-1, 3)).float().to(dev)

    x_hit, t_hit, hit = trace_chunked(f, o_t, d_t, trace_cfg)
    nrm = grad_normals(f, x_hit)                              # (HW,3)

    nmap = nrm.reshape(H, W, 3)
    # roughness = 1 - cos(n, locally-averaged n)  (normal high-pass)
    nm = nmap.permute(2, 0, 1)[None]                          # (1,3,H,W)
    n_avg = F.avg_pool2d(nm, 5, stride=1, padding=2)[0].permute(1, 2, 0)
    n_avg = n_avg / n_avg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    rough = (1 - (nmap * n_avg).sum(-1)).clamp(min=0).cpu().numpy()    # (H,W)

    gray = images[args.view].mean(-1)                         # (H,W)
    gx = gray[:, 1:] - gray[:, :-1]; gy = gray[1:, :] - gray[:-1, :]
    tex = torch.zeros(H, W, device=dev)
    tex[:, :-1] += gx.abs(); tex[:-1, :] += gy.abs()
    tex = tex.cpu().numpy()

    hit_np = hit.cpu().numpy().reshape(H, W)
    if masks is not None:
        fg = masks[args.view].cpu().numpy().astype(bool)
        if fg.shape != (H, W):
            fg = fg.reshape(H, W)
    else:
        fg = images[args.view].sum(-1).cpu().numpy().reshape(H, W) < 2.95
    base = hit_np & fg
    tex_thr = np.median(tex[base]) if base.any() else 0.0
    textured = base & (tex > tex_thr)                         # control for texture

    bump  = greedy_pick(rough, textured, args.n_bump,  args.min_dist, H, W, largest=True)
    clean = greedy_pick(np.where(textured, rough, np.inf), textured,
                        args.n_clean, args.min_dist, H, W, largest=False)
    picks = [("bump", p) for p in bump] + [("clean", p) for p in clean]
    print(f"[picks] {len(bump)} bump, {len(clean)} clean (textured fg)")

    # --- depth sweep per picked pixel ---------------------------------------
    cos_thr = tcfg.get("cos_thresh", 0.1)
    occ_md  = tcfg.get("occ_mode", "from_hit")
    A = len(alts)
    gray_all = images.mean(-1)                                # (V,H,W) for saturation probe

    def sat_at(a, uv):
        """near-saturation flag + gray at a reprojected uv (T,2) in alt view a."""
        uvc = uv.long().clamp(min=0)
        uvc[:, 0].clamp_(max=W - 1); uvc[:, 1].clamp_(max=H - 1)
        g = gray_all[a, uvc[:, 1], uvc[:, 0]]
        return g

    results = []
    for kind, p in picks:
        o_p = o_t[p]; d_p = d_t[p]; t0 = float(t_hit[p])
        ts = torch.linspace(t0 - args.band, t0 + args.band, args.n_depth, device=dev)
        x = o_p[None] + ts[:, None] * d_p[None]               # (T,3)
        n_t = grad_normals(f, x)                              # (T,3) live level-set normals
        with torch.no_grad():
            sdf = f(x).cpu().numpy()                          # (T,)
        z    = np.full((A, args.n_depth), np.nan)
        gate = np.zeros((A, args.n_depth), bool)
        cosA = np.full((A, args.n_depth), np.nan)
        parA = np.full((A, args.n_depth), np.nan)
        occA = np.zeros((A, args.n_depth), bool)
        for j, a in enumerate(alts):
            z[j] = ncc(x, n_t, a).detach().cpu().numpy()
            g, c, par, occ, inb, fg = alt_gate(
                f, x, n_t, a, o_p, K_all, w2c_all, c2w_all, masks, H, W,
                cos_thr, trace_cfg, occ_md)
            gate[j] = g.cpu().numpy(); cosA[j] = c.cpu().numpy()
            parA[j] = par.cpu().numpy(); occA[j] = occ.cpu().numpy()
        z_gated = np.where(gate, z, np.nan)
        zmean       = np.nanmean(z, axis=0)                   # RAW: mean over in-frame alts
        zmean_gated = np.nanmean(z_gated, axis=0)             # TRAINING: mean over gated alts
        ncov  = np.isfinite(z).sum(0)
        ngate = gate.sum(0)
        # saturation of each alt at the rendered surface (depth t0)
        si = int(np.argmin(np.abs(ts.cpu().numpy() - t0)))
        sat = np.full(A, np.nan)
        for j, a in enumerate(alts):
            R = w2c_all[a, :3, :3]; tt = w2c_all[a, :3, 3]
            xc = x[si:si + 1] @ R.T + tt
            uvh = xc @ K_all[a].T; uv = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)
            sat[j] = float(sat_at(a, uv)[0])
        results.append(dict(kind=kind, p=p, y=p // W, x=p % W, t0=t0, si=si,
                            ts=ts.cpu().numpy(), z=z, zmean=zmean,
                            zmean_gated=zmean_gated, gate=gate, cosA=cosA,
                            parA=parA, occA=occA, sat=sat, ncov=ncov,
                            ngate=ngate, sdf=sdf, alts=np.array(alts)))

    # --- figure 1: where the picks are --------------------------------------
    fig1, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(images[args.view].cpu().numpy()); ax[0].set_title(f"view {args.view}")
    r = ax[1].imshow(rough, cmap="magma"); ax[1].set_title("normal roughness")
    plt.colorbar(r, ax=ax[1], fraction=.046)
    ax[2].imshow(np.log1p(tex), cmap="gray"); ax[2].set_title("image texture energy")
    for a in ax:
        for kind, p in picks:
            c = "r" if kind == "bump" else "c"
            a.plot(p % W, p // W, "o", mfc="none", mec=c, ms=10, mew=2)
    fig1.tight_layout(); fig1.savefig(out_dir / f"ncc_picks_v{args.view}_s{step}.png", dpi=110)

    # --- figure 2: the profiles ---------------------------------------------
    ncol = max(args.n_bump, args.n_clean)
    fig2, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 8), squeeze=False)
    for r_i, kind in enumerate(("bump", "clean")):
        rows = [d for d in results if d["kind"] == kind]
        for c_i in range(ncol):
            a = axes[r_i][c_i]
            if c_i >= len(rows):
                a.axis("off"); continue
            d = rows[c_i]; dt = d["ts"] - d["t0"]
            for j in range(d["z"].shape[0]):
                a.plot(dt, d["z"][j], color="0.85", lw=.6)
            a.plot(dt, d["zmean"], "b", lw=1.5, label="mean (in-frame)")
            a.plot(dt, d["zmean_gated"], color="purple", lw=2, label="mean (gated=training)")
            a.axvline(0, color="r", ls="--", lw=1, label="rendered surf")
            sdf = d["sdf"]; zc = np.where(np.diff(np.sign(sdf)) != 0)[0]
            for k in zc:
                a.axvline(dt[k], color="g", ls=":", lw=1)
            a2 = a.twinx(); a2.plot(dt, d["ngate"], color="orange", lw=.8, alpha=.6)
            a2.set_ylim(0, len(alts)); a2.set_ylabel("#gated alts", color="orange")
            a.set_title(f"{kind} (y={d['y']},x={d['x']})  t0={d['t0']:.3f}", fontsize=9)
            a.set_xlabel("depth - t0  (world units)"); a.set_ylabel("ZNCC")
            if c_i == 0 and r_i == 0:
                a.legend(fontsize=7, loc="lower left")
    fig2.suptitle(f"NCC vs depth — view {args.view}, step {step}  "
                  f"(blue=in-frame mean, purple=TRAINING gated mean, "
                  f"red=rendered surf, green=SDF zero, orange=#gated alts)")
    fig2.tight_layout(); fig2.savefig(out_dir / f"ncc_profile_v{args.view}_s{step}.png", dpi=110)

    np.savez(out_dir / f"ncc_profile_v{args.view}_s{step}.npz",
             results=np.array(results, dtype=object), alts=np.array(alts))

    # --- paper-grade attribution per bump pixel -----------------------------
    # Decide whether the low-ZNCC outlier alts are NOISE (occluded / specular /
    # out-of-fg -> ncc_topk safe) or TRUTH-TELLERS (clean, unoccluded, WIDE
    # baseline that correctly veto an aliased match seen by a clustered majority
    # -> ncc_topk would lock in the wrong surface).
    def peak_depth(profile, dt):
        zf = np.where(np.isfinite(profile), profile, -np.inf)
        return dt[int(np.argmax(zf))] if np.isfinite(profile).any() else np.nan

    print("\n=== per-alt table @ rendered surface (bump pixels) ===")
    for d in [r for r in results if r["kind"] == "bump"]:
        si = d["si"]; dt = d["ts"] - d["t0"]
        print(f"\n  pixel (y={d['y']},x={d['x']})  t0={d['t0']:.3f}")
        print(f"  {'alt':>4} {'ZNCC':>6} {'cos':>5} {'parlx°':>7} {'occl':>5} "
              f"{'gated':>6} {'sat':>5}")
        zsurf = d["z"][:, si]
        for j, a in enumerate(d["alts"]):
            print(f"  {int(a):>4} {zsurf[j]:6.2f} {d['cosA'][j, si]:5.2f} "
                  f"{np.degrees(d['parA'][j, si]):7.1f} "
                  f"{str(bool(d['occA'][j, si])):>5} {str(bool(d['gate'][j, si])):>6} "
                  f"{d['sat'][j]:5.2f}")
        # classify the outliers = alts with low ZNCC at the *peak-of-gated* depth
        zg = d["zmean_gated"]
        pj = int(np.nanargmax(np.where(np.isfinite(zg), zg, -np.inf))) \
            if np.isfinite(zg).any() else si
        za = d["z"][:, pj]
        outl = np.where(za < 0.3)[0]                          # disagreeing views
        if outl.size == 0:
            cls = "no outliers"
        else:
            noisy = ((d["occA"][outl, pj]) | (~d["gate"][outl, pj])
                     | (d["sat"][outl] > 0.95) | (d["sat"][outl] < 0.05))
            par_out = np.degrees(d["parA"][outl, pj])
            par_in  = np.degrees(d["parA"][za >= 0.3, pj]) if (za >= 0.3).any() else np.array([0.])
            wide = par_out.mean() > (par_in.mean() + 3)       # outliers wider-baseline
            if noisy.mean() >= 0.5:
                cls = "NOISE outliers (occl/specular/out-fg) -> ncc_topk SAFE"
            elif wide:
                cls = "WIDE-BASELINE outliers veto -> possible ALIASING, topk RISKY"
            else:
                cls = "clean narrow-baseline disagreement -> WEAK signal / PE capacity"
        # wide-baseline-only vs topk-only peak: do they agree?
        par_surf = d["parA"][:, si]
        wide_set = par_surf >= np.nanmedian(par_surf)
        z_wide = np.nanmean(np.where(wide_set[:, None], d["z"], np.nan), axis=0)
        K = min(3, d["z"].shape[0])
        z_topk = np.sort(np.where(np.isfinite(d["z"]), d["z"], -np.inf), axis=0)[::-1][:K].mean(0)
        pw, ptk = peak_depth(z_wide, dt), peak_depth(z_topk, dt)
        disagree = abs(pw - ptk)
        print(f"  -> outlier class: {cls}")
        print(f"  -> peak(wide-baseline)={pw:+.4f}  peak(top{K})={ptk:+.4f}  "
              f"|Δ|={disagree:.4f}  ({'AGREE' if disagree <= 3*trace_cfg.eps else 'DISAGREE=aliasing risk'})")
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
