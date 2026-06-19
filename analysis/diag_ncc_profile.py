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
from lip_tracer import data as _data
from lip_tracer.data import load_views, precompute_alt_cameras
from lip_tracer.sphere_tracing import trace_nograd, trace_idr, TraceConfig
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


def make_validpatch_figure(zr, zb, inside, ncc_min, out_dir, view, step,
                           labels=("on-object", "background"), suffix=""):
    """ICLR figure: spatial distribution + budget of KEPT NCC patches under
    mask-baked-black bg vs real bg (true mask-free), split by `inside` (H,W bool).
    zr/zb = (H,W) per-pixel ZNCC (NaN where patch invalid/untextured). `inside`
    is the object region — GT mask, or a mask-free COLMAP ROI. A 'kept' patch =
    finite ZNCC > ncc_min."""
    lin, lout = labels
    keptb = np.isfinite(zb) & (zb > ncc_min)
    keptr = np.isfinite(zr) & (zr > ncc_min)
    cnt = lambda k: (int((k & inside).sum()), int((k & ~inside).sum()))
    (bi, bo), (ri, ro) = cnt(keptb), cnt(keptr)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for a, (k, t, ci, co) in zip(ax[:2],
            [(keptb, "with masked bg (baked-black)", bi, bo),
             (keptr, "true mask-free (real bg)",     ri, ro)]):
        rgb = np.ones((*k.shape, 3))
        rgb[k & inside]  = [0.13, 0.55, 0.13]   # green: kept patch inside region
        rgb[k & ~inside] = [0.85, 0.10, 0.10]   # red:   kept patch outside region
        a.imshow(rgb)
        a.contour(inside.astype(float), levels=[0.5], colors="k", linewidths=.6)
        a.set_title(f"{t}\n{ci:,} {lin}   {co:,} {lout}", fontsize=10)
        a.axis("off")
    x = np.arange(2)
    ax[2].bar(x - .18, [bi, ri], .36, label=lin,  color="#2a8a2a")
    ax[2].bar(x + .18, [bo, ro], .36, label=lout, color="#d22222")
    ax[2].set_xticks(x); ax[2].set_xticklabels(["masked bg", "real bg"])
    ax[2].set_ylabel("# kept NCC patches"); ax[2].set_title("kept-patch budget")
    ax[2].legend(frameon=False)
    fig.suptitle(f"Distribution of valid (kept) NCC patches — view {view}, step {step}  "
                 f"({lout} patches: {bo:,} -> {ro:,})")
    fig.tight_layout()
    p = out_dir / f"validpatch_dist{suffix}_v{view}_s{step}.png"
    fig.savefig(p, dpi=130)
    print(f"[validpatch{suffix}] {lin} {bi:,}->{ri:,}  {lout} {bo:,}->{ro:,}  -> {p}")
    return p


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
    ap.add_argument("--replot-from", type=Path, default=None,
                    help="skip trace; load a saved ncc_bg_ablation_*.npz and (re)emit "
                         "the valid-patch distribution figure on CPU (no GPU needed).")
    ap.add_argument("--roi-validpatch", action="store_true",
                    help="redo validpatch figure with a MASK-FREE split: COLMAP sparse "
                         "points -> loose union-of-balls 3D ROI; a hit is 'inside' if its "
                         "3D position is within --roi-dilate of the cloud. Needs --replot-from "
                         "for the ZNCC maps (re-traces only for the 3D hit positions).")
    ap.add_argument("--roi-dilate", type=float, default=-1.0,
                    help="ROI ball radius in world units (loose). <0 => auto = 6x median "
                         "nearest-neighbour spacing of the COLMAP cloud.")
    ap.add_argument("--roi-min-views", type=int, default=0,
                    help="clean the COLMAP cloud before building the ROI: keep only points "
                         "seen IN-FRAME by >= this many cameras (mask-free, no mask gating). "
                         "0 = no cleaning. Kills stray low-track floaters.")
    ap.add_argument("--grad-conflict", action="store_true",
                    help="ICLR (b): θ-gradient conflict between object-hit and background-hit "
                         "NCC patches. Traces with trace_idr (differentiable), backprops the "
                         "NCC loss for each subset separately, reports cos(g_obj,g_extra) and "
                         "||g_extra||/||g_obj||.")
    ap.add_argument("--gc-n", type=int, default=30000,
                    help="max hits sampled per subset for --grad-conflict")
    ap.add_argument("--object-mask-dir", type=str, default="",
                    help="subdir of the scene to use as the OBJECT mask for the validpatch "
                         "split (e.g. mask_statue for TnT, where load_views' default mask/ is "
                         "only not-sky). Empty = use the loaded mask. Files matched in sorted "
                         "order to match load_views indexing.")
    ap.add_argument("--bg-ablation", action="store_true",
                    help="ICLR figure: one frozen ckpt, per-pixel mean ZNCC computed "
                         "twice (real bg vs mask-baked-black bg) so the two image sets "
                         "differ ONLY outside the mask (the pedestal). Dumps "
                         "ncc_bg_ablation_*.png/.npz and the silhouette-band ΔZNCC.")
    args = ap.parse_args()

    # The baked-vs-real contrast must come from images that differ ONLY outside
    # the mask, so for the ablation we always load the REAL background and bake
    # the black version ourselves below (load_views must not pre-bake).
    if args.bg_ablation or args.grad_conflict:
        _data.BAKE_BACKGROUND = False

    dev = args.device
    cfg = json.loads((args.run_dir / "config.json").read_text())
    mcfg, tcfg = cfg["model"], cfg["train"]
    out_dir = args.out or (args.run_dir / "diag")
    out_dir.mkdir(exist_ok=True, parents=True)

    # --- fast path: replot the valid-patch figure from a saved npz (no GPU) --
    if args.replot_from is not None and not args.roi_validpatch:
        from PIL import Image
        d = np.load(args.replot_from, allow_pickle=True)
        zr, zb = d["zncc_real"], d["zncc_baked"]
        view, step = int(d["view"]), int(d["step"])
        H, W = zr.shape
        m = np.array(Image.open(Path(cfg["scene"]) / "mask" / f"{view:03d}.png").convert("L"))
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m).resize((W, H)))
        make_validpatch_figure(zr, zb, m > 127, tcfg.get("ncc_min", 0.0),
                               out_dir, view, step)
        return

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

    def ncc(x3d, nrm, vi_b, imgs=None):
        """Per-point ZNCC of ref view vs one alt (NaN where patch out-of-frame)."""
        imgs = images if imgs is None else imgs
        B = x3d.shape[0]
        va = torch.full((B,), args.view, device=dev, dtype=torch.long)
        vb = torch.full((B,), vi_b, device=dev, dtype=torch.long)
        _, _, _, zf = pmvs_ncc_loss(
            imgs, x3d, nrm, va, vb, K_all, w2c_all, H, W,
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

    # ===== ICLR (b): θ-gradient conflict, object-hit vs background-hit NCC ====
    # Are the ~570k extra background patches noise/conflict in PARAMETER space?
    # Trace differentiably (trace_idr, the surface point training uses), then
    # backprop the NCC loss for each subset separately and compare the param
    # gradients: cos(g_obj,g_extra) (alignment) and ||g_extra||/||g_obj|| (who
    # dominates the update). fg/bg split uses the GT mask only to LABEL points.
    if args.grad_conflict:
        assert masks is not None, "--grad-conflict needs masks to label object vs background"
        params = [p for p in f.parameters() if p.requires_grad]
        with torch.no_grad():
            _, _, hit0 = trace_chunked(f, o_t, d_t, trace_cfg)
        fg_flat = masks[args.view].reshape(-1).bool().to(dev)
        obj_idx = torch.where(hit0 & fg_flat)[0]
        bg_idx  = torch.where(hit0 & ~fg_flat)[0]
        g = torch.Generator(device=dev).manual_seed(0)
        sub = lambda ix: ix[torch.randperm(ix.numel(), generator=g, device=dev)[:args.gc_n]]
        obj_idx, bg_idx = sub(obj_idx), sub(bg_idx)
        print(f"[grad-conflict] object hits={obj_idx.numel()}  background hits={bg_idx.numel()}")

        import dataclasses as _dc
        tc_idr = _dc.replace(trace_cfg, sdf_min_beta=0.0)   # compacted trace_idr rejects sdf_min_beta>0
        sel = torch.cat([obj_idx, bg_idx])
        x_th, _, hsel, _, n_raw, _, _ = trace_idr(f, o_t[sel], d_t[sel], tc_idr,
                                                  collect_eik=False, diff_normal=False)
        n_th = n_raw / n_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        no = obj_idx.numel()

        def ncc_loss(x, n):
            """Differentiable mean NCC loss (1-ZNCC over kept) summed across alts."""
            tot = x.new_zeros(()); nk = 0
            va = torch.full((x.shape[0],), args.view, device=dev, dtype=torch.long)
            for a in alts:
                vb = torch.full((x.shape[0],), a, device=dev, dtype=torch.long)
                zncc, keep, _ = pmvs_ncc_loss(
                    images, x, n, va, vb, K_all, w2c_all, H, W,
                    P, half_pix, smode, eff_sigma, g_rad, ncc_min,
                    ncc_color=ncc_clr, patch_bilateral_gamma=eff_bgamma)
                if zncc.numel() and keep.any():
                    tot = tot + (1.0 - zncc[keep]).sum(); nk += int(keep.sum())
            return tot / max(nk, 1), nk

        L_obj, nk_o = ncc_loss(x_th[:no], n_th[:no])
        L_bg,  nk_b = ncc_loss(x_th[no:], n_th[no:])
        z = lambda gi, p: gi if gi is not None else torch.zeros_like(p)
        go_l = [z(gi, p) for gi, p in zip(torch.autograd.grad(L_obj, params, retain_graph=True, allow_unused=True), params)]
        gb_l = [z(gi, p) for gi, p in zip(torch.autograd.grad(L_bg,  params, allow_unused=True), params)]
        g_obj = torch.cat([g.reshape(-1) for g in go_l]); g_bg = torch.cat([g.reshape(-1) for g in gb_l])
        no_, nb_ = g_obj.norm().item(), g_bg.norm().item()
        ratio = nb_ / max(no_, 1e-12)
        cos_global = float((g_obj @ g_bg) / max(no_ * nb_, 1e-12))
        # per-tensor cosine (robust to one dominant param, e.g. the SDF output scale)
        ct, wt = [], []
        for go, gb in zip(go_l, gb_l):
            a, b = go.norm().item(), gb.norm().item()
            if a > 1e-9 and b > 1e-9:
                ct.append(float((go.flatten() @ gb.flatten()) / (a * b))); wt.append(a * b)
        ct, wt = np.array(ct), np.array(wt)
        cos_med  = float(np.median(ct)) if ct.size else float("nan")
        cos_wmean = float((ct * wt).sum() / wt.sum()) if ct.size else float("nan")
        frac_neg = float((ct < -0.2).mean()) if ct.size else float("nan")
        verdict = ("ALIGNED" if cos_wmean > 0.2 else
                   "ORTHOGONAL (independent)" if abs(cos_wmean) <= 0.2 else "CONFLICTING")
        print(f"[grad-conflict] kept patches: obj={nk_o} bg={nk_b}  hits {no}/{no}")
        print(f"[grad-conflict] ||g_obj||={no_:.3e} ||g_extra||={nb_:.3e} ratio={ratio:.2f}")
        print(f"[grad-conflict] cos: global={cos_global:+.3f} (scale-dominated)  "
              f"per-tensor median={cos_med:+.3f} normw-mean={cos_wmean:+.3f}  "
              f"frac<-0.2={frac_neg:.2f}  -> {verdict}")

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        ax[0].bar([0, 1], [no_, nb_], color=["#2a8a2a", "#d22222"])
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["object\npatches", "background\npatches"])
        ax[0].set_ylabel("||∂L_NCC/∂θ||  (per-patch mean)")
        ax[0].set_title(f"gradient norm  (||extra||/||obj||={ratio:.2f}×)")
        ax[1].hist(ct, bins=24, range=(-1, 1), color="#666", weights=wt)
        ax[1].axvline(cos_wmean, color="r", lw=2, label=f"norm-wt mean={cos_wmean:+.2f}")
        ax[1].axvline(0, color="k", lw=.6)
        ax[1].set_xlabel("cos(g_obj, g_extra)  per parameter tensor")
        ax[1].set_ylabel("Σ‖g‖ weight"); ax[1].set_title("per-tensor gradient alignment")
        ax[1].legend(fontsize=9)
        ax[2].axis("off")
        ax[2].text(0.5, 0.7, f"per-tensor median\ncos = {cos_med:+.3f}", ha="center",
                   fontsize=16, transform=ax[2].transAxes)
        ax[2].text(0.5, 0.4, f"global cos = {cos_global:+.3f}\n(saturated by SDF output scale)",
                   ha="center", fontsize=10, color="#777", transform=ax[2].transAxes)
        ax[2].text(0.5, 0.15, verdict, ha="center", fontsize=13, color="#444",
                   transform=ax[2].transAxes)
        fig.suptitle(f"NCC θ-gradient conflict — view {args.view}, step {step}  "
                     f"(obj n={nk_o}, bg n={nk_b})")
        fig.tight_layout()
        p = out_dir / f"grad_conflict_v{args.view}_s{step}.png"
        fig.savefig(p, dpi=130); print(f"[done] -> {p}")
        np.savez(out_dir / f"grad_conflict_v{args.view}_s{step}.npz",
                 cos_global=cos_global, cos_median=cos_med, cos_wmean=cos_wmean,
                 frac_neg=frac_neg, ratio=ratio, g_obj_norm=no_, g_bg_norm=nb_,
                 cos_per_tensor=ct, weight_per_tensor=wt,
                 nk_obj=nk_o, nk_bg=nk_b, view=args.view, step=step)
        return
    # =========================================================================

    x_hit, t_hit, hit = trace_chunked(f, o_t, d_t, trace_cfg)
    nrm = grad_normals(f, x_hit)                              # (HW,3)

    # ===== mask-free ROI variant of the valid-patch figure ===================
    # Split kept patches by whether their 3D hit lies inside a loose COLMAP-cloud
    # ROI (union of balls), instead of the GT mask. Shows whether a mask-free
    # object region rejects the background pedestal patches the mask did.
    if args.roi_validpatch:
        assert args.replot_from is not None, "--roi-validpatch needs --replot-from <npz> for ZNCC maps"
        from scipy.spatial import cKDTree
        from lip_tracer.data import load_colmap_points
        d = np.load(args.replot_from, allow_pickle=True)
        zr, zb = d["zncc_real"], d["zncc_baked"]
        view, step = int(d["view"]), int(d["step"])
        pts_t = load_colmap_points(Path(cfg["scene"]))[:, :3].float()
        if args.roi_min_views > 0:
            # mask-free in-frame visibility count (drop the mask gating of
            # data.colmap_visibility_counts): how many cameras see each point.
            c2w_v = views["c2w"].float(); w2c_v = torch.linalg.inv(c2w_v)
            Rv, tv = w2c_v[:, :3, :3], w2c_v[:, :3, 3]; Kv = views["K"].float()
            cnt = torch.zeros(len(pts_t), dtype=torch.long)
            for i in range(0, len(pts_t), 8192):
                p   = pts_t[i:i + 8192]
                xc  = torch.einsum("vij,nj->vni", Rv, p) + tv[:, None, :]
                uvh = torch.einsum("vij,vnj->vni", Kv, xc)
                uv  = uvh[..., :2] / uvh[..., 2:3].clamp_min(1e-6)
                inb = ((uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0)
                       & (uv[..., 1] < H) & (xc[..., 2] > 1e-4))
                cnt[i:i + 8192] = inb.sum(0)
            keep = cnt >= args.roi_min_views
            print(f"[roi] min_views={args.roi_min_views}: kept {int(keep.sum())}/{len(pts_t)} colmap pts")
            pts_t = pts_t[keep]
        pts = pts_t.numpy().astype(np.float32)
        tree = cKDTree(pts)
        tau = args.roi_dilate
        if tau < 0:                                          # auto: 6x median NN spacing
            nn, _ = tree.query(pts[np.random.default_rng(0).choice(len(pts), min(4000, len(pts)), replace=False)], k=2)
            tau = 6.0 * float(np.median(nn[:, 1]))
        xh = x_hit.detach().cpu().numpy().astype(np.float32)
        dist, _ = tree.query(xh, k=1)
        inside = (dist < tau).reshape(H, W) & hit.cpu().numpy().reshape(H, W)
        print(f"[roi] COLMAP pts={len(pts)}  ball radius tau={tau:.4f}  "
              f"hits inside ROI={int(inside.sum())}")
        roi_suffix = ("_colmaproi"
                      + (f"_mv{args.roi_min_views}" if args.roi_min_views > 0 else "")
                      + (f"_d{args.roi_dilate:g}" if args.roi_dilate > 0 else ""))
        make_validpatch_figure(zr, zb, inside, tcfg.get("ncc_min", 0.0), out_dir,
                               view, step, labels=("inside ROI", "outside ROI"),
                               suffix=roi_suffix)
        return
    # =========================================================================

    # ===== ICLR background ablation ========================================
    # One frozen ckpt (geometry fixed). Compute per-pixel mean ZNCC over the
    # alt views TWICE: with the real photographed background, and with the
    # mask-baked-black background. The two image sets are identical inside the
    # mask, so any ZNCC change is purely the pedestal/surround re-entering the
    # P*P patch footprint. Prediction: a degradation band hugging the
    # silhouette (where patches straddle object<->pedestal), interior unchanged.
    if args.bg_ablation:
        assert masks is not None, "--bg-ablation needs object masks to define the baked bg"
        mask_bool   = masks.bool()                            # (V,H,W)
        images_real = images                                  # loaded BAKE_BACKGROUND=False
        images_baked = images.clone(); images_baked[~mask_bool] = 0.0

        hit_idx = torch.where(hit)[0]
        xh, nh  = x_hit[hit_idx], nrm[hit_idx]

        def mean_zncc_over_alts(imgs, chunk=65536):
            acc = torch.full((hit_idx.shape[0], len(alts)), float("nan"), device=dev)
            for j, a in enumerate(alts):
                parts = [ncc(xh[i:i + chunk], nh[i:i + chunk], a, imgs)
                         for i in range(0, xh.shape[0], chunk)]
                acc[:, j] = torch.cat(parts)
            return acc.nanmean(dim=1)                          # (Nhit,)

        z_real, z_baked = mean_zncc_over_alts(images_real), mean_zncc_over_alts(images_baked)

        def scatter(v):
            full = torch.full((H * W,), float("nan"), device=dev); full[hit_idx] = v
            return full.reshape(H, W).cpu().numpy()
        zr, zb = scatter(z_real), scatter(z_baked)
        zd = zb - zr                                           # masked - real: >0 => bg costs ZNCC

        fg_v = mask_bool[args.view].cpu().numpy().astype(float)
        if args.object_mask_dir:
            # override the validpatch split with a dedicated object mask (e.g.
            # mask_statue on TnT); sorted-glob to match load_views ordering.
            from PIL import Image as _Im
            _mps = sorted(p for p in (Path(cfg["scene"]) / args.object_mask_dir).glob("*.png")
                          if not p.name.startswith("._"))
            _om = np.array(_Im.open(_mps[args.view]).convert("L"))
            if _om.shape != (H, W):
                _om = np.array(_Im.fromarray(_om).resize((W, H)))
            fg_v = (_om > 127).astype(float)
            print(f"[bg-ablation] validpatch split uses object mask {args.object_mask_dir}/"
                  f"{_mps[args.view].name} (coverage {fg_v.mean():.3f})")
        fig, ax = plt.subplots(1, 4, figsize=(22, 6))
        ax[0].imshow(images_real[args.view].cpu().numpy()); ax[0].set_title(f"view {args.view} (real bg)")
        for k, (img, ttl) in enumerate([(zb, "ZNCC  baked-black bg (with-mask)"),
                                        (zr, "ZNCC  real bg (true mask-free)")]):
            im = ax[k + 1].imshow(img, cmap="viridis", vmin=-1, vmax=1)
            ax[k + 1].set_title(ttl); plt.colorbar(im, ax=ax[k + 1], fraction=.046)
        im = ax[3].imshow(zd, cmap="RdBu_r", vmin=-1, vmax=1)
        ax[3].set_title("Δ ZNCC  (masked − real)\nred = background degrades the match")
        plt.colorbar(im, ax=ax[3], fraction=.046)
        ax[3].contour(fg_v, levels=[0.5], colors="k", linewidths=0.8)   # silhouette
        for a in ax:
            a.axis("off")
        fig.suptitle(f"Background ablation — frozen ckpt step {step}; images differ "
                     f"ONLY outside the mask (pedestal)")
        fig.tight_layout()
        outp = out_dir / f"ncc_bg_ablation_v{args.view}_s{step}.png"
        fig.savefig(outp, dpi=120)
        np.savez(out_dir / f"ncc_bg_ablation_v{args.view}_s{step}.npz",
                 zncc_real=zr, zncc_baked=zb, zncc_diff=zd, view=args.view, step=step)
        make_validpatch_figure(zr, zb, fg_v > 0.5, ncc_min, out_dir, args.view, step)
        try:
            from scipy import ndimage
            fgb  = fg_v > 0.5
            band = (ndimage.binary_dilation(fgb, iterations=8)
                    & ~ndimage.binary_erosion(fgb, iterations=8) & np.isfinite(zd))
            intr = ndimage.binary_erosion(fgb, iterations=12) & np.isfinite(zd)
            bmean, imean = np.nanmean(zd[band]), np.nanmean(zd[intr])
            # Sign-aware verdict. Δ=masked-real: POSITIVE band => real bg LOWERS
            # ZNCC at the silhouette (straddle hypothesis). NEGATIVE/~0 => the bg
            # does NOT corrupt the foreground match (refutes it).
            if bmean > 0.05:
                verdict = "real bg LOWERS boundary ZNCC => straddle plausible"
            elif abs(bmean) < 0.03:
                verdict = "band~0 => bg does NOT corrupt foreground NCC (straddle REFUTED)"
            else:
                verdict = "band negative => real bg slightly HELPS boundary (textured pedestal correlates)"
            print(f"[bg-ablation] ΔZNCC=masked-real  silhouette-band mean={bmean:+.3f}  "
                  f"interior mean={imean:+.3f}  -> {verdict}")
        except Exception as e:
            print(f"[bg-ablation] band stats skipped ({e})")
        print(f"[done] -> {outp}")
        return
    # =======================================================================

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
