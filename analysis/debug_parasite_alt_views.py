#!/usr/bin/env python3
"""Parasite-surface alt-view debugger.

For one surface point (a parasite bump on a flat wall), answer two questions in
a single CVPR-grade grid:

  (1) WHERE do the 6 nearest `n_alt` cameras reproject the point, and is that
      reprojection actually USEFUL — in-frame, foreground, un-occluded, not
      grazing?  (top row: ref view + 6 alt crops, each annotated + ZNCC)

  (2) Are the 6 selected NN a CLUSTERED subset that agrees on an aliased match
      while wider-baseline views disagree?  (bottom: camera-centre map with the
      baselines drawn, and ZNCC-vs-ALL-views ordered by camera distance with the
      6 selected NN marked).

This replicates the training gate exactly (loss.py:399-462): in_frame & fg &
not_occl & cos_ok, with occ_mode/eps from the run config, and uses the same
pmvs_ncc_loss for ZNCC.

Usage:
  python debug_parasite_alt_views.py --run-dir outputs/run_..._scan24_... \
         --view 23                 # auto-pick roughest textured fg bump
  python debug_parasite_alt_views.py --run-dir ... --view 23 --px 410 --py 280
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
from matplotlib.patches import Rectangle

from lip_tracer.model import make_model
from lip_tracer.data import load_views, precompute_alt_cameras
from lip_tracer.sphere_tracing import trace_nograd, TraceConfig
from lip_tracer.loss import pmvs_ncc_loss


def trace_chunked(f, o, d, cfg, chunk=131072):
    xs, ts, hs = [], [], []
    for i in range(0, o.shape[0], chunk):
        with torch.no_grad():
            x, t, h = trace_nograd(f, o[i:i + chunk], d[i:i + chunk], cfg)
        xs.append(x); ts.append(t); hs.append(h)
    return torch.cat(xs), torch.cat(ts), torch.cat(hs)


def grad_normal(f, x, chunk=8192):
    """Unit ∇f at x (N,3), grad enabled, chunked to bound memory."""
    out = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(xc).sum(), xc)[0].detach()
        out.append(g)
    g = torch.cat(out)
    return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint (default: ckpt/checkpoint_latest.pt)")
    ap.add_argument("--view", type=int, default=23, help="reference view index")
    ap.add_argument("--px", type=int, default=-1, help="ref pixel x (default auto)")
    ap.add_argument("--py", type=int, default=-1, help="ref pixel y (default auto)")
    ap.add_argument("--crop", type=int, default=70, help="alt-crop half-window px")
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
    ckpt_path = args.ckpt or (args.run_dir / "ckpt" / "checkpoint_latest.pt")
    sd = torch.load(ckpt_path, map_location="cpu")
    f.load_state_dict(sd["f"] if "f" in sd else sd, strict=False)
    step = int(sd.get("step", -1)) if isinstance(sd, dict) else -1
    print(f"[load] {ckpt_path.name}  step={step}")

    # --- scene / cameras ----------------------------------------------------
    down = tcfg.get("down", 1)
    views = load_views(Path(cfg["scene"]), down=down)
    H, W = int(views["H"]), int(views["W"])
    K_all   = views["K"].float().to(dev)
    c2w_all = views["c2w"].float().to(dev)
    w2c_all = torch.linalg.inv(c2w_all)
    images  = views["images"].float().to(dev)                 # (V,H,W,3)
    masks   = views["masks"].to(dev) if "masks" in views else None
    V = images.shape[0]

    n_alt = tcfg["n_alt"]
    alt_nn = precompute_alt_cameras(views, n_alt).to(dev)
    alts = [int(a) for a in alt_nn[args.view].tolist()]
    print(f"[view {args.view}] {n_alt} NN alt views: {alts}")

    trace_cfg = TraceConfig(**cfg["trace"])
    P        = tcfg["ncc_patch"]
    half_pix = tcfg["ncc_half_pix"]
    ncc_min  = tcfg.get("ncc_min", 0.0)
    ncc_clr  = tcfg.get("ncc_color", "gray")
    smode    = tcfg.get("sample_mode", "bilinear")
    g_sigma  = tcfg.get("gaussian_sigma", 2.0)
    g_rad    = tcfg.get("gaussian_radius", 1)
    cos_thr  = tcfg.get("cos_thresh", 0.1)
    occ_md   = tcfg.get("occ_mode", "from_hit")

    def zncc_against(vb_list, x3d, nrm):
        """ZNCC of ref patch vs each view in vb_list (NaN where out-of-frame)."""
        n = len(vb_list)
        va = torch.full((n,), args.view, device=dev, dtype=torch.long)
        vb = torch.tensor(vb_list, device=dev, dtype=torch.long)
        xb = x3d[None].expand(n, 3).contiguous()
        nb = nrm[None].expand(n, 3).contiguous()
        out = pmvs_ncc_loss(images, xb, nb, va, vb, K_all, w2c_all, H, W,
                            P, half_pix, smode, g_sigma, g_rad, ncc_min,
                            return_full=True, ncc_color=ncc_clr)
        return out[3].detach().cpu().numpy()                  # (n,) NaN-padded

    # --- ray for the chosen pixel -------------------------------------------
    K = views["K"][args.view].numpy(); c2w = views["c2w"][args.view].numpy()

    def ray(px, py):
        xf = (px + .5) * down - .5; yf = (py + .5) * down - .5
        dc = np.array([(xf - K[0, 2]) / K[0, 0], (yf - K[1, 2]) / K[1, 1], 1.0])
        dw = dc @ c2w[:3, :3].T; dw /= np.linalg.norm(dw)
        return (torch.from_numpy(c2w[:3, 3].copy()).float().to(dev),
                torch.from_numpy(dw).float().to(dev))

    # --- pick the pixel: explicit, or auto = roughest textured fg pixel ------
    if args.px >= 0 and args.py >= 0:
        px, py = args.px, args.py
    else:
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xf = (xs + .5) * down - .5; yf = (ys + .5) * down - .5
        dc = np.stack([(xf - K[0, 2]) / K[0, 0], (yf - K[1, 2]) / K[1, 1],
                       np.ones_like(xf)], -1)
        dw = dc @ c2w[:3, :3].T; dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
        o_t = torch.from_numpy(np.broadcast_to(c2w[:3, 3], dw.shape).copy()
                               .reshape(-1, 3)).float().to(dev)
        d_t = torch.from_numpy(dw.reshape(-1, 3)).float().to(dev)
        x_hit, _, hit = trace_chunked(f, o_t, d_t, trace_cfg)
        nrm = grad_normal(f, x_hit).reshape(H, W, 3)
        nm = nrm.permute(2, 0, 1)[None]
        na = F.avg_pool2d(nm, 5, 1, 2)[0].permute(1, 2, 0)
        na = na / na.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rough = (1 - (nrm * na).sum(-1)).clamp(min=0).cpu().numpy()
        gray = images[args.view].mean(-1)
        tex = torch.zeros(H, W, device=dev)
        tex[:, :-1] += (gray[:, 1:] - gray[:, :-1]).abs()
        tex[:-1, :] += (gray[1:, :] - gray[:-1, :]).abs()
        tex = tex.cpu().numpy()
        hit_np = hit.cpu().numpy().reshape(H, W)
        fg = (masks[args.view].cpu().numpy().astype(bool) if masks is not None
              else images[args.view].sum(-1).cpu().numpy() < 2.95)
        base = hit_np & fg
        valid = base & (tex > (np.median(tex[base]) if base.any() else 0))
        p = int(np.where(valid.reshape(-1), rough.reshape(-1), -1).argmax())
        py, px = divmod(p, W)
        print(f"[auto-pick] roughest textured fg pixel -> (px={px}, py={py})")

    o_p, d_p = ray(px, py)
    x, _, hit = trace_chunked(f, o_p[None], d_p[None], trace_cfg)
    if not bool(hit[0]):
        print("[warn] picked ray did not hit the surface!")
    x = x[0]
    n = grad_normal(f, x[None])[0]
    print(f"[surface] x={x.tolist()}  hit={bool(hit[0])}")

    # --- per-view gate (in_frame / fg / occl / cos) + reprojection ----------
    def gate_view(a):
        cam = c2w_all[a, :3, 3]
        dirv = cam - x; dist = dirv.norm().clamp(min=1e-6); dp = dirv / dist
        if occ_md == "from_hit":
            _, tp, hitp = trace_nograd(f, (x + 1e-2 * dp)[None], dp[None], trace_cfg)
            not_occl = bool((~hitp[0]) | (tp[0] > dist - 0.1))
        else:
            _, tp, hitp = trace_nograd(f, cam[None], -dp[None], trace_cfg)
            not_occl = bool(hitp[0] & (dist <= tp[0] + 0.1))
        cos = float((n * dp).sum().abs())
        R = w2c_all[a, :3, :3]; tt = w2c_all[a, :3, 3]
        xc = x @ R.T + tt
        uvh = xc @ K_all[a].T; uv = (uvh[:2] / uvh[2].clamp(min=1e-6)).cpu().numpy()
        inb = bool((xc[2] > 0) and 0 <= uv[0] < W and 0 <= uv[1] < H)
        if masks is not None and inb:
            fgp = bool(masks[a, int(min(uv[1], H - 1)), int(min(uv[0], W - 1))])
        else:
            fgp = inb
        return dict(uv=uv, inb=inb, fg=fgp, not_occl=not_occl,
                    cos_ok=cos > cos_thr, cos=cos, dist=float(dist))

    alt_info = {a: gate_view(a) for a in alts}
    z_alt = zncc_against(alts, x, n)
    for a, z in zip(alts, z_alt):
        alt_info[a]["zncc"] = float(z)

    # --- ZNCC + gate over ALL other views (the "distribution") --------------
    others = [v for v in range(V) if v != args.view]
    z_all = zncc_against(others, x, n)
    dist_all = (c2w_all[others, :3, 3] - c2w_all[args.view, :3, 3]).norm(dim=-1).cpu().numpy()
    fg_all = np.array([gate_view(v)["fg"] and gate_view(v)["not_occl"]
                       and gate_view(v)["cos_ok"] for v in others])

    def status(g):
        if not g["inb"]:       return "OUT-OF-FRAME", "tab:gray"
        if not g["fg"]:        return "BACKGROUND",   "tab:red"
        if not g["not_occl"]:  return "OCCLUDED",     "tab:orange"
        if not g["cos_ok"]:    return "GRAZING",      "gold"
        return "USED", "tab:green"

    # ====================================================================== #
    #  FIGURE
    # ====================================================================== #
    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 0.95], hspace=0.28, wspace=0.12)

    # -- top row: reference view + 6 alt crops -------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(images[args.view].cpu().numpy())
    ax0.plot(px, py, "+", c="red", ms=16, mew=3)
    ax0.add_patch(Rectangle((px - args.crop, py - args.crop), 2 * args.crop,
                            2 * args.crop, fill=False, ec="red", lw=1.5))
    ax0.set_title(f"REF  v{args.view}\n(px={px}, py={py})", fontsize=10, fontweight="bold")
    ax0.axis("off")

    cw = args.crop
    for i, a in enumerate(alts):
        ax = fig.add_subplot(gs[0, i + 1])
        g = alt_info[a]; u, v = g["uv"]
        ax.imshow(images[a].cpu().numpy())
        st, col = status(g)
        if g["inb"]:
            ax.plot(u, v, "+", c=col, ms=16, mew=3)
            ax.set_xlim(u - cw, u + cw); ax.set_ylim(v + cw, v - cw)
        else:
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
        for s in ax.spines.values():
            s.set_color(col); s.set_linewidth(4); s.set_visible(True)
        ax.set_title(f"alt v{a}  Z={g['zncc']:+.2f}\n{st}  (cos={g['cos']:.2f})",
                     fontsize=9, color=col)
        ax.set_xticks([]); ax.set_yticks([])

    # -- bottom-left: camera-centre map with baselines -----------------------
    axm = fig.add_subplot(gs[1, 0:3])
    C = c2w_all[:, :3, 3].cpu().numpy()
    mu = C.mean(0); Cc = C - mu
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    P2 = Cc @ Vt[:2].T                                        # all cams in 2D
    xs2 = (x.cpu().numpy() - mu) @ Vt[:2].T                   # surface point in 2D
    axm.scatter(P2[:, 0], P2[:, 1], s=18, c="0.7", label="all cams")
    for a in alts:
        col = status(alt_info[a])[1]
        axm.plot([P2[a, 0], xs2[0]], [P2[a, 1], xs2[1]], "-", c=col, lw=1.2, alpha=0.8)
        axm.scatter(*P2[a], s=70, c=col, zorder=3, edgecolors="k")
    axm.plot([P2[args.view, 0], xs2[0]], [P2[args.view, 1], xs2[1]], "b--", lw=1.5)
    axm.scatter(*P2[args.view], s=160, marker="*", c="blue", zorder=4,
                edgecolors="k", label=f"ref v{args.view}")
    axm.scatter(*xs2, s=120, marker="X", c="k", zorder=5, label="surface pt")
    axm.set_title("camera distribution (PCA top-down)\n"
                  "lines = baseline ref/alt → surface", fontsize=10)
    axm.set_aspect("equal"); axm.legend(fontsize=8); axm.set_xticks([]); axm.set_yticks([])

    # -- bottom-right: ZNCC vs ALL views, ordered by camera distance ---------
    axb = fig.add_subplot(gs[1, 3:7])
    order = np.argsort(dist_all)
    od = dist_all[order]; oz = z_all[order]; ofg = fg_all[order]
    ov = np.array(others)[order]
    nn_set = set(alts)
    bar_c = ["tab:green" if fg else "tab:red" for fg in ofg]
    axb.bar(np.arange(len(ov)), np.nan_to_num(oz, nan=0.0), color=bar_c,
            edgecolor=["k" if int(v) in nn_set else "none" for v in ov],
            linewidth=[2.2 if int(v) in nn_set else 0 for v in ov])
    for i, v in enumerate(ov):
        if int(v) in nn_set:
            axb.text(i, 1.02, "NN", ha="center", fontsize=7, color="k")
        if not np.isfinite(oz[i]):
            axb.text(i, 0.02, "×", ha="center", fontsize=8, color="0.4")
    axb.axhline(0, c="k", lw=0.6)
    axb.set_ylim(-1.05, 1.15)
    axb.set_xticks(np.arange(len(ov)))
    axb.set_xticklabels([str(int(v)) for v in ov], fontsize=6, rotation=90)
    axb.set_xlabel("view index (sorted by camera distance to ref →)")
    axb.set_ylabel("ZNCC at surface point")
    axb.set_title("ZNCC vs ALL views  (black edge = selected NN; "
                  "green=usable / red=bg·occl·grazing; × = out-of-frame)",
                  fontsize=9)

    fig.suptitle(f"Parasite alt-view audit — run {args.run_dir.name}  "
                 f"step {step}  |  ref v{args.view} px({px},{py})",
                 fontsize=12, fontweight="bold")
    out = out_dir / f"parasite_alts_v{args.view}_p{px}_{py}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[saved] {out}")

    # --- terse text summary --------------------------------------------------
    n_used = sum(1 for a in alts if status(alt_info[a])[0] == "USED")
    print(f"\n  selected NN: {n_used}/{n_alt} usable in training gate")
    print(f"  {'alt':>4} {'dist':>6} {'ZNCC':>6} {'cos':>5}  status")
    for a in alts:
        g = alt_info[a]
        print(f"  {a:>4} {g['dist']:6.3f} {g['zncc']:+6.2f} {g['cos']:5.2f}  {status(g)[0]}")
    fin = np.isfinite(z_all)
    print(f"\n  ALL views: {fin.sum()}/{len(others)} in-frame, "
          f"{fg_all.sum()} usable; "
          f"median ZNCC usable={np.nanmedian(z_all[fg_all]) if fg_all.any() else float('nan'):+.2f}")
    # nearest-k vs best-k: is the NN pool worse than the global best?
    best5 = np.sort(z_all[fin])[::-1][:5].mean() if fin.sum() >= 5 else float("nan")
    nn5 = np.nanmean([alt_info[a]["zncc"] for a in alts])
    print(f"  mean ZNCC of NN pool={nn5:+.2f}  vs  best-5 across all views={best5:+.2f}")


if __name__ == "__main__":
    main()
