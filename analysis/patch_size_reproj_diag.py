#!/usr/bin/env python3
"""Patch-size vs reprojection diagnostic — "are my NCC patches too big at init?"

Init checkpoint ONLY, no training. Asks, per valid ray, whether the training
ZNCC objective — evaluated at several patch settings — actually points the init
surface toward the reference (MVS) surface, and how wide its basin is in
reprojected pixels relative to the init's reprojection error.

Pipeline (per sampled valid ray in a reference view v):
  1. Sphere-trace the INIT SDF  ->  t0, x0          (where the model currently is)
  2. Reference MVS depth        ->  t*, x*          (where the surface should be)
  3. Source-view reprojection error of the init:
         Delta = median_{o' in alts} || pi_o'(x0) - pi_o'(x*) ||   (pixels)
  4. For each patch (P, half_pix) in {(5,2),(7,3),(11,5)}:
       - sweep depth t around t*, x(t)=o+t d, live normals n(t)=∇f(x(t))
       - evaluate the SAME training ZNCC (pmvs_ncc_loss) vs each gated alt view
       - L(t) = mean_gated-alt (1 - ZNCC_a(t))   [training NCC term, ncc_min gate]
       - induced reprojection displacement(t) = median_alt ||pi(x(t))-pi(x*)|| px
       - plot  mean ZNCC  vs  |displacement|  (basin profile, one curve / patch)
       - finite-difference at the INIT surface t0:
             g = (L(t0+eps) - L(t0-eps)) / (2 eps)
         gradient is CORRECT iff (t0 - t*) * g > 0   (descent moves t0 toward t*)

Frame note: for TnT (Courthouse) the MVSFormer++ cams/depths under
<scene>/mvsformer_depth_*/<scan>/{depth_est,cams,confidence} are already in the
NORMALISED training frame (cam centres equal (c_raw - bbox_centre)/bbox_scale).
The script asserts this against load_views() so x* lands on the training ray.

Usage:
  python analysis/patch_size_reproj_diag.py \
      --run-dir outputs/run_20260616_085348_Courthouse_sphere_nomask_arccos_none_5025884 \
      --views 0,140,300,500,700,900 --n-rays 300
  # default --ckpt is the init checkpoint (checkpoint_step_000000.pt)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer.model import make_model
from lip_tracer.data import (load_views, precompute_alt_cameras,
                             precompute_alt_cameras_arccos)
from lip_tracer.sphere_tracing import trace_nograd, TraceConfig
from lip_tracer.loss import pmvs_ncc_loss
from lip_tracer.geomvs import _read_pfm

PATCHES_DEFAULT = "5,2;7,3;11,5"


# ----------------------------------------------------------------- helpers ----
def grad_normals(f, x, chunk=8192):
    """Unit ∇f at each x (B,3), chunked, grad enabled."""
    out = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(xc).sum(), xc)[0].detach()
        out.append(g)
    g = torch.cat(out)
    return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def trace_chunked(f, o, d, cfg, chunk=131072):
    xs, ts, hs = [], [], []
    for i in range(0, o.shape[0], chunk):
        with torch.no_grad():
            x, t, h = trace_nograd(f, o[i:i + chunk], d[i:i + chunk], cfg)
        xs.append(x); ts.append(t); hs.append(h)
    return torch.cat(xs), torch.cat(ts), torch.cat(hs)


def load_mvs_view(scan_root: Path, v: int):
    """Return (depth HxW float32, conf HxW float32, K 3x3, c2w 4x4) for MVS view v,
    all in the normalised training frame (cam.txt already normalised)."""
    pfm = scan_root / "depth_est" / f"{v:08d}.pfm"
    cam = scan_root / "cams" / f"{v:08d}_cam.txt"
    depth = np.asarray(_read_pfm(pfm), dtype=np.float32)
    conf_p = scan_root / "confidence" / f"{v:08d}.npy"
    if conf_p.exists():
        conf = np.load(conf_p).astype(np.float32)
        if conf.dtype == np.uint8 or conf.max() > 1.5:
            conf = conf / 255.0
        if conf.shape != depth.shape:        # confidence sometimes at /4 res
            import cv2
            conf = cv2.resize(conf, (depth.shape[1], depth.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
    else:
        conf = np.ones_like(depth)
    lines = cam.read_text().splitlines()
    ext = np.array([[float(x) for x in lines[i].split()] for i in range(1, 5)])
    K = np.array([[float(x) for x in lines[i].split()] for i in range(7, 10)])
    c2w = np.linalg.inv(ext)
    return depth, conf, K.astype(np.float32), c2w.astype(np.float32)


def sample_bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample img (H,W) at uv (N,2)=(x,y); out-of-bounds -> nan."""
    H, W = img.shape
    x, y = uv[:, 0], uv[:, 1]
    ib = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, W - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, H - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    wx = x - x0; wy = y - y0
    val = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
           + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)
    val = val.astype(np.float32)
    val[~ib] = np.nan
    return val


def project(x3d: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor):
    """x3d (...,3) -> uv (...,2), z (...). K,w2c are (3,3),(4,4)."""
    xc = x3d @ w2c[:3, :3].T + w2c[:3, 3]
    uvh = xc @ K.T
    uv = uvh[..., :2] / uvh[..., 2:3].clamp(min=1e-6)
    return uv, xc[..., 2]


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="run dir holding config.json + ckpt/ (resolve preempted runs by job id)")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint (default: ckpt/checkpoint_step_000000.pt = init)")
    ap.add_argument("--views", default="",
                    help="comma list of reference view indices; empty -> 8 spread across all")
    ap.add_argument("--n-views", type=int, default=8, help="auto-spread count when --views empty")
    ap.add_argument("--n-rays", type=int, default=300, help="valid rays sampled per reference view")
    ap.add_argument("--patches", default=PATCHES_DEFAULT,
                    help="';'-sep 'P,half_pix' patch settings, e.g. '5,2;7,3;11,5'")
    ap.add_argument("--band", type=float, default=0.05, help="±depth sweep half-width (world units)")
    ap.add_argument("--n-depth", type=int, default=41)
    ap.add_argument("--eps", type=float, default=0.002,
                    help="finite-difference step at t0 (world units)")
    ap.add_argument("--conf-thr", type=float, default=0.3, help="MVS confidence threshold for valid rays")
    ap.add_argument("--min-gated", type=int, default=2, help="min gated alt views for a valid ray")
    ap.add_argument("--cov-max", type=int, default=50000,
                    help="cap candidate pixels scored for the coverage map/waterfall/Δ "
                         "per view (uniform subsample; 0 = no cap). Bounds the occlusion-"
                         "trace cost on dense views; the ZNCC sweep is always a separate "
                         "--n-rays subset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    cfg = json.loads((args.run_dir / "config.json").read_text())
    mcfg, tcfg = cfg["model"], cfg["train"]
    out_dir = args.out or (args.run_dir / "diag" / "patch_size")
    out_dir.mkdir(parents=True, exist_ok=True)

    patches = [tuple(int(round(float(z))) if i == 0 else float(z)
                     for i, z in enumerate(p.split(",")))
               for p in args.patches.split(";") if p.strip()]
    patches = [(int(P), float(hp)) for P, hp in patches]
    print(f"[patches] {patches}")

    # --- model + INIT checkpoint -------------------------------------------
    f = make_model(**mcfg).to(dev).eval()
    ckpt_path = args.ckpt or (args.run_dir / "ckpt" / "checkpoint_step_000000.pt")
    sd = torch.load(ckpt_path, map_location="cpu")
    f.load_state_dict(sd["f"] if isinstance(sd, dict) and "f" in sd else sd, strict=False)
    step = int(sd.get("step", 0)) if isinstance(sd, dict) else 0
    print(f"[load] {ckpt_path}  step={step}")

    # --- scene / cameras (training frame) ----------------------------------
    scene = Path(cfg["scene"])
    down = tcfg.get("down", 1)
    views = load_views(scene, down=down)
    H, W = views["H"], views["W"]
    K_all   = views["K"].float().to(dev)
    c2w_all = views["c2w"].float().to(dev)
    w2c_all = torch.linalg.inv(c2w_all)
    images  = views["images"].float().to(dev)               # (V,H,W,3)
    masks   = views["masks"].to(dev) if (tcfg.get("use_masks") and "masks" in views) else None
    V = c2w_all.shape[0]

    n_alt = tcfg["n_alt"]
    vs = tcfg.get("view_selection", "nearest")
    if vs in ("arccos", "arccos_nn"):
        alt_nn = precompute_alt_cameras_arccos(views, n_alt).to(dev)
    else:
        alt_nn = precompute_alt_cameras(views, n_alt).to(dev)
    print(f"[views] V={V} H={H} W={W} n_alt={n_alt} view_selection={vs}")

    trace_cfg = TraceConfig(**cfg["trace"])
    cos_thr = tcfg.get("cos_thresh", 0.1)
    occ_md  = tcfg.get("occ_mode", "from_hit")
    ncc_min = tcfg.get("ncc_min", 0.0)
    ncc_clr = tcfg.get("ncc_color", "gray")
    smode   = tcfg.get("sample_mode", "bilinear")
    g_sigma = tcfg.get("gaussian_sigma", 2.0)
    g_rad   = tcfg.get("gaussian_radius", 1)

    # --- MVS reference (already normalised frame) --------------------------
    mvs_root = None
    for d in sorted(scene.glob("mvsformer_depth_*")):
        cands = [p for p in d.rglob("depth_est") if p.is_dir()]
        if cands:
            mvs_root = cands[0].parent
            break
    if mvs_root is None:
        raise FileNotFoundError(f"no mvsformer_depth_*/<scan>/depth_est under {scene}")
    n_pfm = len(list((mvs_root / "depth_est").glob("*.pfm")))
    print(f"[mvs] {mvs_root}  ({n_pfm} depth maps)")
    assert n_pfm >= V, f"MVS depth count {n_pfm} < {V} training views"

    # sanity: MVS cam center == training (normalised) cam center for view 0
    _, _, _, c2w0_mvs = load_mvs_view(mvs_root, 0)
    dc = float(np.linalg.norm(c2w0_mvs[:3, 3] - c2w_all[0, :3, 3].cpu().numpy()))
    assert dc < 1e-2, (f"MVS/train frame mismatch (center diff {dc:.3f}); "
                       "MVS depths are not in the training-normalised frame")
    print(f"[mvs] frame check ok (view0 center diff {dc:.2e})")

    # --- reference views ----------------------------------------------------
    if args.views.strip():
        ref_views = [int(x) for x in args.views.split(",")]
    else:
        ref_views = list(np.linspace(0, V - 1, args.n_views, dtype=int))
    print(f"[ref views] {ref_views}")

    # ZNCC eval helper: per-point ZNCC of ref view va vs one alt vb (NaN padded)
    def ncc_col(x3d, nrm, va_i, vb_i, P, half_pix):
        B = x3d.shape[0]
        va = torch.full((B,), va_i, device=dev, dtype=torch.long)
        vb = torch.full((B,), vb_i, device=dev, dtype=torch.long)
        _, _, _, zf = pmvs_ncc_loss(
            images, x3d, nrm, va, vb, K_all, w2c_all, H, W,
            P, half_pix, smode, g_sigma, g_rad, ncc_min,
            return_full=True, ncc_color=ncc_clr)
        return zf                                          # (B,) NaN where unusable

    # per-(point,alt) gate replicating training mask (in_frame & not_occl & cos_ok [& fg])
    def alt_gate(x3d, nrm, vb_i):
        cam_b = c2w_all[vb_i, :3, 3]
        dirv = cam_b[None] - x3d
        dist = dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dp = dirv / dist; dist = dist.squeeze(-1)
        with torch.no_grad():
            if occ_md == "from_hit":
                _, tp, hitp = trace_nograd(f, x3d + 1e-2 * dp, dp, trace_cfg)
                not_occl = (~hitp) | (tp > dist - 0.1)
            else:
                _, tp, hitp = trace_nograd(
                    f, cam_b[None].expand_as(x3d).contiguous(), -dp, trace_cfg)
                not_occl = hitp & (dist <= tp + 0.1)
        cos_ok = (nrm * dp).sum(-1).abs() > cos_thr
        uv, z = project(x3d, K_all[vb_i], w2c_all[vb_i])
        inb = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        gate = inb & not_occl & cos_ok
        if masks is not None:
            uvc = uv.long().clamp(min=0)
            uvc[:, 0].clamp_(max=W - 1); uvc[:, 1].clamp_(max=H - 1)
            gate = gate & masks[vb_i, uvc[:, 1], uvc[:, 0]].bool()
        return gate, uv

    # accumulators -----------------------------------------------------------
    all_delta = []                                          # init reproj error, ALL valid rays (px)
    cov_maps = []                                           # (v, ngate HxW int, rgb) per ref view
    wf = dict(pixels=0, init_hit=0, hit_mvs=0, gated_ge1=0,
              gated_min=0)                                  # filter waterfall (pooled over views)
    # per patch: lists across rays/depths for the binned ZNCC-vs-|disp| curve
    curve_disp = {p: [] for p in patches}
    curve_zncc = {p: [] for p in patches}
    grad_ok    = {p: [] for p in patches}                  # bool per ray
    grad_mag   = {p: [] for p in patches}
    zncc_t0    = {p: [] for p in patches}                  # ZNCC at init surface
    zncc_tstar = {p: [] for p in patches}                  # ZNCC at reference surface
    n_rays_used = 0

    for v in ref_views:
        depth_m, conf_m, K_mvs, _ = load_mvs_view(mvs_root, v)
        Kt = K_all[v].cpu().numpy(); c2w = c2w_all[v].cpu().numpy()
        o_cam = c2w[:3, 3]

        # candidate fg pixels: init trace hits.  Build full-view rays (down grid).
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xf = (xs + .5) * down - .5; yf = (ys + .5) * down - .5
        d_cam = np.stack([(xf - Kt[0, 2]) / Kt[0, 0],
                          (yf - Kt[1, 2]) / Kt[1, 1], np.ones_like(xf)], -1)  # (H,W,3)
        d_w = d_cam @ c2w[:3, :3].T
        d_w_n = d_w / np.linalg.norm(d_w, axis=-1, keepdims=True)
        o_t = torch.from_numpy(np.broadcast_to(o_cam, d_w.shape).copy().reshape(-1, 3)).float().to(dev)
        d_t = torch.from_numpy(d_w_n.reshape(-1, 3)).float().to(dev)
        x_hit, t_hit, hit = trace_chunked(f, o_t, d_t, trace_cfg)
        hit_np = hit.cpu().numpy()

        # MVS depth at each training pixel: project cam dir into MVS image, sample.
        d_cam_f = d_cam.reshape(-1, 3)                      # (HW,3), z=1
        uv_mvs = np.stack([K_mvs[0, 0] * d_cam_f[:, 0] + K_mvs[0, 2],
                           K_mvs[1, 1] * d_cam_f[:, 1] + K_mvs[1, 2]], -1)  # (HW,2)
        z_mvs = sample_bilinear(depth_m, uv_mvs)            # cam-z depth, normalised units
        c_mvs = sample_bilinear(conf_m, uv_mvs)
        valid_mvs = np.isfinite(z_mvs) & (z_mvs > 1e-4) & (c_mvs > args.conf_thr)
        cand = np.where(hit_np & valid_mvs)[0]
        if cand.size == 0:
            print(f"  [view {v}] no valid rays"); continue
        # t* along the UNIT ray: t* = z_mvs * ||d_cam||  (d_cam has z=1)
        dnorm = np.linalg.norm(d_cam_f, axis=-1)
        t_star_all = z_mvs * dnorm
        alts = [int(a) for a in alt_nn[v].tolist()]

        # ---- supervision coverage over ALL valid candidate rays ---------------
        # For each candidate pixel: gate count at the INIT surface (how many alt
        # views supervise this ray) and the init reprojection error Δ (px). This
        # is the ICLR coverage map / filter waterfall / Δ-histogram source.
        cand_cov = cand if (args.cov_max <= 0 or cand.size <= args.cov_max) \
            else np.sort(rng.choice(cand, args.cov_max, replace=False))
        ngate_cand = np.empty(cand_cov.size, np.int32)
        Delta_cand = np.empty(cand_cov.size, np.float32)
        for s0 in range(0, cand_cov.size, 40000):
            ci = torch.from_numpy(cand_cov[s0:s0 + 40000]).long().to(dev)
            o_c = o_t[ci]; d_c = d_t[ci]
            x0_c = o_c + t_hit[ci][:, None] * d_c
            ts_c = torch.from_numpy(t_star_all[cand_cov[s0:s0 + 40000]]).float().to(dev)
            xs_c = o_c + ts_c[:, None] * d_c
            n0_c = grad_normals(f, x0_c)
            gcols, dcols = [], []
            for a in alts:
                g, uv0 = alt_gate(x0_c, n0_c, a)             # gate + π_a(x0)
                uvs, zs = project(xs_c, K_all[a], w2c_all[a])
                inb = (zs > 0) & (uvs[:, 0] >= 0) & (uvs[:, 0] < W) \
                      & (uvs[:, 1] >= 0) & (uvs[:, 1] < H)
                dpx = (uv0 - uvs).norm(dim=-1)
                dcols.append(torch.where(inb, dpx, torch.full_like(dpx, float("nan"))))
                gcols.append(g)
            ngate_cand[s0:s0 + 40000] = torch.stack(gcols, 0).sum(0).cpu().numpy()
            Delta_cand[s0:s0 + 40000] = torch.nanmedian(
                torch.stack(dcols, 1), dim=1).values.cpu().numpy()
        all_delta.append(Delta_cand)
        cov = np.full(H * W, -1, np.int32); cov[cand_cov] = ngate_cand
        cov_maps.append((v, cov.reshape(H, W), images[v].cpu().numpy()))
        # waterfall in full-ray units (scale the capped subset fractions to cand.size)
        wf["pixels"]    += H * W
        wf["init_hit"]  += int(hit_np.sum())
        wf["hit_mvs"]   += int(cand.size)
        wf["gated_ge1"] += int(round((ngate_cand >= 1).mean() * cand.size))
        wf["gated_min"] += int(round((ngate_cand >= args.min_gated).mean() * cand.size))

        # ---- expensive per-patch sweep: random subset of SUPERVISED rays ------
        sup = cand_cov[ngate_cand >= args.min_gated]
        if sup.size == 0:
            print(f"  [view {v}] no supervised rays"); continue
        sel = rng.choice(sup, size=min(args.n_rays, sup.size), replace=False)
        o_s = o_t[sel]; d_s = d_t[sel]
        t0 = t_hit[sel]                                     # init surface depth
        t_star = torch.from_numpy(t_star_all[sel]).float().to(dev)
        x_star = o_s + t_star[:, None] * d_s
        x0 = o_s + t0[:, None] * d_s
        R = sel.shape[0]

        # ---- depth sweep: points, normals, per-alt ZNCC + gate ----------------
        ts = torch.linspace(-args.band, args.band, args.n_depth, device=dev)   # rel to t*
        # absolute depths per ray: (R, T)
        t_abs = t_star[:, None] + ts[None, :]                                  # (R,T)
        x_sweep = (o_s[:, None, :] + t_abs[:, :, None] * d_s[:, None, :])       # (R,T,3)
        Bf = R * args.n_depth
        x_flat = x_sweep.reshape(Bf, 3)
        n_flat = grad_normals(f, x_flat)

        # gate + per-alt ZNCC for every patch, on the same swept points
        # gate is patch-independent; compute once per alt
        gate_cols = []; uv_alt_cols = []; uvstar_alt = []
        for a in alts:
            g, uv = alt_gate(x_flat, n_flat, a)
            gate_cols.append(g.reshape(R, args.n_depth))
            uv_alt_cols.append(uv.reshape(R, args.n_depth, 2))
            uvs, _ = project(x_star, K_all[a], w2c_all[a])                     # (R,2)
            uvstar_alt.append(uvs)
        gate_stack = torch.stack(gate_cols, 0)               # (A,R,T)
        uv_stack = torch.stack(uv_alt_cols, 0)               # (A,R,T,2)
        uvstar_stack = torch.stack(uvstar_alt, 0)            # (A,R,2)

        # induced reprojection displacement vs x* (px), median over in-frame alts
        disp = (uv_stack - uvstar_stack[:, :, None, :]).norm(dim=-1)           # (A,R,T)
        disp = torch.where(gate_stack, disp, torch.full_like(disp, float("nan")))
        disp_med = torch.nanmedian(disp, dim=0).values                        # (R,T)

        # index of t0 and the eps-bracket inside the sweep for the FD gradient.
        # (t0 is the init depth; build x at t0±eps directly rather than interpolate.)
        x_t0p = o_s + (t0[:, None] + args.eps) * d_s
        x_t0m = o_s + (t0[:, None] - args.eps) * d_s
        n_t0p = grad_normals(f, x_t0p); n_t0m = grad_normals(f, x_t0m)
        gate_t0p = []; gate_t0m = []
        for a in alts:
            gp, _ = alt_gate(x_t0p, n_t0p, a); gate_t0p.append(gp)
            gm, _ = alt_gate(x_t0m, n_t0m, a); gate_t0m.append(gm)
        gate_t0p = torch.stack(gate_t0p, 0); gate_t0m = torch.stack(gate_t0m, 0)  # (A,R)

        for (P, half_pix) in patches:
            # ZNCC along sweep: (A,R,T)
            z_alt = []
            for ai, a in enumerate(alts):
                zf = ncc_col(x_flat, n_flat, v, a, P, half_pix).reshape(R, args.n_depth)
                z_alt.append(zf)
            z_alt = torch.stack(z_alt, 0)                    # (A,R,T)
            z_g = torch.where(gate_stack & torch.isfinite(z_alt), z_alt,
                              torch.full_like(z_alt, float("nan")))
            zncc_mean = torch.nanmean(z_g, dim=0)            # (R,T) mean ZNCC over gated alts
            ngate = (gate_stack & torch.isfinite(z_alt)).sum(0)               # (R,T)
            L = 1.0 - zncc_mean                              # training NCC term

            # rays with enough support at t* (the centre sample)
            ci = args.n_depth // 2
            ok_ray = ngate[:, ci] >= args.min_gated
            if not ok_ray.any():
                continue

            # ---- basin curve points (|disp| vs ZNCC), gated rays ----
            dd = disp_med[ok_ray]; zz = zncc_mean[ok_ray]
            m = torch.isfinite(dd) & torch.isfinite(zz)
            curve_disp[(P, half_pix)].append(dd[m].detach().cpu().numpy())
            curve_zncc[(P, half_pix)].append(zz[m].detach().cpu().numpy())

            # ZNCC at t0 (init) and t* (reference)
            zncc_t0[(P, half_pix)].append(_ncc_at(
                f, o_s, d_s, t0, alts, v, P, half_pix, ncc_col, alt_gate,
                grad_normals, ok_ray, K_all, w2c_all))
            zncc_tstar[(P, half_pix)].append(zncc_mean[:, ci][ok_ray].detach().cpu().numpy())

            # ---- finite-difference gradient at t0 ----
            Lp = _L_at(f, x_t0p, n_t0p, gate_t0p, alts, v, P, half_pix, ncc_col)
            Lm = _L_at(f, x_t0m, n_t0m, gate_t0m, alts, v, P, half_pix, ncc_col)
            g = (Lp - Lm) / (2 * args.eps)                   # (R,) dL/dt at t0
            sign = (t0 - t_star) * g                         # >0 => descent toward t*
            valid_g = torch.isfinite(g) & ok_ray
            grad_ok[(P, half_pix)].append((sign[valid_g] > 0).detach().cpu().numpy())
            grad_mag[(P, half_pix)].append(g[valid_g].abs().detach().cpu().numpy())

        n_rays_used += int(ok_ray.sum())
        sup_pct = 100 * (ngate_cand >= args.min_gated).mean() if cand.size else 0.0
        print(f"  [view {v}] cand={cand.size} supervised={sup_pct:.1f}%  "
              f"sweep {R} -> used {int(ok_ray.sum())}  "
              f"median Δ={np.nanmedian(Delta_cand):.2f}px")

    # ----------------------------------------------------------------- report --
    all_delta = np.concatenate(all_delta) if all_delta else np.array([np.nan])
    med_delta = np.nanmedian(all_delta)
    print("\n================ SUMMARY ================")
    print(f"rays used: {n_rays_used}   median init reprojection error Δ = {med_delta:.2f} px "
          f"(p90={np.nanpercentile(all_delta,90):.2f})")
    print(f"{'patch':>10} {'half_pix':>8} {'grad_correct%':>14} {'|g|_med':>9} "
          f"{'ZNCC@t0':>8} {'ZNCC@t*':>8} {'basin_hw_px':>11}")
    summary = {}
    for (P, half_pix) in patches:
        ok = np.concatenate(grad_ok[(P, half_pix)]) if grad_ok[(P, half_pix)] else np.array([])
        gm = np.concatenate(grad_mag[(P, half_pix)]) if grad_mag[(P, half_pix)] else np.array([np.nan])
        z0 = np.concatenate(zncc_t0[(P, half_pix)]) if zncc_t0[(P, half_pix)] else np.array([np.nan])
        zs = np.concatenate(zncc_tstar[(P, half_pix)]) if zncc_tstar[(P, half_pix)] else np.array([np.nan])
        # basin half-width: |disp| at which binned-mean ZNCC drops to half its peak
        dd = np.concatenate(curve_disp[(P, half_pix)]) if curve_disp[(P, half_pix)] else np.array([])
        zz = np.concatenate(curve_zncc[(P, half_pix)]) if curve_zncc[(P, half_pix)] else np.array([])
        bx, bz, bhw = _binned_profile(dd, zz)
        frac = 100 * np.nanmean(ok) if ok.size else float("nan")
        print(f"{P:>10} {half_pix:>8.1f} {frac:>13.1f}% {np.nanmedian(gm):>9.3f} "
              f"{np.nanmean(z0):>8.3f} {np.nanmean(zs):>8.3f} {bhw:>11.2f}")
        summary[f"{P}_{half_pix}"] = dict(
            grad_correct_pct=float(frac), grad_mag_med=float(np.nanmedian(gm)),
            zncc_t0=float(np.nanmean(z0)), zncc_tstar=float(np.nanmean(zs)),
            basin_hw_px=float(bhw), bin_x=bx.tolist(), bin_z=bz.tolist())
    print("=========================================")
    print(f"Read: if Δ ({med_delta:.1f}px) >> a patch's basin half-width, that patch's\n"
          f"      ZNCC peak is too narrow to see the init error (flat/aliased gradient).\n"
          f"      If a LARGER patch washes ZNCC@t* down or broadens the basin without\n"
          f"      improving grad_correct%, the patch is too big (over-smoothed).")

    # ----------------------------------------------------------------- figures -
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(patches)))
    for (P, half_pix), c in zip(patches, colors):
        dd = np.concatenate(curve_disp[(P, half_pix)]) if curve_disp[(P, half_pix)] else np.array([])
        zz = np.concatenate(curve_zncc[(P, half_pix)]) if curve_zncc[(P, half_pix)] else np.array([])
        bx, bz, bhw = _binned_profile(dd, zz)
        ax.plot(bx, bz, "-o", color=c, ms=3, lw=1.8,
                label=f"P={P}, hp={half_pix} (basin±{bhw:.1f}px)")
    ax.axvline(med_delta, color="r", ls="--", lw=1.2, label=f"median init Δ={med_delta:.1f}px")
    ax.set_xlabel("|reprojection displacement from reference x*|  (pixels)")
    ax.set_ylabel("mean ZNCC over gated alt views")
    ax.set_title("ZNCC basin vs reprojection displacement")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    labels = [f"{P},{hp}" for P, hp in patches]
    fracs = [summary[f"{P}_{hp}"]["grad_correct_pct"] for P, hp in patches]
    bars = ax.bar(labels, fracs, color=colors)
    ax.axhline(50, color="gray", ls=":", lw=1, label="chance (50%)")
    for b, fr in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, fr + 1, f"{fr:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("gradient correct  ((t0−t*)·g > 0)  [%]")
    ax.set_xlabel("patch (P, half_pix)")
    ax.set_title(f"FD gradient correctness at init surface\n(rays={n_rays_used}, median Δ={med_delta:.1f}px)")
    ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

    fig.suptitle(f"Patch-size reprojection diagnostic — {scene.name}  init step {step}")
    fig.tight_layout()
    figp = out_dir / f"patch_size_reproj_{scene.name}_s{step}.png"
    fig.savefig(figp, dpi=120)

    # ---- ICLR fig 2: valid-supervision coverage maps (per ref view) ----------
    ncov = len(cov_maps)
    ncols = min(4, max(1, ncov)); nrows = int(np.ceil(ncov / ncols))
    figc, axc = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows),
                             squeeze=False)
    for i, (v, cmap_i, rgb) in enumerate(cov_maps):
        a = axc[i // ncols][i % ncols]
        a.imshow(rgb)
        masked = np.ma.masked_less(cmap_i, 0)                 # hide non-candidate px
        cm = plt.cm.turbo.copy(); cm.set_bad(alpha=0.0)
        im = a.imshow(masked, cmap=cm, vmin=0, vmax=n_alt, alpha=0.85)
        sup = (cmap_i >= args.min_gated).sum()
        nhit = int((cmap_i >= 0).sum())
        a.set_title(f"view {v}  supervised {100*sup/max(nhit,1):.0f}% of hits", fontsize=9)
        a.axis("off")
    for j in range(ncov, nrows * ncols):
        axc[j // ncols][j % ncols].axis("off")
    figc.colorbar(im, ax=axc.ravel().tolist(), fraction=0.02,
                  label=f"# gated alt views (≥{args.min_gated} = supervised)")
    figc.suptitle(f"Valid-supervision coverage — {scene.name} init  "
                  f"(gray=no candidate/init-miss/no-MVS)")
    figcp = out_dir / f"coverage_map_{scene.name}_s{step}.png"
    figc.savefig(figcp, dpi=120)

    # ---- ICLR fig 3: filter waterfall + Δ histogram --------------------------
    figw, axw = plt.subplots(1, 2, figsize=(15, 5.5))
    stages = [("all pixels", wf["pixels"]), ("init-hit", wf["init_hit"]),
              ("∩ valid MVS", wf["hit_mvs"]), ("∩ ≥1 gated alt", wf["gated_ge1"]),
              (f"∩ ≥{args.min_gated} gated alt\n(supervised)", wf["gated_min"])]
    names = [s[0] for s in stages]; vals = [s[1] for s in stages]
    base = max(vals[0], 1)
    ax = axw[0]
    bars = ax.barh(range(len(stages)), vals,
                   color=plt.cm.viridis(np.linspace(0.15, 0.85, len(stages))))
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for i, (b, val) in enumerate(zip(bars, vals)):
        ax.text(val, b.get_y() + b.get_height() / 2,
                f"  {val:,}  ({100*val/base:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("# rays"); ax.set_title("Filter waterfall (pooled over ref views)")
    ax.grid(alpha=.3, axis="x")

    ax = axw[1]
    dpos = all_delta[np.isfinite(all_delta)]
    if dpos.size:
        hi = np.nanpercentile(dpos, 99)
        ax.hist(np.clip(dpos, 0, hi), bins=60, color="steelblue", alpha=.85)
        ax.axvline(med_delta, color="r", ls="--", lw=1.4,
                   label=f"median={med_delta:.2f}px")
        ax.axvline(np.nanpercentile(dpos, 90), color="orange", ls=":", lw=1.4,
                   label=f"p90={np.nanpercentile(dpos,90):.2f}px")
        ax.legend(fontsize=9)
    ax.set_xlabel("init reprojection error Δ  (px)")
    ax.set_ylabel("# valid rays")
    ax.set_title(f"Δ histogram over all valid rays (n={dpos.size:,})")
    ax.grid(alpha=.3)
    figw.suptitle(f"Supervision coverage & init reprojection error — {scene.name} step {step}")
    figw.tight_layout()
    figwp = out_dir / f"coverage_waterfall_{scene.name}_s{step}.png"
    figw.savefig(figwp, dpi=120)

    np.savez(out_dir / f"patch_size_reproj_{scene.name}_s{step}.npz",
             delta=all_delta, summary=np.array(summary, dtype=object),
             patches=np.array(patches), med_delta=med_delta,
             waterfall=np.array(wf, dtype=object),
             cov_maps=np.array([c[1] for c in cov_maps], dtype=object),
             cov_views=np.array([c[0] for c in cov_maps]))
    print(f"\n[done] -> {figp}\n        {figcp}\n        {figwp}")


# --- small per-call evaluators (kept out of the hot loop for readability) -----
def _L_at(f, x, n, gate, alts, va, P, half_pix, ncc_col):
    """training NCC term L = mean_gated (1-ZNCC) at fixed points x (R,3)."""
    cols = []
    for ai, a in enumerate(alts):
        zf = ncc_col(x, n, va, a, P, half_pix)
        zf = torch.where(gate[ai] & torch.isfinite(zf), zf, torch.full_like(zf, float("nan")))
        cols.append(zf)
    z = torch.stack(cols, 0)                                # (A,R)
    return 1.0 - torch.nanmean(z, dim=0)                    # (R,)


def _ncc_at(f, o_s, d_s, t, alts, va, P, half_pix, ncc_col, alt_gate,
            grad_normals, ok_ray, K_all, w2c_all):
    """mean gated ZNCC at depth t (R,) -> np array over ok_ray rays."""
    x = o_s + t[:, None] * d_s
    n = grad_normals(f, x)
    cols = []
    for a in alts:
        g, _ = alt_gate(x, n, a)
        zf = ncc_col(x, n, va, a, P, half_pix)
        zf = torch.where(g & torch.isfinite(zf), zf, torch.full_like(zf, float("nan")))
        cols.append(zf)
    z = torch.nanmean(torch.stack(cols, 0), dim=0)
    return z[ok_ray].detach().cpu().numpy()


def _binned_profile(dd, zz, edges=None):
    """Binned-mean ZNCC vs |disp| (px) + basin half-width (|disp| where mean ZNCC
    falls to peak - 0.5*(peak-min)). Returns (bin_centers, bin_means, half_width)."""
    if dd.size == 0:
        return np.array([0.]), np.array([np.nan]), float("nan")
    if edges is None:
        hi = np.nanpercentile(dd, 98)
        edges = np.linspace(0, max(hi, 1e-3), 21)
    idx = np.clip(np.digitize(dd, edges) - 1, 0, len(edges) - 2)
    cx = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(len(cx), np.nan)
    for b in range(len(cx)):
        m = idx == b
        if m.any():
            means[b] = np.nanmean(zz[m])
    valid = np.isfinite(means)
    if valid.sum() < 2:
        return cx, means, float("nan")
    peak = np.nanmax(means); lo = np.nanmin(means)
    half = peak - 0.5 * (peak - lo)
    hw = float("nan")
    below = np.where(valid & (means < half))[0]
    if below.size:
        hw = cx[below[0]]
    return cx, means, hw


if __name__ == "__main__":
    main()
