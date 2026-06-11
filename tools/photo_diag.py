"""Per-image photoconsistency diagnostic.

Loads a checkpoint, sphere-traces one reference view, and writes:
  gate_hit.png       — pixels that hit the surface
  gate_in_frame.png  — fraction of alt views that see the hit point
  gate_not_occl.png  — fraction of alt views not occluded (given in_frame)
  gate_cos_ok.png    — fraction of alt views with |n·d| > cos_thresh
  gate_mask.png      — fraction of alt views passing ALL gates
  l1.png             — mean L1 color error on active (mask-passing) alt views
  cost_volume.png    — photometric cost C(t) vs depth for sampled hit rays

Usage:
  python photo_diag.py --ckpt outputs/run_xxx/checkpoint_best_photo.pt --view 0 --down 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lip_tracer.config import TraceConfig
from lip_tracer.data import load_views, load_blender_views, precompute_alt_cameras
from lip_tracer.loss import bilinear_sample
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd


# ---------- model loading ----------

def _load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    architecture = ckpt.get("architecture", "cpl")
    if architecture == "neus":
        hidden = ckpt["f"]["layers.0.weight"].shape[0]
    elif "head_weight" in ckpt["f"]:
        hidden = ckpt["f"]["head_weight"].shape[0]
    else:
        hidden = next(
            v.shape[1] for k, v in ckpt["f"].items()
            if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
            and not k.startswith("encoder")
        )
    group_size     = ckpt.get("group_size", 2)
    activation     = ckpt.get("activation", "groupsort")
    input_encoding = ckpt.get("input_encoding", "pe")
    multires       = ckpt.get("multires", 6)
    if architecture == "neus":
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("layers.") and k.endswith(".weight")))
    else:
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("net.") and k.endswith(".weight")
                                      and "_u" not in k))
    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_encoding,
                   multires=multires, architecture=architecture).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    print(f"  arch={architecture}  hidden={hidden}  depth={depth}  step={ckpt.get('step', '?')}")
    return f


# ---------- ray building ----------

def _make_view_rays(views: dict, v: int, down: int = 1):
    """Build (N,3) origin + direction arrays for one view, downsampled by `down`."""
    K   = views["K"][v].numpy()
    c2w = views["c2w"][v].numpy()
    H   = views["H"] // down
    W   = views["W"] // down
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs_f = (xs + 0.5) * down - 0.5
    ys_f = (ys + 0.5) * down - 0.5
    d_cam = np.stack(
        [(xs_f - K[0, 2]) / K[0, 0],
         (ys_f - K[1, 2]) / K[1, 1],
         np.ones_like(xs_f)],
        axis=-1,
    )
    dirs = d_cam @ c2w[:3, :3].T
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    o_np = np.broadcast_to(c2w[:3, 3], dirs.shape)
    return o_np.reshape(-1, 3).copy(), dirs.reshape(-1, 3), H, W


# ---------- sphere tracing ----------

def _trace_chunked(f, o: torch.Tensor, d: torch.Tensor,
                   t_far: float, iters: int, eps: float,
                   device: str, chunk: int):
    N = o.shape[0]
    x_all   = torch.empty_like(o)
    t_all   = torch.zeros(N)
    hit_all = torch.zeros(N, dtype=torch.bool)
    cfg = TraceConfig(iters=iters, eps=eps, t_far=t_far)
    t0 = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        with torch.no_grad():
            xb, tb, hb = trace_nograd(f, o[s:e].to(device), d[s:e].to(device), cfg)
        x_all[s:e]   = xb.cpu()
        t_all[s:e]   = tb.cpu()
        hit_all[s:e] = hb.cpu()
        if ((s // chunk) % 4 == 0) or (e == N):
            elapsed = time.time() - t0
            print(f"  trace {e:,}/{N:,}  hit={hit_all[:e].float().mean():.1%}  {elapsed:.0f}s")
    return x_all, t_all, hit_all


# ---------- normals ----------

def _compute_normals(f, x_theta: torch.Tensor, hit: torch.Tensor,
                     device: str, chunk: int = 8192) -> torch.Tensor:
    normals = torch.zeros_like(x_theta)
    idx = hit.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return normals
    x_h   = x_theta[idx]
    n_out = torch.empty_like(x_h)
    eps   = 1e-3
    dx    = torch.tensor([[eps, 0, 0], [0, eps, 0], [0, 0, eps]],
                         dtype=torch.float32, device=device)
    for s in range(0, x_h.shape[0], chunk):
        e  = min(s + chunk, x_h.shape[0])
        xb = x_h[s:e].to(device)
        with torch.no_grad():
            g = torch.stack(
                [f(xb + dx[i]) - f(xb - dx[i]) for i in range(3)], dim=-1
            ) / (2 * eps)
        n_out[s:e] = F.normalize(g, dim=-1).cpu()
    normals[idx] = n_out
    return normals


# ---------- gate maps + L1 ----------

def _compute_gates(
    f,
    x_theta: torch.Tensor,   # (N, 3) CPU
    hit: torch.Tensor,        # (N,) bool CPU
    normals: torch.Tensor,    # (N, 3) CPU
    v_ref: int,
    views: dict,
    n_alt: int,
    cos_thresh: float,
    device: str,
    chunk: int,
) -> dict:
    """Accumulate per-alt-view gate stats and L1 error back to (N,) tensors."""
    V   = views["c2w"].shape[0]
    H   = views["H"];  W = views["W"]
    alt_nn  = precompute_alt_cameras(views, n_alt)
    alt_ids = alt_nn[v_ref]                               # (n_alt,)

    images  = views["images"].to(device)                  # (V, H, W, 3) — stays on GPU
    K_all   = views["K"].to(device)
    w2c_all = torch.linalg.inv(views["c2w"]).to(device)
    masks   = views["masks"].to(device) if "masks" in views else None

    N = x_theta.shape[0]
    in_frame_sum = torch.zeros(N)
    not_occl_sum = torch.zeros(N)
    cos_ok_sum   = torch.zeros(N)
    mask_sum     = torch.zeros(N)
    l1_sum       = torch.zeros(N)
    l1_cnt       = torch.zeros(N)

    # Reference view pixel colour at each hit point
    c_src = _sample_at_x(images, v_ref, x_theta, hit, K_all[v_ref], w2c_all[v_ref], H, W, device, chunk)

    cfg = TraceConfig()

    for ki in range(n_alt):
        ak      = int(alt_ids[ki])
        R_ak    = w2c_all[ak, :3, :3]
        t_ak    = w2c_all[ak, :3, 3]
        op_ak   = views["c2w"][ak, :3, 3].to(device)   # camera origin

        in_frame_k = torch.zeros(N, dtype=torch.bool)
        not_occl_k = torch.zeros(N, dtype=torch.bool)
        cos_ok_k   = torch.zeros(N, dtype=torch.bool)
        mask_k     = torch.zeros(N, dtype=torch.bool)
        l1_k       = torch.zeros(N)

        for s in range(0, N, chunk):
            e    = min(s + chunk, N)
            x_b  = x_theta[s:e].to(device)
            n_b  = normals[s:e].to(device)
            h_b  = hit[s:e].to(device)

            # Project into alt camera
            xc  = x_b @ R_ak.T + t_ak
            uvh = xc @ K_all[ak].T
            uv  = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)
            in_b = (h_b & (xc[:, 2] > 0)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < H))

            # Occlusion test (sphere-trace from alt camera to x_b)
            dir_k  = x_b - op_ak
            dist_k = dir_k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dp_k   = dir_k / dist_k
            dist_k = dist_k.squeeze(-1)
            with torch.no_grad():
                _, tp_k, hitp_k = trace_nograd(f, op_ak.expand(e - s, 3), dp_k, cfg)
            not_occ = hitp_k & (dist_k <= tp_k + 0.1)

            # Cosine gate
            cos_ok_b = (n_b * dp_k).sum(-1).abs() > cos_thresh

            # Foreground gate
            fg_gate = torch.ones(e - s, dtype=torch.bool, device=device)
            if masks is not None:
                uv_i = uv.long()
                uv_i[:, 0].clamp_(0, W - 1); uv_i[:, 1].clamp_(0, H - 1)
                fg_gate = masks[ak, uv_i[:, 1], uv_i[:, 0]]

            mask_b = in_b & not_occ & cos_ok_b & fg_gate

            in_frame_k[s:e] = in_b.cpu()
            not_occl_k[s:e] = (in_b & not_occ).cpu()
            cos_ok_k[s:e]   = (in_b & not_occ & cos_ok_b).cpu()
            mask_k[s:e]     = mask_b.cpu()

            # L1 error for active rays
            if mask_b.any():
                uv_m = uv[mask_b].clamp(
                    uv.new_tensor([0, 0]), uv.new_tensor([W - 1, H - 1])
                )
                vi_t   = torch.full((mask_b.sum(),), ak, dtype=torch.long, device=device)
                c_alt  = bilinear_sample(images, vi_t, uv_m, H, W)
                l1_b   = (c_src[s:e][mask_b.cpu()].to(device) - c_alt).abs().sum(-1)
                l1_k[s:e][mask_b.cpu()] = l1_b.cpu()

        in_frame_sum += in_frame_k.float()
        not_occl_sum += not_occl_k.float()
        cos_ok_sum   += cos_ok_k.float()
        mask_sum     += mask_k.float()
        l1_sum       += l1_k
        l1_cnt       += mask_k.float()

        hit_frac = hit.float().mean().item()
        pct = lambda t: f"{t.float().mean().item() / max(hit_frac, 1e-6):.1%}"
        print(f"  alt {ki}/{n_alt} (cam {ak}): "
              f"in_frame={pct(in_frame_k)}  "
              f"not_occl={pct(not_occl_k)}  "
              f"cos_ok={pct(cos_ok_k)}  "
              f"mask={pct(mask_k)}")

    l1_mean = torch.where(l1_cnt > 0, l1_sum / l1_cnt, torch.zeros_like(l1_sum))
    return {
        "hit":       hit.float(),
        "in_frame":  in_frame_sum / n_alt,
        "not_occl":  not_occl_sum / n_alt,
        "cos_ok":    cos_ok_sum   / n_alt,
        "mask":      mask_sum     / n_alt,
        "l1":        l1_mean,
        "l1_valid":  (l1_cnt > 0),
    }


def _sample_at_x(images, v: int, x_theta, hit, K, w2c, H, W, device, chunk) -> torch.Tensor:
    """Sample image v at the reprojection of each hit point."""
    N     = x_theta.shape[0]
    c_out = torch.zeros(N, 3)
    R, t  = w2c[:3, :3], w2c[:3, 3]
    for s in range(0, N, chunk):
        e   = min(s + chunk, N)
        xb  = x_theta[s:e].to(device)
        xc  = xb @ R.T + t
        uvh = xc @ K.T
        uv  = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)
        uv  = uv.clamp(uv.new_tensor([0, 0]), uv.new_tensor([W - 1, H - 1]))
        vi_t = torch.full((e - s,), v, dtype=torch.long, device=device)
        c_out[s:e] = bilinear_sample(images, vi_t, uv, H, W).cpu()
    return c_out


# ---------- cost volume ----------

def _compute_cost_volume(
    f,
    o_rays: torch.Tensor,     # (n_show, 3) selected ray origins
    d_rays: torch.Tensor,     # (n_show, 3) ray directions
    vi_rays: torch.Tensor,    # (n_show,) view index (all = v_ref)
    views: dict,
    alt_nn: torch.Tensor,     # (V, n_alt)
    t_near: float,
    t_far: float,
    L: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return t_cands (L,) and costs (n_show, L)."""
    n_show = o_rays.shape[0]
    n_alt  = alt_nn.shape[1]
    H, W   = views["H"], views["W"]
    images  = views["images"].to(device)
    K_all   = views["K"].to(device)
    w2c_all = torch.linalg.inv(views["c2w"]).to(device)

    t_cands = torch.linspace(t_near, t_far, L)   # (L,)
    costs   = torch.zeros(n_show, L)
    vis_cnt = torch.zeros(n_show, L)

    cfg = TraceConfig()

    for ri in range(n_show):
        vi_r = int(vi_rays[ri])
        o_r  = o_rays[ri].to(device)
        d_r  = d_rays[ri].to(device)
        # Candidate 3D points along the ray
        X = o_r + t_cands[:, None].to(device) * d_r   # (L, 3)

        for ki in range(n_alt):
            ak   = int(alt_nn[vi_r, ki])
            R_ak = w2c_all[ak, :3, :3]
            t_ak = w2c_all[ak, :3, 3]
            op   = views["c2w"][ak, :3, 3].to(device)

            xc  = X @ R_ak.T + t_ak
            uvh = xc @ K_all[ak].T
            uv  = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)
            in_f = ((xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < H))

            # Occlusion for all L candidates at once
            dir_k  = X - op
            dist_k = dir_k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dp_k   = dir_k / dist_k
            dist_k = dist_k.squeeze(-1)
            with torch.no_grad():
                _, tp_k, hitp_k = trace_nograd(f, op.expand(L, 3), dp_k, cfg)
            vis = (in_f & hitp_k & (dist_k <= tp_k + 0.1)).float()

            if vis.sum() == 0:
                continue

            # Colour in alt view
            uv_c  = uv.clamp(uv.new_tensor([0, 0]), uv.new_tensor([W - 1, H - 1]))
            vi_a  = torch.full((L,), ak, dtype=torch.long, device=device)
            c_alt = bilinear_sample(images, vi_a, uv_c, H, W)

            # Colour at candidate position in reference view
            R_r, t_r = w2c_all[vi_r, :3, :3], w2c_all[vi_r, :3, 3]
            xc_r  = X @ R_r.T + t_r
            uvh_r = xc_r @ K_all[vi_r].T
            uv_r  = uvh_r[:, :2] / uvh_r[:, 2:3].clamp(min=1e-6)
            uv_rc = uv_r.clamp(uv_r.new_tensor([0, 0]), uv_r.new_tensor([W - 1, H - 1]))
            vi_r_t = torch.full((L,), vi_r, dtype=torch.long, device=device)
            c_src = bilinear_sample(images, vi_r_t, uv_rc, H, W)

            cost_l = (c_src - c_alt).pow(2).sum(-1)   # (L,)
            costs[ri]   += (vis * cost_l).cpu()
            vis_cnt[ri] += vis.cpu()

    costs = torch.where(vis_cnt > 0, costs / vis_cnt, torch.full_like(costs, float("nan")))
    return t_cands, costs


# ---------- visualisation ----------

def _save_gate_map(arr: torch.Tensor, valid: torch.Tensor | None, path: Path,
                   H: int, W: int, title: str, cmap: str,
                   vmin: float | None = None, vmax: float | None = None) -> None:
    img = arr.reshape(H, W).numpy().astype(np.float32)
    if valid is not None:
        img = np.where(valid.reshape(H, W).numpy(), img, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_cost_strip(t_cands: torch.Tensor, costs: torch.Tensor,
                     t_preds: np.ndarray, path: Path) -> None:
    n = costs.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=False)
    if n == 1:
        axes = [axes]
    t = t_cands.numpy()
    for i, ax in enumerate(axes):
        c = costs[i].numpy()
        ax.plot(t, c, "b-", lw=1.5, label="cost")
        ax.axvline(t_preds[i], color="r", lw=1.5, linestyle="--", label="t_pred")
        # Mark soft-argmin expected depth
        finite = np.isfinite(c)
        if finite.sum() > 1:
            w = np.exp(-c[finite] / (np.nanstd(c[finite]) + 1e-8))
            w /= w.sum()
            t_sa = float((t[finite] * w).sum())
            ax.axvline(t_sa, color="g", lw=1.0, linestyle=":", label="t_sa")
        ax.set_xlabel("depth t", fontsize=8)
        ax.set_title(f"ray {i}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=7)
    fig.suptitle("Photometric cost C(t) vs depth  (r=t_pred  g=soft-argmin)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",       type=Path, required=True,
                    help="checkpoint .pt file")
    ap.add_argument("--scene",      type=Path, default=None,
                    help="scene directory (inferred from config.json if omitted)")
    ap.add_argument("--dataset",    choices=["dtu", "lego"], default=None)
    ap.add_argument("--view",       type=int, default=0,
                    help="reference view index to diagnose")
    ap.add_argument("--n_alt",      type=int, default=6,
                    help="number of nearest-neighbour alt cameras")
    ap.add_argument("--cos_thresh", type=float, default=0.1)
    ap.add_argument("--t_far",      type=float, default=10.0)
    ap.add_argument("--t_near",     type=float, default=0.1,
                    help="near bound for cost-volume candidates")
    ap.add_argument("--iters",      type=int, default=24)
    ap.add_argument("--eps",        type=float, default=1e-3)
    ap.add_argument("--down",       type=int, default=4,
                    help="pixel-grid downsample factor (4 → quarter resolution)")
    ap.add_argument("--chunk",      type=int, default=32768)
    ap.add_argument("--n_cost_rays", type=int, default=8,
                    help="number of rays in the cost-volume strip")
    ap.add_argument("--cost_L",     type=int, default=64,
                    help="number of depth candidates per ray in cost volume")
    ap.add_argument("--out_dir",    type=Path,
                    default=Path("outputs/photo_diag"))
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Infer scene / dataset from config.json in checkpoint directory
    config_json = args.ckpt.parent / "config.json"
    if config_json.exists() and args.scene is None:
        cfg_dict = json.loads(config_json.read_text())
        args.scene = Path(cfg_dict["scene"])
        print(f"[diag] scene from config.json: {args.scene}")
    if args.scene is None:
        raise ValueError("--scene is required when config.json is absent next to the checkpoint")
    if args.dataset is None:
        args.dataset = "lego" if "lego" in str(args.scene).lower() else "dtu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] ckpt={args.ckpt}  view={args.view}  down={args.down}  device={args.device}")

    # Load model
    f = _load_model(args.ckpt, args.device)

    # Load views
    views = (load_blender_views(args.scene, down=1)
             if args.dataset == "lego" else load_views(args.scene))
    V = views["c2w"].shape[0]
    print(f"[diag] V={V}  H={views['H']}  W={views['W']}")

    # Build rays for the reference view at reduced resolution
    o_np, d_np, H, W = _make_view_rays(views, args.view, down=args.down)
    o = torch.from_numpy(o_np).float()
    d = torch.from_numpy(d_np).float()
    N = o.shape[0]
    print(f"[diag] tracing {N:,} rays ({H}×{W}) for view {args.view}")

    # Sphere trace
    x_theta, t_pred, hit = _trace_chunked(
        f, o, d, args.t_far, args.iters, args.eps, args.device, args.chunk
    )
    print(f"[diag] hit rate: {hit.float().mean():.1%}")

    # Surface normals at hit points
    print("[diag] computing normals …")
    normals = _compute_normals(f, x_theta, hit, args.device)

    # Gate maps + L1
    print("[diag] computing gate maps …")
    g = _compute_gates(f, x_theta, hit, normals, args.view, views,
                       args.n_alt, args.cos_thresh, args.device, args.chunk)

    # Save image maps
    hit_mask = g["hit"] > 0
    maps = [
        ("gate_hit",      g["hit"],      None,     "viridis",
         f"Hit (view {args.view}, down={args.down})", 0.0, 1.0),
        ("gate_in_frame", g["in_frame"], hit_mask, "plasma",
         "In-frame fraction over alt views  [gate 1]", 0.0, 1.0),
        ("gate_not_occl", g["not_occl"], hit_mask, "plasma",
         "Not-occluded fraction over alt views  [gate 2]", 0.0, 1.0),
        ("gate_cos_ok",   g["cos_ok"],   hit_mask, "plasma",
         "cos_ok fraction over alt views  [gate 3]", 0.0, 1.0),
        ("gate_mask",     g["mask"],     hit_mask, "Blues",
         "ALL-gates pass fraction (photo loss active)", 0.0, 1.0),
        ("l1",            g["l1"],       g["l1_valid"], "hot",
         "Mean L1 color error  (active rays)", None, None),
    ]
    for name, arr, valid, cmap, title, vmin, vmax in maps:
        p = args.out_dir / f"{name}.png"
        _save_gate_map(arr, valid, p, H, W, title, cmap, vmin=vmin, vmax=vmax)
        print(f"  wrote {p}")

    # Cost volume for selected hit rays
    if args.n_cost_rays > 0 and hit.any():
        print(f"[diag] computing cost volume for {args.n_cost_rays} rays …")
        hit_idx = hit.nonzero(as_tuple=True)[0]
        # Select rays spread uniformly over hit pixels (covers edge + centre)
        sel = torch.linspace(0, hit_idx.numel() - 1, args.n_cost_rays).long()
        sel_idx = hit_idx[sel]

        alt_nn  = precompute_alt_cameras(views, args.n_alt)
        vi_sel  = torch.full((args.n_cost_rays,), args.view, dtype=torch.long)
        t_cands, cost_vol = _compute_cost_volume(
            f, o[sel_idx], d[sel_idx], vi_sel,
            views, alt_nn, args.t_near, args.t_far, args.cost_L, args.device,
        )
        t_preds_sel = t_pred[sel_idx].numpy()
        p = args.out_dir / "cost_volume.png"
        _save_cost_strip(t_cands, cost_vol, t_preds_sel, p)
        print(f"  wrote {p}")

    print(f"[diag] done → {args.out_dir}/")


if __name__ == "__main__":
    main()
