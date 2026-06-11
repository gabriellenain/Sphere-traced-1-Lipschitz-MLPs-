"""Per-pixel ZNCC heatmap projected from 3D surface points back into each view.

For each reference view:
  1. Sphere-trace rays → hit points on the surface
  2. For each hit, project into n_alt nearest views, compute ZNCC patch
  3. Paint the mean ZNCC at the reference pixel → heatmap
  4. Overlay with fg mask boundary to correlate with hole locations

Usage:
    python viz_zncc_surface.py \
        --pt outputs/run_20260515_102909_scan65/checkpoint_latest.pt \
        --views 16 0 24 32 --out outputs/zncc_viz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.data import load_views, precompute_alt_cameras
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd


# ── NCC helpers ──────────────────────────────────────────────────────────────

def extract_patch(images: torch.Tensor, vi: torch.Tensor,
                  uv: torch.Tensor, H: int, W: int, patch: int) -> torch.Tensor:
    """Bilinear patch extraction. Returns (B, 3*P*P)."""
    P = patch; half = (P - 1) / 2.0
    oy = torch.linspace(-half, half, P, device=images.device)
    ox = torch.linspace(-half, half, P, device=images.device)
    px = (uv[:, 0, None, None] + ox[None, None, :]).clamp(0, W - 1)  # (B,P,P)
    py = (uv[:, 1, None, None] + oy[None, :, None]).clamp(0, H - 1)
    x0 = px.long(); x1 = (x0 + 1).clamp(max=W - 1)
    y0 = py.long(); y1 = (y0 + 1).clamp(max=H - 1)
    wx = (px - x0.float()).unsqueeze(-1)
    wy = (py - y0.float()).unsqueeze(-1)
    vi_e = vi[:, None, None]
    vals = (images[vi_e, y0, x0].float() * (1 - wx) * (1 - wy)
          + images[vi_e, y0, x1].float() * wx       * (1 - wy)
          + images[vi_e, y1, x0].float() * (1 - wx) * wy
          + images[vi_e, y1, x1].float() * wx       * wy)   # (B,P,P,3)
    return vals.reshape(vi.shape[0], -1)                     # (B, 3*P*P)


def batch_zncc(pa: torch.Tensor, pb: torch.Tensor,
               std_thr: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """ZNCC between two sets of patches. Returns (zncc, valid) both (B,)."""
    pa = pa - pa.mean(dim=1, keepdim=True)
    pb = pb - pb.mean(dim=1, keepdim=True)
    na = pa.norm(dim=1); nb = pb.norm(dim=1)
    valid = (na > std_thr) & (nb > std_thr)
    zncc = torch.zeros(pa.shape[0], device=pa.device)
    if valid.any():
        zncc[valid] = (pa[valid] * pb[valid]).sum(dim=1) / (
            na[valid] * nb[valid]).clamp(min=1e-6)
    return zncc.clamp(-1.0, 1.0), valid


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt",      required=True)
    ap.add_argument("--scene",   default=None)
    ap.add_argument("--views",   type=int, nargs="+", default=[16, 0, 24, 32])
    ap.add_argument("--n-alt",   type=int, default=6)
    ap.add_argument("--patch",   type=int, default=5)
    ap.add_argument("--down",    type=int, default=2,  help="render downscale factor")
    ap.add_argument("--batch",   type=int, default=4096)
    ap.add_argument("--out",     default="outputs/zncc_viz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt = torch.load(args.pt, map_location="cpu")
    # Try to load config from the run directory's config.json (not stored inline in old checkpoints)
    cfg: dict = ckpt.get("config", {})
    if not cfg:
        cfg_path = Path(args.pt).parent / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            print(f"[info] loaded config from {cfg_path}")
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    hidden     = mcfg.get("hidden", 256)
    depth      = mcfg.get("depth", 8)
    group_size = mcfg.get("group_size", 2)
    activation = mcfg.get("activation", "groupsort")
    input_enc  = mcfg.get("input_encoding", "pe")
    multires   = mcfg.get("multires", 6)
    arch       = mcfg.get("architecture", "cpl")

    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_enc,
                   multires=multires, architecture=arch).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    f.eval()
    for p in f.parameters(): p.requires_grad_(False)

    # TraceConfig from checkpoint (override defaults with trained values)
    from lip_tracer.config import TraceConfig
    tcfg_stored = cfg.get("trace", {}) if isinstance(cfg, dict) else {}
    trace_cfg = TraceConfig(
        iters        = tcfg_stored.get("iters",         36),
        eps          = tcfg_stored.get("eps",           1e-3),
        t_far        = tcfg_stored.get("t_far",         5.0),
        newton_steps = tcfg_stored.get("newton_steps",  2),
        bsphere_radius = tcfg_stored.get("bsphere_radius", 0.0),
        sdf_min_beta = tcfg_stored.get("sdf_min_beta",  0.0),
    )
    print(f"loaded: arch={arch} hidden={hidden} depth={depth}  "
          f"t_far={trace_cfg.t_far} iters={trace_cfg.iters}")

    # ── Scene ────────────────────────────────────────────────────────────────
    if args.scene is None:
        cfg_path = Path(args.pt).parent / "config.json"
        args.scene = json.loads(cfg_path.read_text())["scene"]
    views   = load_views(Path(args.scene), down=args.down)
    images  = views["images"].to(device)   # (V, H, W, 3)  uint8 or float
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    K_all   = views["K"].to(device)
    c2w_all = views["c2w"].to(device)
    w2c_all = torch.linalg.inv(c2w_all)
    masks   = views["masks"].to(device) if "masks" in views else None
    H, W    = images.shape[1], images.shape[2]   # actual loaded resolution
    V       = images.shape[0]
    origins = c2w_all[:, :3, 3]
    alt_nn  = precompute_alt_cameras(views, args.n_alt).to(device)  # (V, n_alt)

    print(f"scene: {V} views  H={H} W={W} (down={args.down})")

    for vi_ref in args.views:
        print(f"\nprocessing view {vi_ref} …")

        # ── Build pixel grid ─────────────────────────────────────────────────
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        px_flat = xs.reshape(-1); py_flat = ys.reshape(-1)
        N = px_flat.shape[0]

        # Ray directions
        d_cam = torch.stack([
            (px_flat - K_all[vi_ref, 0, 2]) / K_all[vi_ref, 0, 0],
            (py_flat - K_all[vi_ref, 1, 2]) / K_all[vi_ref, 1, 1],
            torch.ones(N, device=device),
        ], dim=-1)
        R_cw = c2w_all[vi_ref, :3, :3]          # cam→world
        d_world = F.normalize(d_cam @ R_cw.T, dim=-1)
        o = origins[vi_ref].unsqueeze(0).expand(N, 3)

        # ── Sphere trace in batches ──────────────────────────────────────────
        zncc_map  = torch.full((N,), float("nan"), device=device)
        valid_map = torch.zeros(N, dtype=torch.bool, device=device)
        hit_map   = torch.zeros(N, dtype=torch.bool, device=device)

        # Sanity: SDF at camera origin + project scene center into view
        with torch.no_grad():
            sdf_origin = f(origins[vi_ref:vi_ref+1]).item()
        o_ref = origins[vi_ref]
        print(f"  origin={o_ref.cpu().numpy().round(3)}  f(origin)={sdf_origin:.3f}  t_far={trace_cfg.t_far}")

        # Project scene center (0,0,0) into this view
        center = torch.zeros(1, 3, device=device)
        xc_c = (w2c_all[vi_ref, :3, :3] @ center.T).T + w2c_all[vi_ref, :3, 3]
        uvh_c = (K_all[vi_ref] @ xc_c.T).T
        uv_c = uvh_c[0, :2] / uvh_c[0, 2].clamp(min=1e-6)
        print(f"  scene center projects to px=({uv_c[0].item():.1f}, {uv_c[1].item():.1f})  image={W}x{H}")

        # Shoot a ray directly from origin toward scene center — should hit if model+cam consistent
        dir_to_center = F.normalize(-o_ref.unsqueeze(0), dim=-1)
        with torch.no_grad():
            _, t_test, hit_test = trace_nograd(f, o_ref.unsqueeze(0), dir_to_center, trace_cfg)
        print(f"  ray→center: hit={hit_test[0].item()}  t={t_test[0].item():.3f}")

        for start in range(0, N, args.batch):
            end = min(start + args.batch, N)
            o_b = o[start:end]; d_b = d_world[start:end]

            with torch.no_grad():
                x_surf, t_vals, hit = trace_nograd(f, o_b, d_b, trace_cfg)
            if start == 0:
                print(f"  batch0: hit={hit.sum()}/{end-start}  t min/mean/max={t_vals.min():.2f}/{t_vals.mean():.2f}/{t_vals.max():.2f}")

            hit_map[start:end] = hit
            if not hit.any():
                continue

            x_h = x_surf[hit]   # (Nh, 3)
            Nh  = x_h.shape[0]
            vi_h = torch.full((Nh,), vi_ref, device=device, dtype=torch.long)

            # Project into reference view (should be ~original pixel, sanity check)
            xc_ref = (w2c_all[vi_ref, :3, :3] @ x_h.T).T + w2c_all[vi_ref, :3, 3]
            uvh    = (K_all[vi_ref] @ xc_ref.T).T
            uv_ref_pts = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)

            pa = extract_patch(images, vi_h, uv_ref_pts, H, W, args.patch)  # (Nh, 3P²)

            zncc_per_alt = []
            valid_per_alt = []

            for k in range(args.n_alt):
                ak = alt_nn[vi_ref, k].item()
                vi_alt = torch.full((Nh,), ak, device=device, dtype=torch.long)

                # Project into alt view
                xc_alt = (w2c_all[ak, :3, :3] @ x_h.T).T + w2c_all[ak, :3, 3]
                uvh_alt = (K_all[ak] @ xc_alt.T).T
                uv_alt  = uvh_alt[:, :2] / uvh_alt[:, 2:3].clamp(min=1e-6)

                in_frame = ((xc_alt[:, 2] > 0)
                            & (uv_alt[:, 0] >= 0) & (uv_alt[:, 0] < W)
                            & (uv_alt[:, 1] >= 0) & (uv_alt[:, 1] < H))

                uv_alt_clamped = torch.stack([
                    uv_alt[:, 0].clamp(0, W - 1),
                    uv_alt[:, 1].clamp(0, H - 1),
                ], dim=-1)
                pb = extract_patch(images, vi_alt, uv_alt_clamped, H, W, args.patch)

                zncc_k, val_k = batch_zncc(pa, pb)
                val_k = val_k & in_frame
                zncc_per_alt.append(zncc_k)
                valid_per_alt.append(val_k)

            # Mean ZNCC over valid alt views per point
            zncc_stack = torch.stack(zncc_per_alt, dim=1)    # (Nh, n_alt)
            val_stack  = torch.stack(valid_per_alt, dim=1)   # (Nh, n_alt)
            n_valid    = val_stack.float().sum(dim=1)         # (Nh,)
            zncc_mean  = (zncc_stack * val_stack.float()).sum(dim=1) / n_valid.clamp(min=1e-6)
            has_valid  = n_valid > 0

            idx = torch.where(hit)[0]
            zncc_map[start + idx[has_valid]]  = zncc_mean[has_valid]
            valid_map[start + idx[has_valid]] = True

        # ── Build heatmap image ──────────────────────────────────────────────
        zncc_img = zncc_map.reshape(H, W).cpu().numpy()  # [-1, 1], nan = no hit
        hit_img  = hit_map.reshape(H, W).cpu().numpy()
        valid_img = valid_map.reshape(H, W).cpu().numpy()

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from scipy.ndimage import binary_dilation, binary_erosion

        # ZNCC heatmap: blue=low(-1) → red=high(+1), grey=no hit
        cmap = plt.cm.RdYlGn   # red=bad, green=good
        zncc_norm = (zncc_img + 1) / 2   # [0, 1]
        heat = cmap(zncc_norm)[:, :, :3]  # (H, W, 3)
        heat[~valid_img] = [0.3, 0.3, 0.3]   # grey = no hit / no valid alt
        heat[~hit_img]   = [0.15, 0.15, 0.15] # dark = miss

        heat_u8 = (heat * 255).astype(np.uint8)

        # Overlay: original image + ZNCC tint
        orig_full = images[vi_ref].cpu().numpy()  # may be full-res if images not downscaled
        if orig_full.max() <= 1.0:
            orig_full = (orig_full * 255).astype(np.uint8)
        else:
            orig_full = orig_full.astype(np.uint8)
        # Resize to match the traced resolution (H, W)
        orig_u8 = np.array(Image.fromarray(orig_full).resize((W, H)))

        # Blend: 60% original + 40% heatmap, only on hits
        blend = orig_u8.astype(np.float32) * 0.5 + heat_u8.astype(np.float32) * 0.5
        blend[~hit_img] = orig_u8[~hit_img].astype(np.float32) * 0.3
        blend_u8 = blend.clip(0, 255).astype(np.uint8)

        # Silhouette boundary in white
        if masks is not None:
            m_full = masks[vi_ref].cpu().numpy().astype(np.uint8) * 255
            m = np.array(Image.fromarray(m_full).resize((W, H))) > 128
            se = np.ones((5, 5), dtype=bool)
            bnd = binary_dilation(m, se) & ~binary_erosion(m, se)
            blend_u8[bnd] = [255, 255, 255]
            heat_u8[bnd]  = [255, 255, 255]

        # Colorbar strip (left side, 20px wide)
        bar_h = H; bar_w = 20
        bar_vals = np.linspace(1, 0, bar_h)
        bar_rgb  = (cmap(bar_vals)[:, :3] * 255).astype(np.uint8)
        bar_img  = np.repeat(bar_rgb[:, None, :], bar_w, axis=1)

        combined = np.concatenate([
            bar_img,
            np.array(Image.fromarray(heat_u8).resize((W, H))),
            np.array(Image.fromarray(blend_u8).resize((W, H))),
        ], axis=1)

        out_path = out / f"view{vi_ref:03d}_zncc.png"
        Image.fromarray(combined).save(out_path)

        # Stats
        z = zncc_img[valid_img]
        if z.size > 0:
            print(f"  hits={hit_img.sum()}/{N} ({100*hit_img.mean():.1f}%)  "
                  f"valid_zncc={valid_img.sum()}  "
                  f"zncc mean={z.mean():.3f}  p10={np.percentile(z,10):.3f}  "
                  f"p25={np.percentile(z,25):.3f}  p50={np.percentile(z,50):.3f}  "
                  f"frac_low(zncc<0.3)={(z < 0.3).mean():.2f}")
        else:
            print(f"  hits={hit_img.sum()}/{N}  NO valid ZNCC — check K/image alignment")
        print(f"  → {out_path}")

    print(f"\nlegend: green=high ZNCC (NCC agrees)  red=low ZNCC (NCC fights)  grey=no hit  white=silhouette boundary")


if __name__ == "__main__":
    main()
