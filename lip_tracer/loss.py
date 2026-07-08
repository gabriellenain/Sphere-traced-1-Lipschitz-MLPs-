"""Loss functions for 1-Lip sphere-tracing training."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import FTheta
from .sphere_tracing import trace_nograd


# ---------- radial distortion ----------

def radial_distort(xn: Tensor, k1: Tensor, k2: Tensor) -> Tensor:
    """Forward radial distortion on *normalised* image coords (..., 2).

        x_d = x · (1 + k1·r² + k2·r⁴),   r² = xn·xn.

    k1, k2 are scalars (one shared lens). With k1=k2=0 this is the identity, so
    the pinhole path is recovered exactly. Used at every 3D→pixel reprojection."""
    r2 = (xn * xn).sum(-1, keepdim=True)
    return xn * (1.0 + k1 * r2 + k2 * r2 * r2)


def _distort_project(xc: Tensor, K: Tensor, distort) -> Tensor:
    """Project camera-space points xc (..., 3) to distorted pixels (..., 2).

    `distort` maps normalised coords (..., 2) → distorted. `K` is (B, 3, 3); a
    trailing patch axis on xc (e.g. xc=(B, P, 3)) is broadcast automatically.
    Only the radial path — callers keep the plain pinhole K·xc divide for None."""
    xn = xc[..., :2] / xc[..., 2:3].clamp(min=1e-6)
    xn = distort(xn)
    extra = xc.dim() - 2                       # patch axes between B and the 2-vec
    sl = (slice(None),) + (None,) * extra
    fx = K[:, 0, 0][sl]; fy = K[:, 1, 1][sl]
    cx = K[:, 0, 2][sl]; cy = K[:, 1, 2][sl]
    return torch.stack([xn[..., 0] * fx + cx, xn[..., 1] * fy + cy], dim=-1)


# ---------- photometric ----------

def bilinear_sample(images: Tensor, vi: Tensor, uv: Tensor, H: int, W: int) -> Tensor:
    """Sample one colour per ray via bilinear interpolation.

    images : (V, H, W, 3) — full image stack, stays on GPU, never gathered
    vi     : (B,) long    — view index per ray
    uv     : (B, 2) float — pixel coords [x, y]
    Returns  (B, 3)
    """
    u = uv[:, 0].clamp(0, W - 1);  v = uv[:, 1].clamp(0, H - 1)
    u0 = u.long();  u1 = (u0 + 1).clamp(max=W - 1)
    v0 = v.long();  v1 = (v0 + 1).clamp(max=H - 1)
    wu = (u - u0.float()).unsqueeze(-1)   # (B, 1)
    wv = (v - v0.float()).unsqueeze(-1)   # (B, 1)
    return (images[vi, v0, u0].float() * (1 - wu) * (1 - wv)
            + images[vi, v0, u1].float() * wu       * (1 - wv)
            + images[vi, v1, u0].float() * (1 - wu) * wv
            + images[vi, v1, u1].float() * wu       * wv)


def gaussian_sample(images: Tensor, vi: Tensor, uv: Tensor, H: int, W: int,
                    sigma: float = 0.8, radius: int = 2) -> Tensor:
    """Gaussian-weighted image sample — smooth drop-in for bilinear_sample.

    Uses a (2r+1)² stencil centred on uv with Gaussian weights.
    Gradient w.r.t. uv is smooth everywhere (no bilinear kink at pixel edges).

    images : (V, H, W, 3)
    vi     : (B,) long
    uv     : (B, 2) float
    Returns  (B, 3)
    """
    r  = radius
    ks = 2 * r + 1
    ax = torch.arange(-r, r + 1, device=uv.device, dtype=torch.float32)   # (ks,)

    frac_x = uv[:, 0] - uv[:, 0].floor()   # (B,) offset within pixel
    frac_y = uv[:, 1] - uv[:, 1].floor()

    wx = torch.exp(-0.5 * ((ax[None, :] - frac_x[:, None]) / sigma) ** 2)  # (B, ks)
    wy = torch.exp(-0.5 * ((ax[None, :] - frac_y[:, None]) / sigma) ** 2)
    w2d = (wy[:, :, None] * wx[:, None, :])             # (B, ks, ks)
    w2d = w2d / w2d.sum(dim=(1, 2), keepdim=True)       # normalise

    cx = (uv[:, 0].long()[:, None, None] + ax[None, None, :].long()).clamp(0, W - 1)  # (B, 1, ks)
    cy = (uv[:, 1].long()[:, None, None] + ax[None, :, None].long()).clamp(0, H - 1)  # (B, ks, 1)
    colors = images[vi[:, None, None], cy, cx].float()  # (B, ks, ks, 3)
    return (colors * w2d.unsqueeze(-1)).sum(dim=(1, 2))  # (B, 3)


def _pick_sampler(sample_mode: str):
    """Return the point-sampler function for the given mode."""
    if sample_mode == "gaussian":
        return gaussian_sample
    return bilinear_sample


def _extract_patch(images: Tensor, vi: Tensor, uv: Tensor,
                   H: int, W: int, patch: int) -> Tensor:
    """Extract bilinearly-interpolated patches without materialising full per-batch images.

    images : (V, H, W, 3)  — full image stack on GPU
    vi     : (B,)           — view index per ray
    uv     : (B, 2)         — pixel coords (x, y)
    Returns  (B, 3*patch*patch)

    Memory: O(B * P²) instead of O(B * H * W).
    """
    P    = patch
    half = (P - 1) / 2.0
    oy   = torch.linspace(-half, half, P, device=images.device)   # (P,)
    ox   = torch.linspace(-half, half, P, device=images.device)   # (P,)

    # Float patch coords: (B, P, P)
    px = (uv[:, 0, None, None] + ox[None, None, :]).clamp(0, W - 1)
    py = (uv[:, 1, None, None] + oy[None, :, None]).clamp(0, H - 1)

    x0 = px.long();  x1 = (x0 + 1).clamp(max=W - 1)
    y0 = py.long();  y1 = (y0 + 1).clamp(max=H - 1)
    wx = (px - x0.float()).unsqueeze(-1)   # (B, P, P, 1)
    wy = (py - y0.float()).unsqueeze(-1)   # (B, P, P, 1)

    vi_e = vi[:, None, None]               # (B, 1, 1) for broadcasting
    c00  = images[vi_e, y0, x0].float()   # (B, P, P, 3)
    c10  = images[vi_e, y1, x0].float()
    c01  = images[vi_e, y0, x1].float()
    c11  = images[vi_e, y1, x1].float()

    vals = c00 * (1 - wx) * (1 - wy) + c01 * wx * (1 - wy) \
         + c10 * (1 - wx) * wy       + c11 * wx * wy         # (B, P, P, 3)
    return vals.reshape(vi.shape[0], -1)                      # (B, 3*P*P)


def pmvs_ncc_loss(
    images: Tensor,
    x3d: Tensor, normals: Tensor,
    vi_a: Tensor, vi_b: Tensor,
    K_all: Tensor, w2c_all: Tensor,
    H: int, W: int,
    patch: int = 7, half_pix: float = 3.0,
    sample_mode: str = "bilinear",
    gaussian_sigma: float = 0.8, gaussian_radius: int = 2,
    ncc_min: float = 0.4,
    return_full: bool = False,
    ncc_color: str = "gray",
    ncc_grad_alpha: float = 0.0,
    patch_wsigma: float = 0.0,
    patch_bilateral_gamma: float = 0.0,
    world_patch: float = -1.0,
    distort=None,
    dbg: dict | None = None,
) -> tuple[Tensor, Tensor, int] | tuple[Tensor, Tensor, int, Tensor]:
    """PMVS-style ZNCC: NCC on a 3D oriented patch projected into two views.

    Builds a P×P grid of 3D points on the tangent plane at c(p) with normal n(p),
    projects the grid into both views, then computes ZNCC on the resulting patches.
    Scale is set so the grid spans ±half_pix pixels in the reference view.

    x3d     : (B, 3)  — 3D surface points c(p)
    normals : (B, 3)  — surface normals n(p) (need not be unit)
    vi_a    : (B,)    — reference view index
    vi_b    : (B,)    — target view index
    Returns (N,) loss ∈ [0, 2], 0 = perfect correlation.
    """
    P = patch
    B_in = x3d.shape[0]

    def _full(zncc_t: Tensor | None, valid_m: Tensor | None,
              textured_m: Tensor | None) -> Tensor:
        """Per-input ZNCC aligned to x3d (B_in,), NaN where unusable.
        Grad flows into the usable entries (in-place index assign)."""
        zf = x3d.new_full((B_in,), float("nan"))
        if zncc_t is not None and zncc_t.numel() > 0:
            vidx = valid_m.nonzero(as_tuple=True)[0]      # (Bv,)
            tidx = vidx[textured_m]                        # (Bt,)
            zf = zf.clone()
            zf[tidx] = zncc_t
        return zf

    # 1. Orthonormal tangent frame from normals
    n = F.normalize(normals, dim=-1)                         # (B, 3)
    up = n.new_zeros(n.shape[0], 3); up[:, 1] = 1.0
    swap = n[:, 1].abs() > 0.9                               # near-vertical normals
    up[swap, 1] = 0.0;  up[swap, 0] = 1.0
    t1 = F.normalize(torch.cross(n, up, dim=-1), dim=-1)    # (B, 3)
    t2 = torch.cross(n, t1, dim=-1)                          # (B, 3)

    # 2. World-space step calibrated to ±half_pix pixels in the reference view
    R_a   = w2c_all[vi_a, :3, :3]                                              # (B, 3, 3)
    t_a   = w2c_all[vi_a, :3, 3]                                               # (B, 3)
    xc_a  = torch.einsum('bij,bj->bi', R_a, x3d) + t_a                        # (B, 3)
    z_ref = xc_a[:, 2].clamp(min=1e-3)                                         # (B,)
    f_x   = K_all[vi_a, 0, 0]                                                  # (B,)
    if world_patch > 0.0:
        # Object-fixed: constant world footprint (full grid span = world_patch),
        # view-independent. Footprint no longer floats with the reference
        # distance; far/low-GSD views whose patch goes sub-pixel collapse to a
        # near-constant patch and are dropped by the texture gate (std>1e-4).
        step_3d = x3d.new_full((z_ref.shape[0],), world_patch / max(P - 1, 1)).detach()  # (B,)
    else:
        step_3d = ((2.0 * half_pix / max(P - 1, 1)) * z_ref / f_x).detach()    # (B,) — patch scale fixed, gradient only through center

    # 3. P×P grid of 3D points on the tangent plane
    offs = torch.linspace(-(P - 1) / 2, (P - 1) / 2, P, device=x3d.device)
    oi, oj = torch.meshgrid(offs, offs, indexing='ij')       # (P, P)
    oi = oi.reshape(-1);  oj = oj.reshape(-1)                # (P*P,)
    # Spatial patch weight w = exp(-r/α) over grid points (r = dist to centre).
    # None → uniform (legacy). Used as weighted ZNCC moments below.
    if patch_wsigma > 0.0:
        _r = torch.sqrt(oi * oi + oj * oj)                   # (P*P,)
        pw = torch.exp(-_r / patch_wsigma).reshape(1, -1, 1)  # (1, P*P, 1)
    else:
        pw = None
    pts3d = (x3d.unsqueeze(1)
             + step_3d[:, None, None] * (oi[None, :, None] * t1.unsqueeze(1)
                                        + oj[None, :, None] * t2.unsqueeze(1)))  # (B, P*P, 3)

    # 4. Project grid into both views
    def _project(vi: Tensor) -> tuple[Tensor, Tensor]:
        R   = w2c_all[vi, :3, :3]                                                   # (B, 3, 3)
        t_v = w2c_all[vi, :3, 3]                                                    # (B, 3)
        xc  = (R.unsqueeze(1) @ pts3d.unsqueeze(-1)).squeeze(-1) + t_v.unsqueeze(1) # (B, P*P, 3)
        if distort is None:
            ph  = (K_all[vi].unsqueeze(1) @ xc.unsqueeze(-1)).squeeze(-1)           # (B, P*P, 3)
            uv  = ph[:, :, :2] / ph[:, :, 2:3].clamp(min=1e-6)                     # (B, P*P, 2)
        else:
            uv  = _distort_project(xc, K_all[vi], distort)                          # (B, P*P, 2)
        return uv, xc[:, :, 2]                                                       # (B,P*P,2), (B,P*P)

    uv_a, z_pa = _project(vi_a)
    uv_b, z_pb = _project(vi_b)

    # 5. Validity: all patch pixels in-bounds and positive depth in both views
    all_in_a = (z_pa > 0).all(1) \
             & (uv_a[:, :, 0] >= 0).all(1) & (uv_a[:, :, 0] < W).all(1) \
             & (uv_a[:, :, 1] >= 0).all(1) & (uv_a[:, :, 1] < H).all(1)
    all_in_b = (z_pb > 0).all(1) \
             & (uv_b[:, :, 0] >= 0).all(1) & (uv_b[:, :, 0] < W).all(1) \
             & (uv_b[:, :, 1] >= 0).all(1) & (uv_b[:, :, 1] < H).all(1)
    valid = all_in_a & all_in_b
    if not valid.any():
        empty = torch.empty(0, device=images.device)
        if return_full:
            return empty, empty.bool(), 0, _full(None, None, None)
        return empty, empty.bool(), 0

    uv_av = uv_a[valid];  uv_bv = uv_b[valid]    # (B', P*P, 2)
    vi_av = vi_a[valid];  vi_bv = vi_b[valid]     # (B',)

    # 6. Sample at projected grid points
    def _sample_grid_bilinear(uv: Tensor, vi: Tensor) -> Tensor:
        u  = uv[:, :, 0].clamp(0, W - 1)    # (B', P*P)
        v  = uv[:, :, 1].clamp(0, H - 1)
        u0 = u.long();  u1 = (u0 + 1).clamp(max=W - 1)
        v0 = v.long();  v1 = (v0 + 1).clamp(max=H - 1)
        wu = (u - u0.float()).unsqueeze(-1)
        wv = (v - v0.float()).unsqueeze(-1)
        vi_e = vi[:, None]
        c00  = images[vi_e, v0, u0].float(); c10 = images[vi_e, v1, u0].float()
        c01  = images[vi_e, v0, u1].float(); c11 = images[vi_e, v1, u1].float()
        return (c00 * (1 - wu) * (1 - wv) + c01 * wu * (1 - wv)
              + c10 * (1 - wu) * wv       + c11 * wu * wv)   # (B', P*P, 3)

    def _sample_grid_gaussian(uv: Tensor, vi: Tensor) -> Tensor:
        r  = gaussian_radius;  sigma = gaussian_sigma
        ax = torch.arange(-r, r + 1, device=uv.device, dtype=torch.float32)  # (ks,)
        frac_x = uv[:, :, 0] - uv[:, :, 0].floor()   # (B', P*P)
        frac_y = uv[:, :, 1] - uv[:, :, 1].floor()
        wx = torch.exp(-0.5 * ((ax[None, None, :] - frac_x[:, :, None]) / sigma) ** 2)  # (B', P*P, ks)
        wy = torch.exp(-0.5 * ((ax[None, None, :] - frac_y[:, :, None]) / sigma) ** 2)
        w2d = wy[:, :, :, None] * wx[:, :, None, :]      # (B', P*P, ks, ks)
        w2d = w2d / w2d.sum(dim=(-2, -1), keepdim=True)
        cx = (uv[:, :, 0].long()[:, :, None, None] + ax[None, None, None, :].long()).clamp(0, W - 1)
        cy = (uv[:, :, 1].long()[:, :, None, None] + ax[None, None, :, None].long()).clamp(0, H - 1)
        colors = images[vi[:, None, None, None], cy, cx].float()  # (B', P*P, ks, ks, 3)
        return (colors * w2d.unsqueeze(-1)).sum(dim=(-3, -2))      # (B', P*P, 3)

    _sample_grid = _sample_grid_gaussian if sample_mode == "gaussian" else _sample_grid_bilinear
    pa = _sample_grid(uv_av, vi_av)
    pb = _sample_grid(uv_bv, vi_bv)
    if ncc_color == "gray":
        # Rec.601 luminance — single channel, robust to per-channel
        # exposure/white-balance drift between DTU views (3x cheaper too).
        _lw = pa.new_tensor([0.299, 0.587, 0.114])
        pa = (pa * _lw).sum(-1, keepdim=True)
        pb = (pb * _lw).sum(-1, keepdim=True)

    # 7. Per-channel ZNCC averaged over channels
    raw_a, raw_b = pa, pb                             # keep for gradient term

    # Gipuma (Galliani et al. 2015) adaptive support weights: weight each patch
    # pixel by photometric similarity to the centre IN THE REFERENCE VIEW (a),
    # w(p,q)=exp(-|I_p-I_q|/γ). Makes a large patch edge-aware so it never bleeds
    # across depth/object boundaries. Detached → fixed support weights (the cost,
    # not the weights, drives geometry). Combines multiplicatively with spatial pw.
    if patch_bilateral_gamma > 0.0:
        c = (pa.shape[1] - 1) // 2                     # centre index of the P*P grid
        d_ref = (pa - pa[:, c:c + 1, :]).abs().mean(dim=2, keepdim=True)   # (B', P*P, 1)
        pw_bil = torch.exp(-d_ref / patch_bilateral_gamma).detach()
        pw = pw_bil if pw is None else pw * pw_bil

    if pw is not None:
        _wsum = pw.sum(dim=1, keepdim=True)            # per-patch (B',1,1) or shared (1,1,1)
        _sw = pw.sqrt()

        def _center(p: Tensor) -> Tensor:
            # weighted zero-mean, then ×√w so the existing norm/dot below
            # yield weighted std / weighted covariance (proper weighted ZNCC).
            return (p - (pw * p).sum(dim=1, keepdim=True) / _wsum) * _sw
    else:
        def _center(p: Tensor) -> Tensor:
            return p - p.mean(dim=1, keepdim=True)

    pa = _center(pa)                                 # (B', P*P, C)
    pb = _center(pb)
    std_a = pa.norm(dim=1)                           # (B', C)
    std_b = pb.norm(dim=1)
    textured = (std_a > 1e-4).all(1) & (std_b > 1e-4).all(1)
    if not textured.any():
        empty = torch.empty(0, device=images.device)
        if return_full:
            return empty, empty.bool(), int(valid.sum()), _full(None, None, None)
        return empty, empty.bool(), int(valid.sum())
    pa = pa[textured] / std_a[textured].unsqueeze(1).clamp(min=1e-6)  # (B'', P*P, C)
    pb = pb[textured] / std_b[textured].unsqueeze(1).clamp(min=1e-6)
    zncc = (pa * pb).sum(dim=1).mean(dim=1).clamp(-1.0, 1.0)         # (B'',) mean over channels

    if ncc_grad_alpha > 0.0:
        # Gipuma-style edge term, but ZNCC-consistent: a second ZNCC on the
        # patch gradient magnitude (sharper minimum than intensity NCC).
        # ρ = (1-α)(1-ZNCC_I) + α(1-ZNCC_∇)  ⇔  zncc ← (1-α)·zncc_I + α·zncc_∇
        def _grad_mag(p: Tensor) -> Tensor:
            p2 = p.reshape(p.shape[0], P, P, -1)
            gx = torch.zeros_like(p2);  gy = torch.zeros_like(p2)
            gx[:, :, 1:-1, :] = 0.5 * (p2[:, :, 2:, :] - p2[:, :, :-2, :])
            gy[:, 1:-1, :, :] = 0.5 * (p2[:, 2:, :, :] - p2[:, :-2, :, :])
            return torch.sqrt(gx * gx + gy * gy + 1e-12).reshape(
                p.shape[0], P * P, -1)
        ga = _center(_grad_mag(raw_a)[textured])
        gb = _center(_grad_mag(raw_b)[textured])
        ga = ga / ga.norm(dim=1, keepdim=True).clamp(min=1e-6)
        gb = gb / gb.norm(dim=1, keepdim=True).clamp(min=1e-6)
        zncc_g = (ga * gb).sum(dim=1).mean(dim=1).clamp(-1.0, 1.0)
        if dbg is not None and zncc.numel() > 0:
            dbg["zncc_I"] = float(zncc.detach().mean())
            dbg["zncc_grad"] = float(zncc_g.detach().mean())
        zncc = ((1.0 - ncc_grad_alpha) * zncc
                + ncc_grad_alpha * zncc_g).clamp(-1.0, 1.0)
    keep = zncc > ncc_min                                             # PMVS photometric gate
    if return_full:
        return zncc, keep, int(valid.sum()), _full(zncc, valid, textured)
    return zncc, keep, int(valid.sum())


def photo_loss(
    f: FTheta,
    x_theta: Tensor, hit: Tensor, n: Tensor,
    vi: Tensor, alt_nn: Tensor, origins_all: Tensor,
    images: Tensor, K_all: Tensor, w2c_all: Tensor,
    feature_maps: Tensor | None,
    masks: Tensor | None, fg_self: Tensor,
    H: int, W: int,
    uv_self: Tensor,
    n_alt: int, cos_thresh: float,
    w_photo: float, w_feature: float, w_ncc: float, ncc_patch: int, ncc_half_pix: float,
    sample_mode: str, gaussian_sigma: float, gaussian_radius: int,
    step: int, ncc_min: float = 0.4, occ_mode: str = "pinhole",
    hit_bg: Tensor | None = None,
    ncc_sat_tau: float = -1.0,
    w_ncc_normal: float = 0.0,
    ncc_detach_normals: bool = True,
    ncc_topk: int = 0,
    ncc_abs_tau: float = -1.0,
    ncc_normal_patch: int = -1,
    ncc_normal_half_pix: float = -1.0,
    ncc_color: str = "gray",
    ncc_grad_alpha: float = 0.0,
    ncc_patch_wsigma: float = 0.0,
    ncc_patch_bilateral_gamma: float = 0.0,
    ncc_world_patch: float = -1.0,
    distort=None,
    trace_cfg=None,
    prof=None,
) -> tuple[Tensor, dict]:
    """Multi-view photoconsistency loss with occlusion test.

    Returns (loss, debug_stats_dict).
    hit_bg: optional bool (B,) — bounding-sphere exit rays; tracked separately in stats.
    trace_cfg: optional TraceConfig passed to the occlusion trace_nograd calls
        (so the K-budget matches the primary trace). Falls back to the module default.
    prof: optional StepProfiler — if set, the occlusion trace is recorded under
        its own "occ_trace" bucket (separate from photo+NCC compute).
    """
    B = vi.shape[0]
    alt = alt_nn[vi]  # (B, n_alt)

    # primary-camera colour — (B, 3) only, no (B, H, W, 3) intermediate
    _sampler = _pick_sampler(sample_mode)
    _skw = dict(sigma=gaussian_sigma, radius=gaussian_radius)
    c_self = (_sampler(images, vi, uv_self, H, W, **_skw)
              if sample_mode == "gaussian" else bilinear_sample(images, vi, uv_self, H, W))
    # Reference-saturation gate: drop rays whose reference pixel is near-saturated
    # (max channel ≥ τ) — a specular highlight blown out in the reference view
    # contaminates every (ref,alt) pair, so it must be removed ray-wise, not per
    # pair. Disabled when ncc_sat_tau ≤ 0.
    if ncc_sat_tau > 0.0:
        ref_unsat = c_self.amax(dim=-1) < ncc_sat_tau          # (B,)
    else:
        ref_unsat = None
    n_ref_sat = 0 if ref_unsat is None else int((~ref_unsat).sum())
    feat_self = None
    Hf = Wf = 0
    if feature_maps is not None and w_feature > 0:
        Hf, Wf = feature_maps.shape[1:3]
        uv_self_f = uv_self * uv_self.new_tensor([Wf / W, Hf / H])
        feat_self = F.normalize(bilinear_sample(feature_maps, vi, uv_self_f, Hf, Wf).float(), dim=-1)

    # Normal-branch patch geometry: sentinel <0 → share the position branch's.
    n_patch    = ncc_normal_patch    if ncc_normal_patch    > 0 else ncc_patch
    n_half_pix = ncc_normal_half_pix if ncc_normal_half_pix > 0 else ncc_half_pix

    loss_terms: list[Tensor] = []
    l1_vals:   list[Tensor] = []
    feat_vals: list[Tensor] = []
    ncc_vals:  list[Tensor] = []
    ncc_n_vals: list[Tensor] = []
    ncc_pos_terms: list[Tensor] = []   # raw (un-weighted) terms kept for per-branch
    ncc_n_terms:   list[Tensor] = []   # gradient diagnostics (∂NCC/∂x vs ∂NCC/∂n)
    ncc_zncc_vals: list[Tensor] = []
    ncc_zncc_I_vals: list[float] = []      # intensity-only ZNCC (grad-blend split)
    ncc_zncc_g_vals: list[float] = []      # gradient-magnitude ZNCC
    ncc_kept = ncc_textured = ncc_valid = 0
    ncc_n_kept = ncc_n_textured = ncc_n_valid = 0
    # Σ zncc over kept (z>ncc_min, textured) pairs. Its mean is the ZNCC the
    # loss actually optimizes (≈ 1 - loss), as opposed to ncc_zncc which is the
    # mean over ALL valid pairs (incl. the dropped anti-correlated ones).
    ncc_zncc_kept_sum = 0.0

    # Top-K robust aggregation: per surface point, keep the K best-correlating
    # alt views across the n_alt pool (PMVS/COLMAP-style occlusion/grazing
    # rejection) instead of averaging (1−ZNCC) over all valid views. Collect
    # per-view full-length ZNCC columns, reduce after the loop.
    use_topk = ncc_topk > 0
    zpos_cols: list[Tensor] = []
    znrm_cols: list[Tensor] = []

    def _scatter_col(zf: Tensor, midx: Tensor) -> Tensor:
        col = x_theta.new_full((B,), float("nan"))
        if zf.numel() > 0:
            col = col.clone()
            col[midx] = zf
        return col
    n_total = n_in_frame = n_not_occl = n_cos_ok = n_mask = n_mask_bg = 0

    alt_flat = alt.reshape(-1)                                        # (B*n_alt,)
    op_all   = origins_all[alt_flat]                                  # (B*n_alt, 3)
    x_flat   = x_theta.detach().unsqueeze(1).expand(B, n_alt, 3).reshape(-1, 3)
    dir_all  = op_all - x_flat                                        # hit-point → alt cam
    dist_all = dir_all.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dp_all   = dir_all / dist_all                                     # unit, hit→alt

    _occ_cm = prof.timed("occ_trace") if prof is not None else contextlib.nullcontext()
    # The occlusion trace is thresholded to a visibility boolean, so it can run at
    # a coarser config than the primary trace. Sentinel (<0) occ_* fields fall back
    # to the primary values, so the default behaviour is unchanged.
    if trace_cfg is not None:
        import dataclasses
        occ_cfg = dataclasses.replace(
            trace_cfg,
            iters=trace_cfg.occ_iters if trace_cfg.occ_iters > 0 else trace_cfg.iters,
            newton_steps=(trace_cfg.occ_newton_steps
                          if trace_cfg.occ_newton_steps >= 0 else trace_cfg.newton_steps),
            eps=trace_cfg.occ_eps if trace_cfg.occ_eps > 0 else trace_cfg.eps,
        )
        _trace_kw = {"cfg": occ_cfg}
        _occ_slack = trace_cfg.occ_depth_slack
    else:
        _trace_kw = {}
        _occ_slack = 1e-2
    with _occ_cm:
        if occ_mode == "from_hit":
            # Trace from hit point toward the alt camera instead of from the pinhole.
            # Skips the empty space traversal so each trace converges in far fewer steps.
            # Occlusion check: trace must NOT hit anything before reaching the alt camera.
            _eps_occ = 1e-2
            with torch.no_grad():
                _, tp_all, hitp_all = trace_nograd(f, x_flat + _eps_occ * dp_all, dp_all, **_trace_kw)
            tp_all   = tp_all.reshape(B, n_alt)
            hitp_all = hitp_all.reshape(B, n_alt)
        else:  # "pinhole"
            with torch.no_grad():
                _, tp_all, hitp_all = trace_nograd(f, op_all, -dp_all, **_trace_kw)
            tp_all   = tp_all.reshape(B, n_alt)
            hitp_all = hitp_all.reshape(B, n_alt)

    dist_all = dist_all.squeeze(-1).reshape(B, n_alt)
    dp_all   = dp_all.reshape(B, n_alt, 3)

    for k in range(n_alt):
        ak      = alt[:, k]
        op      = origins_all[ak]
        dp      = dp_all[:, k]
        dist    = dist_all[:, k]
        tp      = tp_all[:, k]
        hitp    = hitp_all[:, k]

        Kp   = K_all[ak];  w2cp = w2c_all[ak]
        xc   = torch.einsum("bij,bj->bi", w2cp[:, :3, :3], x_theta) + w2cp[:, :3, 3]
        if distort is None:
            uv_h = torch.einsum("bij,bj->bi", Kp, xc)
            uv   = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
        else:
            uv   = _distort_project(xc, Kp, distort)
        in_frame = (xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) \
                   & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        if occ_mode == "from_hit":
            # Trace started at hit-point toward alt cam: occluded iff it hits before reaching cam.
            not_occl = ~hitp | (tp > dist - _occ_slack)
        else:
            depth_ok = dist <= tp + _occ_slack
            not_occl = hitp & depth_ok

        if step % 50 == 0 and k == 0:
            delta = tp - dist
            print(f"  occ alt0: hitp={hitp.float().mean():.2f}  "
                  f"not_occl={not_occl.float().mean():.2f}  "
                  f"tp-dist min/mean/max={delta.min():.3f}/{delta.mean():.3f}/{delta.max():.3f}")

        cos_ok = (n * dp).sum(-1).abs() > cos_thresh
        if masks is not None:
            uv_c = uv.long().clamp(0).clone()
            uv_c[:, 0].clamp_(max=W - 1); uv_c[:, 1].clamp_(max=H - 1)
            fg_alt = masks[ak, uv_c[:, 1], uv_c[:, 0]]
            mask = hit & in_frame & not_occl & cos_ok & fg_alt & fg_self
        else:
            mask = hit & in_frame & not_occl & cos_ok
        if ref_unsat is not None:
            mask = mask & ref_unsat

        n_total    += B
        n_in_frame += int(in_frame.sum())
        n_not_occl += int(not_occl.sum())
        n_cos_ok   += int(cos_ok.sum())
        n_mask     += int(mask.sum())
        if hit_bg is not None:
            n_mask_bg += int((mask & hit_bg).sum())

        if step % 50 == 0 and k == 0:
            x_norm_str = (f"[{x_theta[hit].detach().norm(dim=-1).min():.3f},"
                          f"{x_theta[hit].detach().norm(dim=-1).max():.3f}]"
                          if hit.any() else "[no hits]")
            print(f"  dbg alt0: in_frame={in_frame.float().mean():.2f}  "
                  f"not_occl={not_occl.float().mean():.2f}  cos_ok={cos_ok.float().mean():.2f}  "
                  f"x_norm={x_norm_str}")

        if mask.any():
            gt_alt = (_sampler(images, ak, uv, H, W, **_skw)
                      if sample_mode == "gaussian" else bilinear_sample(images, ak, uv, H, W))
            if w_photo > 0:
                l1_term = (c_self - gt_alt).abs().sum(-1)[mask].mean()
                loss_terms.append(w_photo * l1_term)
                l1_vals.append(l1_term.detach())
            if feature_maps is not None and feat_self is not None and w_feature > 0:
                uv_f = uv * uv.new_tensor([Wf / W, Hf / H])
                feat_alt = F.normalize(bilinear_sample(feature_maps, ak, uv_f, Hf, Wf).float(), dim=-1)
                cos = (feat_self * feat_alt).sum(-1).clamp(-1.0, 1.0)
                feat_term = (1.0 - cos)[mask].mean()
                loss_terms.append(w_feature * feat_term)
                feat_vals.append(feat_term.detach())
            if w_ncc > 0:
                _dbg = {} if ncc_grad_alpha > 0.0 else None
                n_for_patch = n[mask].detach() if ncc_detach_normals else n[mask]
                zncc, keep, n_valid, *zf = pmvs_ncc_loss(
                    images,
                    x_theta[mask], n_for_patch,
                    vi[mask], ak[mask],
                    K_all, w2c_all,
                    H, W, ncc_patch, ncc_half_pix,
                    sample_mode, gaussian_sigma, gaussian_radius,
                    ncc_min,
                    return_full=use_topk,
                    ncc_color=ncc_color,
                    ncc_grad_alpha=ncc_grad_alpha,
                    patch_wsigma=ncc_patch_wsigma,
                    patch_bilateral_gamma=ncc_patch_bilateral_gamma,
                    world_patch=ncc_world_patch,
                    distort=distort,
                    dbg=_dbg,
                )
                if _dbg and "zncc_I" in _dbg:
                    ncc_zncc_I_vals.append(_dbg["zncc_I"])
                    ncc_zncc_g_vals.append(_dbg["zncc_grad"])
                ncc_valid    += n_valid
                ncc_textured += int(zncc.numel())
                ncc_kept     += int(keep.sum())
                if zncc.numel() > 0:
                    ncc_zncc_vals.append(zncc.detach().mean())
                if keep.any():
                    ncc_zncc_kept_sum += float(zncc[keep].detach().sum())
                if use_topk:
                    zpos_cols.append(_scatter_col(
                        zf[0], mask.nonzero(as_tuple=True)[0]))
                elif ncc_abs_tau >= 0.0:
                    # Fixed-batch hinge reward:  L = −(1/|B|) Σ_i H_i·[z_i − τ]_+.
                    # With τ=ncc_min the gradient flows through exactly the same
                    # pairs as the legacy keep-gate (z>τ); the ONLY change vs the
                    # kept-mean is the θ-independent denominator |B| (sampled batch).
                    # Deleting a z>τ hit therefore always costs its (z−τ)+ — the
                    # below-survivor-mean culling incentive is gone; z≤τ pairs are
                    # neutral (no gradient, deletion-neutral), matching ncc_min.
                    if zncc.numel() > 0:
                        ncc_term = -(zncc - ncc_abs_tau).clamp(min=0).sum() / B
                        loss_terms.append(w_ncc * ncc_term)
                        ncc_vals.append(ncc_term.detach())
                        ncc_pos_terms.append(ncc_term)
                elif keep.any():
                    ncc_term = (1.0 - zncc[keep]).mean()
                    loss_terms.append(w_ncc * ncc_term)
                    ncc_vals.append(ncc_term.detach())
                    ncc_pos_terms.append(ncc_term)
            if w_ncc_normal > 0:
                # Normal branch: position detached, normal carries gradient.
                # Gradient flows only through ∂NCC/∂n · ∂n/∂θ (local
                # orientation/curvature), decoupled from the level-set position.
                zncc_n, keep_n, n_valid_n, *zf_n = pmvs_ncc_loss(
                    images,
                    x_theta[mask].detach(), n[mask],
                    vi[mask], ak[mask],
                    K_all, w2c_all,
                    H, W, n_patch, n_half_pix,
                    sample_mode, gaussian_sigma, gaussian_radius,
                    ncc_min,
                    return_full=use_topk,
                    ncc_color=ncc_color,
                    ncc_grad_alpha=ncc_grad_alpha,
                    patch_wsigma=ncc_patch_wsigma,
                    patch_bilateral_gamma=ncc_patch_bilateral_gamma,
                    distort=distort,
                )
                ncc_n_valid    += n_valid_n
                ncc_n_textured += int(zncc_n.numel())
                ncc_n_kept     += int(keep_n.sum())
                if use_topk:
                    znrm_cols.append(_scatter_col(
                        zf_n[0], mask.nonzero(as_tuple=True)[0]))
                elif keep_n.any():
                    ncc_n_term = (1.0 - zncc_n[keep_n]).mean()
                    loss_terms.append(w_ncc_normal * ncc_n_term)
                    ncc_n_vals.append(ncc_n_term.detach())
                    ncc_n_terms.append(ncc_n_term)

    # --- top-K reduction over the alt-view pool ---
    def _topk_reduce(cols: list[Tensor]) -> Tensor | None:
        if not cols:
            return None
        Z = torch.stack(cols, dim=1)                       # (B, n_cols) w/ grad
        ok = ~torch.isnan(Z) & (Z > ncc_min)               # usable + gate
        Zf = Z.masked_fill(~ok, -2.0)                       # sentinel < [-1,1]
        K = min(ncc_topk, Zf.shape[1])
        vals, _ = Zf.topk(K, dim=1)                         # (B, K) best views
        sel = vals > -1.5                                   # real picks only
        cnt = sel.sum(1)                                    # (B,)
        rows = cnt > 0
        if not rows.any():
            return None
        per_row = torch.where(sel, 1.0 - vals,
                              torch.zeros_like(vals)).sum(1) / cnt.clamp(min=1)
        return per_row[rows].mean()

    # ZNCC diagnostics: mean over the WHOLE valid pool vs mean over the
    # top-K selected views (per surface point, then averaged). The gap
    # quantifies how much the robust selection buys over the plain mean.
    ncc_zncc_mean = ncc_zncc_topk = 0.0
    if use_topk and zpos_cols:
        with torch.no_grad():
            Zd = torch.stack(zpos_cols, dim=1)             # (B, n_cols)
            okd = ~torch.isnan(Zd) & (Zd > ncc_min)
            cpx = okd.sum(1)
            rmean = cpx > 0
            if rmean.any():
                zsum = torch.where(okd, Zd, torch.zeros_like(Zd)).sum(1)
                ncc_zncc_mean = float(
                    (zsum[rmean] / cpx[rmean].clamp(min=1)).mean())
                Zfd = Zd.masked_fill(~okd, -2.0)
                Kd = min(ncc_topk, Zfd.shape[1])
                vd, _ = Zfd.topk(Kd, dim=1)
                seld = vd > -1.5
                cks = seld.sum(1)
                rk = cks > 0
                ncc_zncc_topk = float(
                    (torch.where(seld, vd, torch.zeros_like(vd)).sum(1)[rk]
                     / cks[rk].clamp(min=1)).mean())

    if use_topk and w_ncc > 0:
        t = _topk_reduce(zpos_cols)
        if t is not None:
            loss_terms.append(w_ncc * t)
            ncc_vals.append(t.detach())
            ncc_pos_terms.append(t)
    if use_topk and w_ncc_normal > 0:
        tn = _topk_reduce(znrm_cols)
        if tn is not None:
            loss_terms.append(w_ncc_normal * tn)
            ncc_n_vals.append(tn.detach())
            ncc_n_terms.append(tn)

    # --- per-branch gradient diagnostic (every 50 steps) ---
    # Isolates the "leverage" ∂NCC/∂x (position branch) vs ∂NCC/∂n (normal
    # branch), with the network factors ∂x/∂θ, ∂n/∂θ excluded.
    #
    # The two raw norms are NOT comparable: ‖∂NCC/∂x‖ is ΔNCC per world-unit of
    # patch-*center* displacement, while ‖∂NCC/∂n‖ is ΔNCC per *radian* of
    # normal tilt (n is unit here — normalized in train.py before photo_loss
    # and again in pmvs_ncc_loss). A unit tilt of the tangent frame moves the
    # patch *edge* by the patch half-extent r = half_pix·z_ref/f_x — the exact
    # calibration step_3d uses. Dividing the normal leverage by r expresses it
    # as ΔNCC per world-unit of edge motion, the same units as the position
    # leverage, so ncc_grad_n_xeq and ncc_grad_pos can be compared directly
    # (and their ratio read as "how much orientation buys vs translation").
    # r is taken as a scalar mean over hit points (it depends only on the
    # surface point and its own reference view, not on the alt pairing).
    ncc_grad_pos = ncc_grad_pos_median = ncc_grad_pos_p90 = 0.0
    ncc_grad_n = ncc_grad_n_median = ncc_grad_n_p90 = 0.0
    ncc_grad_n_xeq = ncc_grad_n_xeq_median = ncc_grad_n_xeq_p90 = 0.0
    ncc_grad_ratio = 0.0
    if step % 50 == 0:
        if ncc_pos_terms and x_theta.requires_grad:
            gp = torch.autograd.grad(torch.stack(ncc_pos_terms).mean(), x_theta,
                                     retain_graph=True, allow_unused=True)[0]
            if gp is not None:
                gp_norm = gp.norm(dim=-1)
                ncc_grad_pos = gp_norm.mean().item()
                ncc_grad_pos_median = gp_norm.median().item()
                ncc_grad_pos_p90 = torch.quantile(gp_norm, 0.9).item()
        if ncc_n_terms and n.requires_grad:
            gn = torch.autograd.grad(torch.stack(ncc_n_terms).mean(), n,
                                     retain_graph=True, allow_unused=True)[0]
            if gn is not None:
                gn_norm = gn.norm(dim=-1)
                ncc_grad_n = gn_norm.mean().item()
                ncc_grad_n_median = gn_norm.median().item()
                ncc_grad_n_p90 = torch.quantile(gn_norm, 0.9).item()
                # Patch half-extent r = half_pix·z_ref/f_x in world units.
                with torch.no_grad():
                    sel = hit if hit.any() else torch.ones_like(hit)
                    xs = x_theta[sel].detach()
                    vs = vi[sel]
                    R_r = w2c_all[vs, :3, :3]
                    t_r = w2c_all[vs, :3, 3]
                    z_ref = (torch.einsum('bij,bj->bi', R_r, xs) + t_r)[:, 2].clamp(min=1e-3)
                    f_x = K_all[vs, 0, 0]
                    r_patch = float((n_half_pix * z_ref / f_x).mean())
                if r_patch > 1e-12:
                    inv_r = 1.0 / r_patch
                    ncc_grad_n_xeq = ncc_grad_n * inv_r
                    ncc_grad_n_xeq_median = ncc_grad_n_median * inv_r
                    ncc_grad_n_xeq_p90 = ncc_grad_n_p90 * inv_r
                    if ncc_grad_pos > 1e-12:
                        ncc_grad_ratio = ncc_grad_n_xeq / ncc_grad_pos

    if loss_terms:
        loss = torch.stack(loss_terms).mean()
    else:
        loss = f(x_theta.detach()[:1]).sum() * 0.0
    stats = dict(n_mask=n_mask, n_in_frame=n_in_frame,
                 n_not_occl=n_not_occl, n_cos_ok=n_cos_ok, n_total=n_total,
                 n_mask_bg=n_mask_bg, n_ref_sat=n_ref_sat,
                 l1=torch.stack(l1_vals).mean().item() if l1_vals else 0.0,
                 feature=torch.stack(feat_vals).mean().item() if feat_vals else 0.0,
                 ncc=torch.stack(ncc_vals).mean().item() if ncc_vals else 0.0,
                 ncc_normal=torch.stack(ncc_n_vals).mean().item() if ncc_n_vals else 0.0,
                 ncc_weighted=(w_ncc * torch.stack(ncc_vals).mean().item()) if ncc_vals else 0.0,
                 ncc_normal_weighted=(w_ncc_normal * torch.stack(ncc_n_vals).mean().item()) if ncc_n_vals else 0.0,
                 ncc_zncc=torch.stack(ncc_zncc_vals).mean().item() if ncc_zncc_vals else 0.0,
                 ncc_zncc_used=(ncc_zncc_kept_sum / ncc_kept) if ncc_kept else 0.0,
                 ncc_zncc_mean=ncc_zncc_mean,
                 ncc_zncc_topk=ncc_zncc_topk,
                 ncc_grad_alpha=ncc_grad_alpha,
                 ncc_zncc_I=(sum(ncc_zncc_I_vals) / len(ncc_zncc_I_vals)
                             if ncc_zncc_I_vals else 0.0),
                 ncc_zncc_grad=(sum(ncc_zncc_g_vals) / len(ncc_zncc_g_vals)
                                if ncc_zncc_g_vals else 0.0),
                 ncc_valid=ncc_valid, ncc_textured=ncc_textured, ncc_kept=ncc_kept,
                 ncc_n_valid=ncc_n_valid, ncc_n_textured=ncc_n_textured, ncc_n_kept=ncc_n_kept,
                 ncc_grad_pos=ncc_grad_pos,
                 ncc_grad_pos_median=ncc_grad_pos_median,
                 ncc_grad_pos_p90=ncc_grad_pos_p90,
                 ncc_grad_n=ncc_grad_n,
                 ncc_grad_n_median=ncc_grad_n_median,
                 ncc_grad_n_p90=ncc_grad_n_p90,
                 ncc_grad_n_xeq=ncc_grad_n_xeq,
                 ncc_grad_n_xeq_median=ncc_grad_n_xeq_median,
                 ncc_grad_n_xeq_p90=ncc_grad_n_xeq_p90,
                 ncc_grad_ratio=ncc_grad_ratio)
    return loss, stats


# ---------- geometry / regularisation ----------

def idr_mask_loss(
    f: FTheta, o: Tensor, d: Tensor, hit: Tensor, fg: Tensor,
    alpha: float, n_samples: int, t_near: float, t_far: float,
) -> tuple[Tensor, dict]:
    """IDR mask loss — faithful to Yariv et al. 2020, eq. (7) + appendix A.4.

    For P^out rays (¬(hit ∧ fg)):

        S_{p,α} = σ(−α · min_t f(c + t·v))
        loss    = 1/(α|P|) · Σ_{p∈P^out} BCE(O_p, S_{p,α})

    The min is a HARD min over `n_samples` stratified-uniform points along the
    ray in [t_near, t_far] — a dense sweep that is *independent of the sphere
    tracer*. torch.min routes gradient only through the argmin sample, which is
    exactly IDR's envelope-theorem gradient (∂ min_t f = ∂ f(c + t*·v)).

    No soft-min / β: that was a codebase modification whose logsumexp bias
    crippled the loss on hole rays. This is the paper's exact formulation.

    f  : SDF network        o,d : (B,3) ray origin / unit dir
    hit: (B,) tracer hit    fg  : (B,) GT foreground mask
    Returns (loss, stats).
    """
    pout = ~(hit & fg)                    # P^out
    fn   = pout & fg                      # false negatives: miss but should hit
    fp   = pout & hit                     # false positives: hit but should miss

    zero = o.new_zeros(1).squeeze()
    if pout.sum() == 0:
        return zero, dict(idr_n_pout=0, idr_n_fn=0, idr_n_fp=0,
                          idr_S_fn=float('nan'), idr_S_fp=float('nan'),
                          idr_sdf_fn=float('nan'), idr_sdf_fp=float('nan'))

    o_p = o[pout];  d_p = d[pout]
    fg_p = fg[pout].float()
    hit_p = hit[pout]
    M = int(pout.sum())

    # stratified uniform sweep of the ray (IDR samples 100 uniform points)
    dt   = (t_far - t_near) / n_samples
    base = torch.linspace(t_near, t_far - dt, n_samples, device=o.device)      # (N,)
    ts   = base.unsqueeze(0) + torch.rand(M, n_samples, device=o.device) * dt  # (M, N)

    # Envelope theorem (IDR §3.4 / appendix A.4): ∂ min_t f = ∂ f(c + t*·v),
    # the gradient flows only through the argmin point t*. Find t* under
    # no_grad over the full sweep, then evaluate f once at t* with gradient —
    # ~n_samples× cheaper backward, identical loss and gradient.
    with torch.no_grad():
        pts_all = o_p.unsqueeze(1) + ts.unsqueeze(-1) * d_p.unsqueeze(1)        # (M, N, 3)
        sdf_all = f(pts_all.reshape(-1, 3)).reshape(M, n_samples)
        t_star  = ts.gather(1, sdf_all.argmin(dim=1, keepdim=True)).squeeze(1)  # (M,)
    x_star  = o_p + t_star.unsqueeze(-1) * d_p                                  # (M, 3)
    sdf_min = f(x_star)                                                         # (M,) — grad only here

    sdf_pred = -alpha * sdf_min
    loss = (1.0 / alpha) * F.binary_cross_entropy_with_logits(
        sdf_pred, fg_p, reduction='sum'
    ) / float(fg.shape[0])

    with torch.no_grad():
        S = torch.sigmoid(sdf_pred)
        fg_b = fg_p.bool()
        S_fn = S[fg_b].mean().item()        if fn.any() else float('nan')
        S_fp = S[hit_p].mean().item()       if fp.any() else float('nan')
        sdf_fn = sdf_min[fg_b].mean().item()  if fn.any() else float('nan')
        sdf_fp = sdf_min[hit_p].mean().item() if fp.any() else float('nan')

    stats = dict(
        idr_n_pout=M,
        idr_n_fn=int(fn.sum()),    # miss & fg  — "holes", want S → 1
        idr_n_fp=int(fp.sum()),    # hit  & ~fg — "blobs", want S → 0
        idr_S_fn=S_fn,             # mean σ for fn rays (want → 1)
        idr_S_fp=S_fp,             # mean σ for fp rays (want → 0)
        idr_sdf_fn=sdf_fn,         # mean min-SDF for fn rays (want → 0/neg)
        idr_sdf_fp=sdf_fp,         # mean min-SDF for fp rays
    )
    return loss, stats


def mask_loss_min_sdf(
    sdf_min: Tensor, fg: Tensor, alpha: float,
    fg_offset: float = 0.1,
    bg_offset: float = 0.05,
    focal_gamma: float = 2.0,
    balance_classes: bool = True,
    normalize_by_alpha: bool = False,
) -> Tensor:
    """Mask loss via minimum SDF along traced rays (Yariv et al. IDR / NeuS).

    S_{p,α} = σ(−α · (min_t f(o + td) − fg_offset · O_p + bg_offset · (1−O_p)))

    Improvements over the vanilla IDR formulation:

    - bg_offset (symmetric to fg_offset): a bg ray with sdf_min ≈ 0 (network
      surface grazes the ray, common at silhouette edges with noisy masks)
      gets sdf_adj = +bg_offset → S < 0.5 → low BCE on target=0. Without it,
      sdf_min = 0 yields S = 0.5 and a constant 0.69 loss for borderline bg
      rays, which is mostly mask noise.

    - focal_gamma > 0 weights each ray by (1 − p_correct)^γ. Easy hits
      (p ≈ 0.9) are downweighted ~100×; hard misses (p ≈ 0.02) keep full
      weight. Concentrates gradient on the few rays that actually disagree
      with the target — exactly the missing-piece rays. γ=2 standard.

    - balance_classes scales fg/bg contributions to 1 / class_frac so an
      imbalanced batch (typical DTU: ~30 % fg) doesn't dilute the fg signal.

    - normalize_by_alpha=True keeps the original 1/α scaling (gradient w.r.t.
      sdf_min ≈ 1 − S, capped); set False if scheduling α and you want the
      schedule to actually sharpen the signal.

    Works for all rays. Requires sdf_min from trace_unrolled (soft-min if
    cfg.sdf_min_beta > 0, hard min otherwise).
    """
    fg_f = fg.float()
    sdf_adj = sdf_min - fg_f * fg_offset + (1.0 - fg_f) * bg_offset
    S = torch.sigmoid(-alpha * sdf_adj).clamp(1e-6, 1.0 - 1e-6)

    bce = -(fg_f * torch.log(S) + (1.0 - fg_f) * torch.log(1.0 - S))   # (B,)

    if focal_gamma > 0:
        p_t = fg_f * S + (1.0 - fg_f) * (1.0 - S)                      # (B,)
        bce = bce * (1.0 - p_t).pow(focal_gamma)

    if balance_classes:
        fg_frac = fg_f.mean().clamp(1e-3, 1.0 - 1e-3)
        w = fg_f / fg_frac + (1.0 - fg_f) / (1.0 - fg_frac)            # mean(w) = 2
        bce = bce * w * 0.5                                            # rescale to mean(w) = 1

    loss = bce.mean()
    return loss / alpha if normalize_by_alpha else loss


def dvr_mask_loss(f: FTheta, o: Tensor, u: Tensor, fg: Tensor,
                  t_far: float, fg_margin: float, bg_margin: float,
                  n_fg: int, n_bg: int) -> tuple[Tensor, Tensor]:
    """DVR-style mask/free-space loss using stratified ray samples for both branches.

    fg branch: min SDF over n_fg stratified samples must be <= fg_margin.
               Gradient at the argmin sample, with uniform ray coverage.
    bg branch: all n_bg stratified samples must have f >= bg_margin (free-space).
    """
    zero = torch.zeros(1, device=o.device).squeeze()

    if fg.any() and n_fg > 0:
        n_fg_rays = int(fg.sum())
        dt = t_far / n_fg
        t_base = torch.linspace(0, t_far - dt, n_fg, device=o.device)
        t = t_base.unsqueeze(0) + torch.rand(n_fg_rays, n_fg, device=o.device) * dt
        pts = o[fg].unsqueeze(1) + t.unsqueeze(-1) * u[fg].unsqueeze(1)
        sdf_pts = f(pts.reshape(-1, 3)).reshape(n_fg_rays, n_fg)
        fg_loss = F.relu(sdf_pts.min(dim=1).values - fg_margin).mean()
    else:
        fg_loss = zero

    bg = ~fg
    if bg.any() and n_bg > 0:
        n_bg_rays = int(bg.sum())
        dt = t_far / n_bg
        t_base = torch.linspace(0, t_far - dt, n_bg, device=o.device)
        t = t_base.unsqueeze(0) + torch.rand(n_bg_rays, n_bg, device=o.device) * dt
        pts = o[bg].unsqueeze(1) + t.unsqueeze(-1) * u[bg].unsqueeze(1)
        bg_loss = F.relu(bg_margin - f(pts.reshape(-1, 3))).mean()
    else:
        bg_loss = zero

    return fg_loss, bg_loss


def silhouette_loss(f: FTheta, o: Tensor, u: Tensor, fg: Tensor,
                    sil_k: int, sil_t_near: float, sil_t_far: float,
                    sil_s: float) -> Tensor:
    """Volumetric silhouette BCE loss (unbiased stratified sampling)."""
    B = o.shape[0]
    dt    = (sil_t_far - sil_t_near) / sil_k
    t_sil = torch.linspace(sil_t_near, sil_t_far, sil_k, device=o.device)
    t_sil = t_sil.unsqueeze(0) + torch.rand(B, sil_k, device=o.device) * dt
    pts   = o.unsqueeze(1) + t_sil.unsqueeze(-1) * u.unsqueeze(1)
    q_k   = torch.sigmoid(-sil_s * f(pts.reshape(-1, 3))).reshape(B, sil_k)
    p_sil = 1.0 - (1.0 - q_k).prod(dim=1)
    return F.binary_cross_entropy(p_sil.clamp(1e-4, 1 - 1e-4), fg.float())


def eikonal_loss(f: FTheta, eik_pts: Tensor, n_vol: int, device: str) -> Tensor:
    """‖∇sdf‖ = 1 at trace samples + random volume points."""
    vol     = (2 * torch.rand(n_vol, 3, device=device) - 1) * 2.0
    eik_all = torch.cat([eik_pts, vol], dim=0).requires_grad_(True)
    with torch.enable_grad():
        grad = torch.autograd.grad(f(eik_all).sum(), eik_all, create_graph=True)[0]
    return (grad.norm(dim=-1) - 1.0).square().mean()


def cam_free_loss(f: FTheta, o: Tensor) -> Tensor:
    """Camera origins must lie outside the surface: relu(-f(o))."""
    return F.relu(-f(o)).mean()


def sfm_sdf_loss(f: FTheta, sfm_pts: Tensor, batch: int) -> Tensor:
    """SFM points should be on the surface: f(x_sfm)² ≈ 0."""
    idx = torch.randint(0, sfm_pts.shape[0], (batch,), device=sfm_pts.device)
    return f(sfm_pts[idx]).square().mean()


def geo_neus_sdf_loss(f: FTheta, sfm_pts: Tensor,
                      vis: Tensor | None = None, view_sel: int | None = None,
                      batch: int = 0) -> Tensor:
    """Geo-Neus's exact SDF loss on filtered COLMAP sparse points.

    Verbatim from Geo-Neus exp_runner.py (L1 toward zero, weight 1.0):
        pts2sdf  = sdf(pts_view)
        sdf_loss = F.l1_loss(pts2sdf, 0, reduction='sum') / N
    which is exactly mean(|f(x_sfm)|). Difference from sfm_sdf_loss: L1 instead
    of squared error — Geo-Neus uses L1 for robustness to the residual outliers
    that survive COLMAP filtering.

    Per-view mode (Geo-Neus's gen_pts_view): pass `vis` — the (V, P) boolean
    visibility matrix from data.colmap_visibility_matrix — and `view_sel`, the
    view index for this step (Geo-Neus cycles `iter % n_images`). Only the
    points visible in that one view are supervised, exactly like indexing
    view_id.npy each iteration.

    Global mode (fallback): omit vis/view_sel; optionally subsample `batch`
    points per step. 0 = use all points.

    sfm_pts : (P, 3) — COLMAP points in the SAME normalized world space as
                       f's input (already visibility/consistency-filtered).
                       Row order must match the columns of `vis`.
    """
    if vis is not None and view_sel is not None:
        seen = vis[view_sel]                       # (P,) points visible in this view
        if not bool(seen.any()):
            return sfm_pts.new_zeros(())           # empty view → no penalty this step
        pts = sfm_pts[seen]
    else:
        pts = sfm_pts
        if batch > 0 and sfm_pts.shape[0] > batch:
            idx = torch.randint(0, sfm_pts.shape[0], (batch,), device=sfm_pts.device)
            pts = sfm_pts[idx]
    return f(pts).abs().mean()


def free_space_loss(f: FTheta, sfm_origins: Tensor, sfm_targets: Tensor,
                    n_sfm_pairs: int, batch: int, n_free: int) -> Tensor:
    """Points along camera→SFM rays (before the surface) must satisfy f > 0."""
    idx    = torch.randint(0, n_sfm_pairs, (batch,), device=sfm_origins.device)
    o_fs   = sfm_origins[idx]
    x_fs   = sfm_targets[idx]
    t_fs   = (1.0 - torch.rand(batch, n_free, 1, device=o_fs.device).pow(3.0)).clamp(max=0.98)
    pts_fs = o_fs.unsqueeze(1) + t_fs * (x_fs - o_fs).unsqueeze(1)
    return F.relu(-f(pts_fs.reshape(-1, 3))).mean()


def sfm_behind_loss(f: FTheta, sfm_origins: Tensor, sfm_targets: Tensor,
                    n_sfm_pairs: int, batch: int, eps: float) -> Tensor:
    """A small step behind each SFM point along its camera ray should be inside."""
    idx = torch.randint(0, n_sfm_pairs, (batch,), device=sfm_origins.device)
    o = sfm_origins[idx]
    x = sfm_targets[idx]
    d = F.normalize(x - o, dim=-1)
    return F.relu(f(x + eps * d)).mean()


def surface_loss(
    f: FTheta, o: Tensor, u: Tensor, vi: Tensor,
    c2w_all: Tensor, mvs_depth_flat: Tensor, mvs_valid_flat: Tensor,
    idx: Tensor,
) -> Tensor:
    """|f(x*)| = 0 at back-projected depth-prior surface points. No hit required."""
    mvs_d = mvs_depth_flat[idx]
    valid = mvs_valid_flat[idx]
    if not valid.any():
        return torch.zeros(1, device=o.device).squeeze()
    z_cams    = c2w_all[vi, :3, 2]
    cos_theta = (u * z_cams).sum(-1).abs().clamp(min=1e-6)
    t_target  = mvs_d / cos_theta
    x_star    = o + t_target.unsqueeze(-1) * u
    return f(x_star[valid]).abs().mean()


def mvs_depth_loss(
    x_theta: Tensor, o: Tensor, u: Tensor, vi: Tensor, hit: Tensor,
    c2w_all: Tensor, mvs_depth_flat: Tensor, mvs_valid_flat: Tensor,
    idx: Tensor, step: int,
) -> Tensor:
    """Smooth-L1 alignment between sphere-traced depth and MVS depth prior."""
    mvs_d = mvs_depth_flat[idx]
    # Only supervise rays that hit AND have a valid depth prior.
    # Without the hit gate, x_theta for misses is ~t_far, producing spurious gradients.
    mvs_v = mvs_valid_flat[idx] & hit
    if not mvs_v.any():
        return torch.zeros(1, device=o.device).squeeze()
    z_cams    = c2w_all[vi, :3, 2]
    cos_theta = (u * z_cams).sum(-1).abs().clamp(min=1e-6)
    t_target  = mvs_d / cos_theta
    t_pred    = ((x_theta - o) * u).sum(-1)
    loss = F.smooth_l1_loss(t_pred[mvs_v], t_target[mvs_v])
    if step % 50 == 0:
        with torch.no_grad():
            err = (t_pred[mvs_v] - t_target[mvs_v]).abs()
            print(f"  mvs_depth: valid={mvs_v.sum().item()}/{idx.shape[0]}"
                  f"  |Δt| mean={err.mean():.3f} p90={err.quantile(.9):.3f}")
    return loss


def mvs_sdf_loss(
    f: FTheta,
    x: Tensor,
    c2w_all: Tensor,
    w2c_all: Tensor,
    K_all: Tensor,
    mvs_depth_maps: Tensor,
    mvs_valid_maps: Tensor,
    mvs_normal_maps: Tensor,
    H: int,
    W: int,
    down: int,
    n_views: int,
    trunc: float,
    smooth: float,
    far_thresh: float,
    far_att: float,
    near_thresh: float,
    near_att: float,
    step: int,
) -> Tensor:
    """Volumetric SDF target from MVS depth, following the depth-carving loss.

    For detached 3D samples x, project into several depth maps, back-project
    D(p), approximate the signed distance with the depth normal, keep the
    closest valid MVS surface, and supervise f(x). This mirrors the paper loss
    structure: eikonal samples -> carving target -> L1/SmoothL1 with in-range
    masking and near/far attenuation.
    """
    V_all, H_d, W_d = mvs_depth_maps.shape
    if n_views > 0 and n_views < V_all:
        view_ids = torch.randperm(V_all, device=x.device)[:n_views]
    else:
        view_ids = torch.arange(V_all, device=x.device)
    V = view_ids.shape[0]

    x_det = x.detach()
    w2c = w2c_all[view_ids]
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    xc = torch.einsum("vij,nj->nvi", R, x_det) + t.unsqueeze(0)
    uv_h = torch.einsum("vij,nvj->nvi", K_all[view_ids], xc)
    uv = uv_h[..., :2] / uv_h[..., 2:3].clamp(min=1e-6)

    in_frame = (xc[..., 2] > 1e-3) & (uv[..., 0] >= 0) & (uv[..., 0] < W) \
               & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    px = (uv[..., 0] / down).long().clamp(0, W_d - 1)
    py = (uv[..., 1] / down).long().clamp(0, H_d - 1)
    vid = view_ids.view(1, V).expand(x.shape[0], V)

    depth = mvs_depth_maps[vid, py, px]
    valid = in_frame & mvs_valid_maps[vid, py, px]
    if not valid.any():
        return f(x[:1].detach()).sum() * 0.0

    K = K_all[view_ids].unsqueeze(0)
    xD_cam = torch.stack([
        (uv[..., 0] - K[..., 0, 2]) / K[..., 0, 0] * depth,
        (uv[..., 1] - K[..., 1, 2]) / K[..., 1, 1] * depth,
        depth,
    ], dim=-1)

    c2w = c2w_all[view_ids]
    xD = torch.einsum("vij,nvj->nvi", c2w[:, :3, :3], xD_cam) + c2w[:, :3, 3].unsqueeze(0)
    cam_o = c2w[:, :3, 3].unsqueeze(0)
    view_dir = F.normalize(xD - cam_o, dim=-1)

    n_cam = mvs_normal_maps[vid, py, px]
    n_world = F.normalize(torch.einsum("vij,nvj->nvi", c2w[:, :3, :3], n_cam), dim=-1)
    normal_scale = -(n_world * view_dir).sum(-1)
    valid = valid & torch.isfinite(normal_scale) & (normal_scale > 1e-3)
    if not valid.any():
        return f(x[:1].detach()).sum() * 0.0

    delta = xD - x_det.unsqueeze(1)
    sign = torch.sign((delta * view_dir).sum(-1)).clamp(min=-1.0, max=1.0)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    target_all = (sign * normal_scale * delta.norm(dim=-1)).clamp(-trunc, trunc)

    abs_target = target_all.abs().masked_fill(~valid, float("inf"))
    best_abs, best_view = abs_target.min(dim=1)
    in_range = torch.isfinite(best_abs)
    if not in_range.any():
        return f(x[:1].detach()).sum() * 0.0
    target = target_all[torch.arange(x.shape[0], device=x.device), best_view].detach()

    pred = f(x[in_range])
    target = target[in_range]
    if smooth > 0:
        loss = F.smooth_l1_loss(pred / smooth, target / smooth, reduction="none") * smooth
    else:
        loss = F.l1_loss(pred, target, reduction="none")

    far_weight = torch.ones_like(target)
    if far_thresh > 0 and far_att != 1.0:
        far_weight = torch.where(target.abs() > far_thresh,
                                 torch.full_like(target, far_att),
                                 far_weight)
    near_weight = torch.ones_like(target)
    if near_thresh > 0 and near_att != 1.0:
        near_weight = torch.where(target.abs() < near_thresh,
                                  torch.full_like(target, near_att),
                                  near_weight)
    loss = (loss * far_weight * near_weight).mean()

    if step % 50 == 0:
        with torch.no_grad():
            err = (pred.detach() - target).abs()
            print(f"  mvs_sdf: valid={in_range.sum().item()}/{x.shape[0]} views={V}"
                  f"  |f-l| mean={err.mean():.3f} p90={err.quantile(.9):.3f}"
                  f"  target=[{target.min():.3f},{target.max():.3f}]")
    return loss


def mvsdf_carving_loss(
    f: FTheta,
    x: Tensor,
    w2c_all: Tensor,
    K_all: Tensor,
    depth_maps: Tensor,
    valid_maps: Tensor,
    H: int, W: int, down: int,
    out_thresh_perc: float,
    trunc: float,
    smooth: float,
    far_thresh: float, far_att: float,
    near_thresh: float, near_att: float,
    step: int,
) -> Tensor:
    """MVSDF-style depth loss with multi-view carving consensus (Zhang et al. 2021).

    For each 3D sample point, projects into all depth maps and votes:
      - inside   (point_depth > surface_depth, behind surface): SDF target < 0
      - outside  (point_depth < surface_depth, in front):       SDF target > 0

    Low-confidence / invalid depth observations are excluded from the vote.
    scene_outside = n_outside / n_valid > out_thresh_perc

    SDF target = -(point_depth - surface_depth), clamped to ±trunc.
    Supervision is only applied where at least one view sees the point.
    """
    V, H_d, W_d = depth_maps.shape
    N = x.shape[0]
    BIG = 1e6

    x_det = x.detach()

    total_in_range  = torch.zeros(N, device=x.device)
    total_valid     = torch.zeros(N, device=x.device)
    total_inside    = torch.zeros(N, device=x.device)
    best_inside_d   = torch.full((N,), BIG,  device=x.device)
    best_outside_d  = torch.full((N,), -BIG, device=x.device)

    for v in range(V):
        R = w2c_all[v, :3, :3]; t = w2c_all[v, :3, 3]
        xc = x_det @ R.T + t                                         # (N, 3)
        point_depth = xc[:, 2]                                       # (N,)

        xp  = xc @ K_all[v].T                                        # (N, 3)
        uv  = xp[:, :2] / xp[:, 2:3].clamp(min=1e-6)                # (N, 2) full-res
        uv_d = uv / down                                              # (N, 2) depth-map res

        # normalize to [-1,1] for grid_sample (align_corners=False)
        u_n = uv_d[:, 0] / W_d * 2 - 1
        v_n = uv_d[:, 1] / H_d * 2 - 1
        grid = torch.stack([u_n, v_n], dim=1).view(1, N, 1, 2)

        in_range = (xc[:, 2] > 0) & (u_n >= -1) & (u_n <= 1) & (v_n >= -1) & (v_n <= 1)

        gathered = F.grid_sample(
            depth_maps[v].unsqueeze(0).unsqueeze(0), grid,
            mode='nearest', padding_mode='zeros', align_corners=False,
        ).view(N)
        g_valid = F.grid_sample(
            valid_maps[v].float().unsqueeze(0).unsqueeze(0), grid,
            mode='nearest', padding_mode='zeros', align_corners=False,
        ).view(N) > 0.5

        valid   = (gathered > 0) & in_range & g_valid
        inside  = (point_depth > gathered * 0.99) & valid
        outside = valid & ~inside
        dist    = point_depth - gathered                              # + = inside, − = outside

        total_in_range += in_range.float()
        total_valid    += valid.float()
        total_inside   += inside.float()

        # keep closest-to-surface distance per vote direction (RunningTopK k=1)
        best_inside_d  = torch.where(inside  & (dist < best_inside_d),  dist, best_inside_d)
        best_outside_d = torch.where(outside & (dist > best_outside_d), dist, best_outside_d)

    # Vote only over valid depth observations. Invalid / filtered pixels were
    # already excluded from distance computation and should not push the point
    # toward outside.
    outside_perc   = (total_valid - total_inside) / (total_valid + 1e-9)
    scene_valid    = total_valid > 0
    scene_outside  = (outside_perc > out_thresh_perc) & scene_valid
    scene_inside   = scene_valid & ~scene_outside

    # signed depth diff: positive inside, negative outside → negate for SDF target
    # Clamp BIG sentinel values to ±trunc before computing target
    safe_inside_d  = best_inside_d.clamp(max=trunc)
    safe_outside_d = best_outside_d.clamp(min=-trunc)
    ave_dist = safe_inside_d * scene_inside.float() + safe_outside_d * scene_outside.float()
    target   = (-ave_dist).clamp(-trunc, trunc).detach()

    # Only supervise where at least one view has a valid depth measurement.
    # Points in frustum but with zero valid depth (sparse MVS coverage) would
    # otherwise get best_inside_d=BIG → target=-trunc for all of them, which
    # collapses the SDF to constant -trunc everywhere.
    in_range_mask = scene_valid
    if not in_range_mask.any():
        return f(x[:1].detach()).sum() * 0.0

    pred   = f(x[in_range_mask])
    tgt    = target[in_range_mask]

    if smooth > 0:
        loss = F.smooth_l1_loss(pred / smooth, tgt / smooth, reduction='none') * smooth
    else:
        loss = F.l1_loss(pred, tgt, reduction='none')

    far_w  = torch.where(tgt.abs() > far_thresh,  torch.full_like(tgt, far_att),  torch.ones_like(tgt))
    near_w = torch.where(tgt.abs() < near_thresh, torch.full_like(tgt, near_att), torch.ones_like(tgt))
    loss   = (loss * far_w * near_w).mean()

    if step % 50 == 0:
        with torch.no_grad():
            err = (pred.detach() - tgt).abs()
            near_mask = tgt.abs() < near_thresh
            far_mask  = tgt.abs() > far_thresh
            print(f"  mvsdf_carv: pts={in_range_mask.sum()}/{N} "
                  f"inside={scene_inside.sum()} outside={scene_outside.sum()} "
                  f"|f-t| mean={err.mean():.3f} p90={err.quantile(.9):.3f} "
                  f"tgt=[{tgt.min():.3f},{tgt.max():.3f}] "
                  f"near_frac={near_mask.float().mean():.2f} far_frac={far_mask.float().mean():.2f}")
    return loss


def behind_hit_loss(f: FTheta, x_theta: Tensor, hit: Tensor,
                    u: Tensor, eps: float) -> Tensor:
    """Points just behind the surface along the ray must have f < 0."""
    if not hit.any():
        return torch.zeros(1, device=x_theta.device).squeeze()
    return F.relu(f(x_theta[hit].detach() + eps * u[hit])).mean()


# ---------- soft-argmin photo-coherence (Section 2 of method.pdf) ----------

def soft_argmin_photo_loss(
    f: FTheta,
    x_theta: Tensor, hit: Tensor, o: Tensor, d: Tensor, vi: Tensor,
    alt_nn: Tensor, origins_all: Tensor,
    images: Tensor, K_all: Tensor, w2c_all: Tensor,
    H: int, W: int,
    n_cands: int,
    tau: float,
    t_near: float,
    t_far: float,
    use_bg: bool = False,
    bg_color: float = 1.0,
    trace_cfg=None,
    prof=None,
) -> tuple[Tensor, dict]:
    """Soft-argmin photometric loss (method PDF §2, Eq. 5).

    For every hit ray, samples n_cands stratified depth candidates, projects
    each into all alt views with sphere-trace occlusion (same as photo_loss),
    computes the visibility-weighted photometric cost (Eq. 2), forms a Boltzmann
    distribution (Eq. 4), and returns the expected squared distance from the
    traced depth to the candidates (Eq. 5):

        loss_pull = E_p[(t_l − t_pred)²] / (t_far − t_near)²

    Occlusion: for each candidate x_l, traces from each alt camera origin toward
    x_l and only counts the alt view as visible if the trace hits near x_l
    (not occluded by intervening geometry).  This is the key difference from the
    previous frustum-only check — it prevents candidates inside the object from
    getting spuriously low cost.

    Gradient flows through t_pred = (x_theta − o)·d → x_theta → IDR correction → θ.
    Only hit rays are active: miss rays have a detached x_theta in IDR mode.

    Returns (loss_pull, stats_dict).
    """
    B     = o.shape[0]
    L     = n_cands
    n_alt = alt_nn.shape[1]
    alt   = alt_nn[vi]     # (B, n_alt)
    zero  = o.new_zeros(1).squeeze()

    # Cost computation restricted to hit rays (cheaper: BL = B_hit×L instead of B×L).
    # The loss backward however uses the full-batch x_theta so that only one level of
    # advanced indexing appears in the autograd graph (avoids NaN from nested indexing).
    hit_idx = hit.nonzero(as_tuple=True)[0]   # (B_hit,)
    if hit_idx.numel() == 0:
        return zero, dict(sa_t_err=0.0, sa_cost_min=0.0, sa_signal_frac=0.0, sa_tau=tau)

    o_h   = o[hit_idx];   d_h   = d[hit_idx]
    vi_h  = vi[hit_idx];  alt_h = alt[hit_idx]
    B_h   = hit_idx.numel()
    BL    = B_h * L

    # --- Source pixel UV from ray direction ---
    R_self = w2c_all[vi_h, :3, :3]
    d_cam  = torch.einsum("bij,bj->bi", R_self, d_h)
    dz     = d_cam[:, 2].clamp(min=1e-6)
    uv_src = torch.stack([
        K_all[vi_h, 0, 0] * d_cam[:, 0] / dz + K_all[vi_h, 0, 2],
        K_all[vi_h, 1, 1] * d_cam[:, 1] / dz + K_all[vi_h, 1, 2],
    ], dim=-1)

    with torch.no_grad():
        c_src = bilinear_sample(images, vi_h, uv_src, H, W).float()   # (B_hit, 3)

    # --- Stratified candidate depths ---
    dt      = (t_far - t_near) / L
    t_base  = torch.linspace(t_near + dt / 2, t_far - dt / 2, L, device=o.device)
    jitter  = (torch.rand(B_h, L, device=o.device) - 0.5) * dt
    t_cands = (t_base.unsqueeze(0) + jitter).clamp(t_near, t_far)     # (B_hit, L)

    X_flat = (o_h.unsqueeze(1) + t_cands.unsqueeze(-1) * d_h.unsqueeze(1)).reshape(BL, 3)

    # --- Visibility-weighted photometric cost with sphere-trace occlusion ---
    cost_sum = o.new_zeros(BL)
    vis_sum  = o.new_zeros(BL)

    with torch.no_grad():
        for k in range(n_alt):
            ak    = alt_h[:, k]
            ak_BL = ak.unsqueeze(1).expand(B_h, L).reshape(BL)

            R_k  = w2c_all[ak_BL, :3, :3]
            tv_k = w2c_all[ak_BL, :3,  3]
            xc   = torch.einsum("bij,bj->bi", R_k, X_flat) + tv_k
            uvh  = torch.einsum("bij,bj->bi", K_all[ak_BL], xc)
            uv   = (uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)).nan_to_num(0.0)
            in_frustum = ((xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                          & (uv[:, 1] >= 0) & (uv[:, 1] < H))

            op_k   = origins_all[ak_BL]
            dir_k  = X_flat - op_k
            dist_k = dir_k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dp_k   = dir_k / dist_k
            dist_k = dist_k.squeeze(-1)
            _sa_cm = prof.timed("occ_trace") if prof is not None else contextlib.nullcontext()
            _sa_kw = {"cfg": trace_cfg} if trace_cfg is not None else {}
            with _sa_cm:
                _, tp_k, hitp_k = trace_nograd(f, op_k, dp_k, **_sa_kw)
            not_occl = hitp_k & (dist_k <= tp_k + 0.1)

            nu = (in_frustum & not_occl).float()

            uv_safe = torch.stack([uv[:, 0].clamp(0, W - 1),
                                   uv[:, 1].clamp(0, H - 1)], dim=-1)
            c_alt    = bilinear_sample(images, ak_BL, uv_safe, H, W).float()
            c_src_BL = c_src.unsqueeze(1).expand(B_h, L, 3).reshape(BL, 3)
            cost_sum += nu * (c_alt - c_src_BL).pow(2).sum(-1)
            vis_sum  += nu

    vis_BL = vis_sum.reshape(B_h, L)
    costs  = (cost_sum / vis_sum.clamp(min=1e-6)).reshape(B_h, L)
    costs  = costs.masked_fill(vis_BL == 0, float("inf"))

    # --- Optional background escape candidate ---
    if use_bg:
        with torch.no_grad():
            cost_bg = (o.new_full((B_h, 3), bg_color) - c_src).pow(2).sum(-1, keepdim=True)
        costs_all_h = torch.cat([costs, cost_bg],                  dim=1)
        t_all_h     = torch.cat([t_cands, o.new_full((B_h, 1), t_far)], dim=1)
    else:
        costs_all_h = costs
        t_all_h     = t_cands

    # Boltzmann weights (B_hit, L[+1]) — detached, no gradient through p
    p_h   = torch.softmax(-costs_all_h / max(tau, 1e-8), dim=1).detach()
    t_sa  = (p_h * t_all_h).sum(dim=1)                                # (B_hit,)

    has_signal_h = (vis_BL > 0).any(dim=1)                            # (B_hit,)

    # --- Expand back to full batch for the backward pass ---
    # Using full-batch x_theta avoids nested advanced-indexing in autograd (NaN source).
    # p and t_all are zero-filled for miss rays — they don't contribute to the loss
    # because active = has_signal_full & hit gates the mean.
    Lp = t_all_h.shape[1]
    p_full      = o.new_zeros(B, Lp)
    t_all_full  = o.new_zeros(B, Lp)
    has_sig_full = torch.zeros(B, dtype=torch.bool, device=o.device)
    p_full[hit_idx]       = p_h
    t_all_full[hit_idx]   = t_all_h
    has_sig_full[hit_idx] = has_signal_h

    # t_pred uses full-batch x_theta — single level of indexing in backward
    t_pred  = ((x_theta - o) * d).sum(-1)                             # (B,) with grad
    sq_dist = (p_full * (t_all_full - t_pred.unsqueeze(1)).pow(2)).sum(dim=1)  # (B,)
    active  = has_sig_full & hit
    norm    = (t_far - t_near) ** 2
    loss_pull = sq_dist[active].mean() / norm if active.any() else zero

    with torch.no_grad():
        t_pred_h = t_pred[hit_idx].detach()
        if has_signal_h.any():
            _t_err    = float((t_sa[has_signal_h] - t_pred_h[has_signal_h]).abs().mean())
            _cost_min = float(costs_all_h[has_signal_h].min(dim=1).values.mean())
        else:
            _t_err = _cost_min = 0.0
        stats = dict(
            sa_t_err=_t_err,
            sa_cost_min=_cost_min,
            sa_signal_frac=float(has_signal_h.float().mean()),
            sa_tau=tau,
        )

    return loss_pull, stats
