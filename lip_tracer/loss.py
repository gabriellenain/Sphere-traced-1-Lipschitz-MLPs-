"""Loss functions for 1-Lip sphere-tracing training."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import FTheta
from .sphere_tracing import trace_nograd


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
    step_3d = ((2.0 * half_pix / max(P - 1, 1)) * z_ref / f_x).detach()        # (B,) — patch scale fixed, gradient only through center

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
        ph  = (K_all[vi].unsqueeze(1) @ xc.unsqueeze(-1)).squeeze(-1)               # (B, P*P, 3)
        uv  = ph[:, :, :2] / ph[:, :, 2:3].clamp(min=1e-6)                         # (B, P*P, 2)
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

    if pw is not None:
        _wsum = pw.sum()
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
    w_ncc_normal: float = 0.0,
    ncc_topk: int = 0,
    ncc_normal_patch: int = -1,
    ncc_normal_half_pix: float = -1.0,
    ncc_color: str = "gray",
    ncc_grad_alpha: float = 0.0,
    ncc_patch_wsigma: float = 0.0,
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
    _trace_kw = {"cfg": trace_cfg} if trace_cfg is not None else {}
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
        uv_h = torch.einsum("bij,bj->bi", Kp, xc)
        uv   = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
        in_frame = (xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) \
                   & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        if occ_mode == "from_hit":
            # Trace started at hit-point toward alt cam: occluded iff it hits before reaching cam.
            not_occl = ~hitp | (tp > dist - 0.1)
        else:
            depth_ok = dist <= tp + 1e-1
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
                zncc, keep, n_valid, *zf = pmvs_ncc_loss(
                    images,
                    x_theta[mask], n[mask].detach(),
                    vi[mask], ak[mask],
                    K_all, w2c_all,
                    H, W, ncc_patch, ncc_half_pix,
                    sample_mode, gaussian_sigma, gaussian_radius,
                    ncc_min,
                    return_full=use_topk,
                    ncc_color=ncc_color,
                    ncc_grad_alpha=ncc_grad_alpha,
                    patch_wsigma=ncc_patch_wsigma,
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
                if use_topk:
                    zpos_cols.append(_scatter_col(
                        zf[0], mask.nonzero(as_tuple=True)[0]))
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
                 n_mask_bg=n_mask_bg,
                 l1=torch.stack(l1_vals).mean().item() if l1_vals else 0.0,
                 feature=torch.stack(feat_vals).mean().item() if feat_vals else 0.0,
                 ncc=torch.stack(ncc_vals).mean().item() if ncc_vals else 0.0,
                 ncc_normal=torch.stack(ncc_n_vals).mean().item() if ncc_n_vals else 0.0,
                 ncc_weighted=(w_ncc * torch.stack(ncc_vals).mean().item()) if ncc_vals else 0.0,
                 ncc_normal_weighted=(w_ncc_normal * torch.stack(ncc_n_vals).mean().item()) if ncc_n_vals else 0.0,
                 ncc_zncc=torch.stack(ncc_zncc_vals).mean().item() if ncc_zncc_vals else 0.0,
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


