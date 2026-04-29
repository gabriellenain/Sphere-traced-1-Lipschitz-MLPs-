"""Loss functions for 1-Lip sphere-tracing training."""
from __future__ import annotations

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
) -> Tensor:
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
        return torch.empty(0, device=images.device)

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

    # 7. Per-channel ZNCC averaged over RGB
    pa = pa - pa.mean(dim=1, keepdim=True)           # (B', P*P, 3)
    pb = pb - pb.mean(dim=1, keepdim=True)
    std_a = pa.norm(dim=1)                           # (B', 3)
    std_b = pb.norm(dim=1)
    textured = (std_a > 1e-4).all(1) & (std_b > 1e-4).all(1)
    if not textured.any():
        return torch.empty(0, device=images.device)
    pa = pa[textured] / std_a[textured].unsqueeze(1).clamp(min=1e-6)  # (B'', P*P, 3)
    pb = pb[textured] / std_b[textured].unsqueeze(1).clamp(min=1e-6)
    return 1.0 - (pa * pb).sum(dim=1).mean(dim=1).clamp(-1.0, 1.0)   # mean over channels


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
    step: int,
) -> tuple[Tensor, dict]:
    """Multi-view photoconsistency loss with occlusion test.

    Returns (loss, debug_stats_dict).
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

    loss_terms: list[Tensor] = []
    l1_vals:   list[Tensor] = []
    feat_vals: list[Tensor] = []
    ncc_vals:  list[Tensor] = []
    n_total = n_in_frame = n_not_occl = n_cos_ok = n_mask = 0

    # Batch all occlusion traces across alt views — (B*n_alt, 3) instead of n_alt serial traces
    alt_flat = alt.reshape(-1)                                        # (B*n_alt,)
    op_all   = origins_all[alt_flat]                                  # (B*n_alt, 3)
    dir_all  = x_theta.unsqueeze(1).expand(B, n_alt, 3).reshape(-1, 3) - op_all
    dist_all = dir_all.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dp_all   = dir_all / dist_all
    with torch.no_grad():
        _, tp_all, hitp_all = trace_nograd(f, op_all, dp_all)
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

        depth_ok = dist <= tp + 1e-1
        not_occl = hitp & depth_ok

        if step % 50 == 0 and k == 0:
            delta = tp - dist
            print(f"  occ alt0: hitp={hitp.float().mean():.2f}  "
                  f"depth_ok={depth_ok.float().mean():.2f}  "
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
                ncc = pmvs_ncc_loss(
                    images,
                    x_theta[mask], n[mask].detach(),
                    vi[mask], ak[mask],
                    K_all, w2c_all,
                    H, W, ncc_patch, ncc_half_pix,
                    sample_mode, gaussian_sigma, gaussian_radius,
                )
                if ncc.numel() > 0:
                    ncc_term = ncc.mean()
                    loss_terms.append(w_ncc * ncc_term)
                    ncc_vals.append(ncc_term.detach())

    if loss_terms:
        loss = torch.stack(loss_terms).mean()
    else:
        loss = f(x_theta.detach()[:1]).sum() * 0.0
    stats = dict(n_mask=n_mask, n_in_frame=n_in_frame,
                 n_not_occl=n_not_occl, n_cos_ok=n_cos_ok, n_total=n_total,
                 l1=torch.stack(l1_vals).mean().item() if l1_vals else 0.0,
                 feature=torch.stack(feat_vals).mean().item() if feat_vals else 0.0,
                 ncc=torch.stack(ncc_vals).mean().item() if ncc_vals else 0.0)
    return loss, stats


# ---------- geometry / regularisation ----------

def mask_loss_min_sdf(sdf_min: Tensor, fg: Tensor, alpha: float,
                      fg_offset: float = 0.1) -> Tensor:
    """Mask loss via minimum SDF along traced rays (Yariv et al. IDR / NeuS).

    S_{p,α} = σ(−α · (min_t f(o + td) − fg_offset · O_p))

    fg_offset shifts sdf_min for foreground rays so that a hit (sdf_min≈0)
    maps to σ(α·offset) > 0.5 instead of σ(0)=0.5, giving a stronger
    gradient toward S=1 without needing the intersection point.
    Background rays are unaffected (offset=0 for bg).

    Works for all rays. Requires sdf_min from trace_unrolled.
    """
    sdf_adj = sdf_min - fg.float() * fg_offset
    S = torch.sigmoid(-alpha * sdf_adj)
    return F.binary_cross_entropy(S.clamp(1e-6, 1 - 1e-6), fg.float()) / alpha


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


def free_space_loss(f: FTheta, sfm_origins: Tensor, sfm_targets: Tensor,
                    n_sfm_pairs: int, batch: int, n_free: int) -> Tensor:
    """Points along camera→SFM rays (before the surface) must satisfy f > 0."""
    idx    = torch.randint(0, n_sfm_pairs, (batch,), device=sfm_origins.device)
    o_fs   = sfm_origins[idx]
    x_fs   = sfm_targets[idx]
    t_fs   = (1.0 - torch.rand(batch, n_free, 1, device=o_fs.device).pow(3.0)).clamp(max=0.98)
    pts_fs = o_fs.unsqueeze(1) + t_fs * (x_fs - o_fs).unsqueeze(1)
    return F.relu(-f(pts_fs.reshape(-1, 3))).mean()


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
      - invalid  (in frustum but no depth measurement):         counts as 0.5 outside
                 — matches carving_t (paper default). This applies a weak free-space
                 prior to pixels where depth estimation failed.

    scene_outside = (n_outside + 0.5 * n_invalid) / n_in_range > out_thresh_perc

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

    # vote — invalid pixels (in frustum, no depth) count as 0.5 outside (carving_t paper default)
    total_invalid  = total_in_range - total_valid
    outside_perc   = (total_valid - total_inside + total_invalid * 0.5) / (total_in_range + 1e-9)
    scene_in_range = total_in_range > 0
    scene_valid    = total_valid > 0
    scene_outside  = (outside_perc > out_thresh_perc) & scene_in_range
    scene_inside   = scene_in_range & ~scene_outside

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
            print(f"  mvsdf_carv: pts={in_range_mask.sum()}/{N} "
                  f"inside={scene_inside.sum()} outside={scene_outside.sum()} "
                  f"|f-t| mean={err.mean():.3f} p90={err.quantile(.9):.3f} "
                  f"tgt=[{tgt.min():.3f},{tgt.max():.3f}]")
    return loss


def behind_hit_loss(f: FTheta, x_theta: Tensor, hit: Tensor,
                    u: Tensor, eps: float) -> Tensor:
    """Points just behind the surface along the ray must have f < 0."""
    if not hit.any():
        return torch.zeros(1, device=x_theta.device).squeeze()
    return F.relu(f(x_theta[hit].detach() + eps * u[hit])).mean()
