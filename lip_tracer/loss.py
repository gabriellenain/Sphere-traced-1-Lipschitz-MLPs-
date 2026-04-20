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
    Returns  (B, 3)       — no (B, H, W, 3) intermediate
    """
    u = uv[:, 0].clamp(0, W - 1);  v = uv[:, 1].clamp(0, H - 1)
    u0 = u.long();  u1 = (u0 + 1).clamp(max=W - 1)
    v0 = v.long();  v1 = (v0 + 1).clamp(max=H - 1)
    wu = (u - u0.float()).unsqueeze(-1)   # (B, 1)
    wv = (v - v0.float()).unsqueeze(-1)   # (B, 1)
    return (images[vi, v0, u0] * (1 - wu) * (1 - wv)
            + images[vi, v0, u1] * wu       * (1 - wv)
            + images[vi, v1, u0] * (1 - wu) * wv
            + images[vi, v1, u1] * wu       * wv)


def ncc_loss(imgs: Tensor, uv_a: Tensor, uv_b: Tensor,
             H: int, W: int, patch: int = 7) -> Tensor:
    """NCC loss between patches centred at uv_a and uv_b.

    imgs : (B, 3, H, W)
    uv_a : (B, 2)  pixel coords for patch A
    uv_b : (B, 2)  pixel coords for patch B
    Returns (B,) loss ∈ [0, 2], 0 = perfect correlation.
    """
    p    = patch
    half = (p - 1) / 2.0
    oy = torch.linspace(-half, half, p, device=imgs.device) / ((H - 1) / 2)
    ox = torch.linspace(-half, half, p, device=imgs.device) / ((W - 1) / 2)
    gy, gx   = torch.meshgrid(oy, ox, indexing="ij")
    grid_off = torch.stack([gx, gy], dim=-1)

    def extract(uv: Tensor) -> Tensor:
        cx = uv[:, 0] / ((W - 1) / 2) - 1.0
        cy = uv[:, 1] / ((H - 1) / 2) - 1.0
        g  = torch.stack([cx, cy], dim=-1)[:, None, None, :] + grid_off[None]
        return F.grid_sample(imgs, g, mode="bilinear",
                             align_corners=True, padding_mode="border")

    pa = extract(uv_a).flatten(1)
    pb = extract(uv_b).flatten(1)
    pa = pa - pa.mean(dim=1, keepdim=True)
    pb = pb - pb.mean(dim=1, keepdim=True)
    ncc = (pa / pa.norm(dim=1, keepdim=True).clamp(1e-6)
           * pb / pb.norm(dim=1, keepdim=True).clamp(1e-6)).sum(dim=1)
    return 1.0 - ncc


def photo_loss(
    f: FTheta,
    x_theta: Tensor, hit: Tensor, n: Tensor,
    vi: Tensor, alt_nn: Tensor, origins_all: Tensor,
    images: Tensor, K_all: Tensor, w2c_all: Tensor,
    masks: Tensor | None, fg_self: Tensor,
    H: int, W: int,
    uv_self: Tensor,
    n_alt: int, cos_thresh: float,
    w_photo: float, w_ncc: float, ncc_patch: int,
    step: int,
) -> tuple[Tensor, dict]:
    """Multi-view photoconsistency loss with occlusion test.

    Returns (loss, debug_stats_dict).
    """
    B = vi.shape[0]
    alt = alt_nn[vi]  # (B, n_alt)

    # primary-camera colour — (B, 3) only, no (B, H, W, 3) intermediate
    c_self = bilinear_sample(images, vi, uv_self, H, W)
    # only needed for NCC patch sampling (w_ncc > 0)
    imgs_self = images[vi].permute(0, 3, 1, 2) if w_ncc > 0 else None

    loss_terms: list[Tensor] = []
    n_total = n_in_frame = n_not_occl = n_cos_ok = n_mask = 0

    for k in range(n_alt):
        ak      = alt[:, k]
        op      = origins_all[ak]
        dir_to_x = x_theta - op
        dist    = dir_to_x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dp      = dir_to_x / dist

        Kp   = K_all[ak];  w2cp = w2c_all[ak]
        xc   = torch.einsum("bij,bj->bi", w2cp[:, :3, :3], x_theta) + w2cp[:, :3, 3]
        uv_h = torch.einsum("bij,bj->bi", Kp, xc)
        uv   = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)
        in_frame = (xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) \
                   & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        with torch.no_grad():
            _, tp, hitp = trace_nograd(f, op, dp)
            depth_ok  = dist.squeeze(-1) <= tp + 1e-1
            not_occl  = hitp & depth_ok

        if step % 50 == 0 and k == 0:
            delta = tp - dist.squeeze(-1)
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
            gt_alt = bilinear_sample(images, ak, uv, H, W)          # (B, 3) — no (B,H,W,3)
            if w_photo > 0:
                loss_terms.append(w_photo * (c_self - gt_alt).abs().sum(-1)[mask].mean())
            if w_ncc > 0:
                # NCC needs full patches — only reached when w_ncc > 0
                imgs_alt = images[ak].permute(0, 3, 1, 2)
                ncc_ab = ncc_loss(imgs_self[mask], uv_self[mask], uv[mask], H, W, ncc_patch)
                ncc_ba = ncc_loss(imgs_alt[mask],  uv[mask], uv_self[mask], H, W, ncc_patch)
                loss_terms.append(w_ncc * (ncc_ab + ncc_ba).mean() * 0.5)

    loss = torch.stack(loss_terms).mean() if loss_terms else torch.zeros(1, device=vi.device).squeeze()
    stats = dict(n_mask=n_mask, n_in_frame=n_in_frame,
                 n_not_occl=n_not_occl, n_cos_ok=n_cos_ok, n_total=n_total)
    return loss, stats


# ---------- geometry / regularisation ----------

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
    """‖∇f‖ = 1 at trace samples + random volume points."""
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


def mvs_depth_loss(
    x_theta: Tensor, o: Tensor, u: Tensor, vi: Tensor,
    c2w_all: Tensor, mvs_depth_flat: Tensor, mvs_valid_flat: Tensor,
    idx: Tensor, step: int,
) -> Tensor:
    """Smooth-L1 alignment between sphere-traced depth and MVS depth prior."""
    mvs_d = mvs_depth_flat[idx]
    mvs_v = mvs_valid_flat[idx]
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


def behind_hit_loss(f: FTheta, x_theta: Tensor, hit: Tensor,
                    u: Tensor, eps: float) -> Tensor:
    """Points just behind the surface along the ray must have f < 0."""
    if not hit.any():
        return torch.zeros(1, device=x_theta.device).squeeze()
    return F.relu(f(x_theta[hit].detach() + eps * u[hit])).mean()
