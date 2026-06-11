"""Block-coordinate photometric bundle adjustment for sphere-traced SDFs.

Extends the photometric objective from E(θ) to E(θ, φ) by making the per-camera
extrinsics φ = {φ_i} trainable, and refines (θ, φ) by *block coordinate descent*
(block-diagonal alternation) starting from a converged checkpoint:

    repeat for `cycles`:
        φ-block:  argmin_φ  E(θ, φ)            (θ frozen, block_phi steps)
        θ-block:  argmin_θ  E(θ, φ) + R(θ)     (φ frozen, block_theta steps)

with

    E(θ, φ) = Σ_{i,j,k} L( c[i,j], c[k, π_{k,φ_k}( x_{θ,φ_i}(i,j) )] ).

No mesh is ever extracted: x_{θ,φ_i}(i,j) is always the sphere-traced
intersection of the ray from pixel j in camera i with f_θ = 0.

  * the SOURCE pose φ_i builds the tracing ray (rays_from_pixels → trace), so
    x_{θ,φ_i} moves with the source camera;
  * the TARGET pose φ_k is used for reprojection π_{k,φ_k} and the visibility
    (occlusion) test — both live inside `photo_loss`, fed the live w2c/origins
    recomputed from φ each step.

Pose gradients flow through the IDR differentiable intersection with the camera
kept live (`idr_intersection`):

    x = p − (f(p)/(n·d))·d,   p = o + t₀·d,   t₀ = depth.detach(), n detached.

The depth t₀ is solved by the model's normal trace mode (`trace_idr` by default —
mostly no-grad and cheap); we then re-derive x with (o, d, θ) live. At
convergence f(p)≈0, so x≈p geometrically, but ∂f(p)/∂(o,d)=∇f≠0: the correction's
*value* vanishes while its *gradient* w.r.t. φ does not. (The repo's `trace_idr`
detaches p, so its x_θ carries θ-gradient only — that is why we recompute the
intersection here rather than use the trace's own x_θ.) With o, d detached the
same expression reduces to `trace_idr`'s θ-only correction, so E is one and the
same objective in both blocks; only the active variable changes.

R(θ) in the θ-block is the same SDF regularisation the model was trained with
(eikonal + silhouette/mask + behind-hit, read from TrainConfig), so f stays a
valid 1-Lipschitz SDF across cycles. The φ-block optimises the bare photometric
E. Camera 0 is gauge-locked (lock_first) so BA cannot drift the scene rigidly.
Intrinsics stay fixed by default; with opt_intrinsics the per-camera K (fx, fy,
cx, cy) is refined jointly in the φ-block (see CameraParams).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace, asdict as dataclasses_asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .config import Config, TrainConfig, TraceConfig
from .loss import (photo_loss, eikonal_loss, idr_mask_loss, mask_loss_min_sdf,
                   dvr_mask_loss, behind_hit_loss)
from .sphere_tracing import trace_idr, trace_unrolled


# ───────────────────────────── φ parametrization ───────────────────────────

def rodrigues(log_rot: Tensor) -> Tensor:
    """(V, 3) so(3) tangent → (V, 3, 3) rotation matrix."""
    theta = log_rot.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (V, 1)
    k = log_rot / theta                                          # (V, 3) unit axis
    V = log_rot.shape[0]
    K = log_rot.new_zeros(V, 3, 3)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] =  k[:, 1]
    K[:, 1, 0] =  k[:, 2]; K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] =  k[:, 0]
    sin = theta.sin().unsqueeze(-1)
    cos = theta.cos().unsqueeze(-1)
    I = torch.eye(3, device=log_rot.device, dtype=log_rot.dtype).expand(V, 3, 3)
    return I + sin * K + (1 - cos) * (K @ K)


def _perturb_poses(c2w: Tensor, rot_deg: float, trans_units: float,
                   seed: int, lock_first: bool = True,
                   only_cam: int = -1) -> Tensor:
    """Apply a known SE(3) perturbation to each camera for a recovery test.

    Each camera i (skipping i=0 when lock_first, so the gauge stays fixed) is
    rotated by `rot_deg` about a random unit axis and shifted by `trans_units`
    in a random unit direction — fixed magnitude, random direction, so the
    injected error is a controlled constant per camera. Deterministic in `seed`.

    When `only_cam >= 0`, ONLY that camera is perturbed and every other camera
    stays at its calibrated GT pose — the cleanest minimal recovery test: cam 0
    (gauge-locked) plus the untouched GT cameras rigidly pin the scene gauge and
    the SDF, so the perturbed camera has a unique correct pose to recover.
    """
    g = torch.Generator().manual_seed(int(seed))
    out = c2w.clone()
    V = c2w.shape[0]
    ang = math.radians(rot_deg)
    for i in range(V):
        if lock_first and i == 0:
            continue
        if only_cam >= 0 and i != only_cam:
            continue
        axis = torch.randn(3, generator=g); axis = axis / axis.norm().clamp(min=1e-8)
        dR = rodrigues((axis * ang)[None])[0]
        out[i, :3, :3] = dR @ c2w[i, :3, :3]
        tdir = torch.randn(3, generator=g); tdir = tdir / tdir.norm().clamp(min=1e-8)
        out[i, :3, 3] = c2w[i, :3, 3] + tdir * trans_units
    return out


class CameraParams(nn.Module):
    """Learnable extrinsics delta φ (and optionally intrinsics) on a frozen base.

    Rotation is R = exp(skew(log_rot)) @ R_base — a left-multiplied so(3) delta
    that starts at the identity (log_rot = 0), so φ at step 0 reproduces the
    calibrated cameras exactly. Translation is t = t_base + dt. Camera 0 is
    gauge-locked (its delta is masked to zero) so BA cannot drift the whole
    scene rigidly.

    When `opt_intrinsics`, the per-camera pinhole K is also refined via a
    dimensionless delta dK = (s_fx, s_fy, r_cx, r_cy) per camera, starting at 0:

        fx = fx_base · exp(s_fx),   cx = cx_base + r_cx · fx_base,
        fy = fy_base · exp(s_fy),   cy = cy_base + r_cy · fy_base.

    Focal is multiplicative and principal-point offsets are scaled by the base
    focal, so all four are dimensionless and share one learning rate regardless
    of the calibration's pixel units. Intrinsics are NOT gauge-locked (they are
    observable once cam-0 extrinsics fix the scene gauge).
    """
    def __init__(self, c2w_base: Tensor, lock_first: bool = True,
                 K_base: Tensor | None = None, opt_intrinsics: bool = False):
        super().__init__()
        V = c2w_base.shape[0]
        self.register_buffer("R_base", c2w_base[:, :3, :3].contiguous().float())
        self.register_buffer("t_base", c2w_base[:, :3, 3].contiguous().float())
        self.log_rot = nn.Parameter(torch.zeros(V, 3))
        self.dt      = nn.Parameter(torch.zeros(V, 3))
        mask = torch.ones(V, 1)
        if lock_first:
            mask[0] = 0.0
        self.register_buffer("free_mask", mask)

        self.opt_intrinsics = opt_intrinsics
        if K_base is not None:
            self.register_buffer("K_base", K_base.contiguous().float())
            if opt_intrinsics:
                self.dK = nn.Parameter(torch.zeros(V, 4))   # s_fx, s_fy, r_cx, r_cy
        else:
            self.register_buffer("K_base", torch.empty(0))

    def intrinsics(self) -> Tensor:
        """Live per-camera intrinsics K (V, 3, 3) from the base + learned delta."""
        if not self.opt_intrinsics:
            return self.K_base
        K = self.K_base.clone()
        fx0, fy0 = self.K_base[:, 0, 0], self.K_base[:, 1, 1]
        K[:, 0, 0] = fx0 * self.dK[:, 0].exp()
        K[:, 1, 1] = fy0 * self.dK[:, 1].exp()
        K[:, 0, 2] = self.K_base[:, 0, 2] + self.dK[:, 2] * fx0
        K[:, 1, 2] = self.K_base[:, 1, 2] + self.dK[:, 3] * fy0
        return K

    def intrinsics_params(self) -> list[nn.Parameter]:
        """The learnable intrinsics parameters (empty when opt_intrinsics is off)."""
        return [self.dK] if self.opt_intrinsics else []

    def forward(self) -> Tensor:
        log_rot = self.log_rot * self.free_mask
        dt      = self.dt      * self.free_mask
        dR = rodrigues(log_rot)              # (V, 3, 3)
        R  = dR @ self.R_base                # (V, 3, 3)
        t  = self.t_base + dt                # (V, 3)
        V  = R.shape[0]
        c2w = R.new_zeros(V, 4, 4)
        c2w[:, :3, :3] = R
        c2w[:, :3, 3]  = t
        c2w[:,  3, 3]  = 1.0
        return c2w

    @torch.no_grad()
    def deltas(self) -> tuple[float, float]:
        """(mean rotation °, mean translation) of the current φ vs. base — for logging."""
        m = self.free_mask.squeeze(-1) > 0
        if not m.any():
            return 0.0, 0.0
        rot_deg = self.log_rot[m].norm(dim=-1).mean().item() * 180.0 / math.pi
        trans   = self.dt[m].norm(dim=-1).mean().item()
        return rot_deg, trans


def rays_from_pixels(c2w: Tensor, K: Tensor, px: Tensor, py: Tensor,
                     vi: Tensor) -> tuple[Tensor, Tensor]:
    """Rebuild (o, d) for a batch of rays from current camera params.

    c2w: (V, 4, 4)   K: (V, 3, 3)   px, py: (B,) float pixel coords   vi: (B,) long
    Returns o: (B, 3), d: (B, 3) unit-norm — both differentiable wrt c2w (hence φ).
    """
    Kb = K[vi]; cw = c2w[vi]                                # (B, 3, 3), (B, 4, 4)
    x = (px - Kb[:, 0, 2]) / Kb[:, 0, 0]
    y = (py - Kb[:, 1, 2]) / Kb[:, 1, 1]
    d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (B, 3)
    d_w   = torch.einsum("bij,bj->bi", cw[:, :3, :3], d_cam)
    d_w   = d_w / d_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    o     = cw[:, :3, 3]
    return o, d_w


# ───────────────────────────── prepared data ───────────────────────────────

@dataclass
class BAContext:
    """Everything the photometric objective + SDF regularisers need, prepared once.

    Camera *poses* are NOT stored here as live tensors — they are reconstructed
    from CameraParams every step. `c2w_base` is the frozen calibration that seeds
    φ; K_all are the fixed intrinsics.
    """
    c2w_base: Tensor            # (V, 4, 4) calibrated extrinsics — φ base
    K_all:    Tensor            # (V, 3, 3) intrinsics (fixed)
    images:   Tensor            # (V, H, W, 3) on device
    masks:    Tensor | None     # (V, H, W) bool fg masks on device, or None
    feature_maps: Tensor | None # (V, Hf, Wf, C) or None
    alt_nn:   Tensor            # (V, n_alt) nearest-neighbour target cameras
    det:      dict              # deterministic rays: px, py, vi, gt, fg (CPU tensors)
    fg_idx:   Tensor            # CPU indices of foreground rays
    bg_idx:   Tensor            # CPU indices of background rays
    n_fg:     int               # foreground rays per batch
    n_bg:     int               # background rays per batch
    H: int
    W: int
    device: str


def prepare_context(views: dict, train_cfg: TrainConfig, device: str) -> BAContext:
    """Build a BAContext from loaded views — mirrors train.py's data prep.

    Reuses make_deterministic_rays / precompute_alt_cameras so the ray grid and
    target-camera neighbourhoods match training exactly.
    """
    from .data import make_deterministic_rays, precompute_alt_cameras

    # fp16 only helps on CUDA; several fp16 ops (e.g. grid_sample) are unimplemented
    # on CPU, so keep images fp32 there to allow a CPU run of the recovery test.
    images = views["images"].to(device)
    images = images.half() if torch.device(device).type == "cuda" else images.float()
    masks  = views["masks"].to(device) if "masks" in views else None
    K_all  = views["K"].to(device).float()
    c2w_base = views["c2w"].to(device).float()
    H, W   = views["H"], views["W"]

    det = make_deterministic_rays(views, down=train_cfg.down, device=device)
    fg_idx = det["fg"].nonzero(as_tuple=True)[0]
    bg_idx = (~det["fg"]).nonzero(as_tuple=True)[0]

    # batch split mirrors train.py: bg rays only matter when a bg-sensitive loss
    # is active (mask/silhouette/behind-hit). Otherwise spend the whole batch on fg.
    bg_active = any(w > 0 for w in (train_cfg.w_mask_bg, train_cfg.w_idr_mask,
                                    train_cfg.w_sil, train_cfg.w_behind_hit))
    batch = train_cfg.bundle.batch or train_cfg.batch
    if not 0.0 <= train_cfg.fg_fraction <= 1.0:
        raise ValueError(f"fg_fraction must be in [0, 1], got {train_cfg.fg_fraction}")
    if len(bg_idx) == 0 or (not bg_active and not train_cfg.force_fg_bg_split):
        n_fg, n_bg = batch, 0
    elif len(fg_idx) == 0:
        n_fg, n_bg = 0, batch
    else:
        n_fg = int(batch * train_cfg.fg_fraction)
        n_bg = batch - n_fg

    alt_nn = precompute_alt_cameras(views, train_cfg.n_alt).to(device)
    return BAContext(c2w_base=c2w_base, K_all=K_all, images=images, masks=masks,
                     feature_maps=None, alt_nn=alt_nn, det=det,
                     fg_idx=fg_idx, bg_idx=bg_idx, n_fg=n_fg, n_bg=n_bg,
                     H=H, W=W, device=device)


def _sample_batch(ctx: BAContext) -> Tensor:
    """Draw one batch of ray indices (uniform within fg / bg strata)."""
    parts = []
    if ctx.n_fg > 0 and len(ctx.fg_idx) > 0:
        parts.append(ctx.fg_idx[torch.randint(0, len(ctx.fg_idx), (ctx.n_fg,))])
    if ctx.n_bg > 0 and len(ctx.bg_idx) > 0:
        parts.append(ctx.bg_idx[torch.randint(0, len(ctx.bg_idx), (ctx.n_bg,))])
    return torch.cat(parts)


# ─────────────────────── differentiable intersection ───────────────────────

def idr_intersection(f, o: Tensor, d: Tensor, t: Tensor, n_geo: Tensor,
                     hit: Tensor) -> Tensor:
    """IDR ray-surface intersection that keeps the camera (o, d) live.

        x = p − (f(p) / (n·d)) · d,   p = o + t₀·d,   t₀ = t.detach(), n detached.

    The repo's `trace_idr` detaches the hit point, so its x_θ carries
    θ-gradient only. Here p stays live in (o, d) — and f's params stay live in θ —
    so x carries BOTH the θ-gradient AND the exact first-order pose gradient
    ∂x/∂(o,d). At convergence f(p)≈0, so x≈p geometrically, but ∂f(p)/∂(o,d)=∇f≠0:
    the correction's *value* vanishes while its *gradient* w.r.t. φ does not. With
    o, d detached this reduces exactly to `trace_idr`'s θ-only correction, so the
    two blocks share one objective; only the live variable differs.

    `t`, `n_geo` come from a (mostly no-grad) `trace_idr`/`trace_unrolled` solve,
    so we keep that efficient depth solve and only add one extra f(p) evaluation.
    """
    t0 = t.detach()
    p  = o + t0.unsqueeze(-1) * d                          # live in o, d
    n  = n_geo.detach()
    n_dot_d = (n * d).sum(-1)
    sign    = torch.where(n_dot_d != 0, n_dot_d.sign(), torch.ones_like(n_dot_d))
    n_dot_d_safe = (sign * n_dot_d.abs().clamp(min=1e-3)).detach()
    f_p = f(p)                                             # live → ∂/∂(o, d, θ)
    correction = torch.where(hit, f_p / n_dot_d_safe, torch.zeros_like(f_p))
    return p - correction.unsqueeze(-1) * d                # bg rays: x = p


# ─────────────────────────── objective E(θ, φ) ──────────────────────────────

def photometric_objective(f, cam_params: CameraParams, idx: Tensor,
                          ctx: BAContext, train_cfg: TrainConfig,
                          trace_cfg: TraceConfig, step: int):
    """Evaluate E(θ, φ) on one ray batch and return (E, aux).

    Faithful to the formula: rays are built from the *source* pose φ_i, the
    surface point is the sphere-traced f_θ = 0 intersection (re-derived with the
    camera live via `idr_intersection` so ∂x_θ/∂φ_i is exact), and reprojection +
    visibility use the *target* pose φ_k (via the live w2c/origins passed into
    photo_loss). aux carries the trace by-products the θ-block needs for its SDF
    regularisers.
    """
    dev = ctx.device
    vi      = ctx.det["vi"][idx].to(dev)
    fg_self = ctx.det["fg"][idx].to(dev)
    px      = ctx.det["px"][idx].to(dev)
    py      = ctx.det["py"][idx].to(dev)

    # live cameras from current φ — source poses build rays, target poses reproject.
    # K_all is the live per-camera intrinsics (== ctx.K_all unless opt_intrinsics).
    c2w_all     = cam_params()                       # (V, 4, 4)
    K_all       = cam_params.intrinsics()            # (V, 3, 3)
    origins_all = c2w_all[:, :3, 3]
    w2c_all     = torch.linalg.inv(c2w_all)
    o, u = rays_from_pixels(c2w_all, K_all, px, py, vi)   # ← φ_i

    # Efficient (mostly no-grad) depth solve in the model's trace mode. We discard
    # the trace's own x_θ (it detaches o,u) and re-derive the intersection below
    # with the camera live, so pose gradients flow without unrolling the trace.
    _trace = trace_idr if trace_cfg.grad_mode == "idr" else trace_unrolled
    # The compacted trace_idr dropped the soft-min (sdf_min_beta>0) path that only
    # feeds the differentiable silhouette regulariser (mask_loss_min_sdf, w_sil>0).
    # sdf_min_beta affects *only* the sdf_min output, never t/hit/x_theta/n_raw, so
    # forcing hard-min (beta=0) leaves the geometry — and the photometric E — bit-
    # identical. Only safe when sdf_min isn't consumed: guard the θ-block on it.
    _trace_cfg = trace_cfg
    if trace_cfg.grad_mode == "idr" and trace_cfg.sdf_min_beta > 0:
        if train_cfg.w_sil > 0:
            raise NotImplementedError(
                "BA θ-block with w_sil>0 needs soft-min sdf_min, which the compacted "
                "trace_idr no longer supports; use grad_mode='backprop' (trace_unrolled) "
                "or set w_sil=0 for the BA pass")
        _trace_cfg = replace(trace_cfg, sdf_min_beta=0.0)
    _, t, hit, eik_pts, n_raw, sdf_min, hit_bg = _trace(
        f, o, u, _trace_cfg, collect_eik=train_cfg.w_eikonal > 0,
        diff_normal=train_cfg.w_ncc_normal > 0)
    # IDR intersection with live (o, u): carries ∂x_θ/∂φ_i AND ∂x_θ/∂θ.
    x_theta = idr_intersection(f, o, u, t, n_raw, hit)
    n = n_raw / n_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    hit_for_photo = hit | (hit_bg & fg_self)

    # primary-camera reprojection of the hit point (φ_i view) for the L1 term
    w2c_self  = w2c_all[vi]
    xc_self   = torch.einsum("bij,bj->bi", w2c_self[:, :3, :3], x_theta) + w2c_self[:, :3, 3]
    uv_h_self = torch.einsum("bij,bj->bi", K_all[vi], xc_self)
    uv_self   = uv_h_self[:, :2] / uv_h_self[:, 2:3].clamp(min=1e-6)

    radius = max(1, int(math.ceil(2.0 * train_cfg.gaussian_sigma)))
    E, stats = photo_loss(
        f, x_theta, hit_for_photo, n,
        vi, ctx.alt_nn, origins_all,                  # ← origins from φ_k (visibility)
        ctx.images, K_all, w2c_all,                   # ← w2c/K from φ_k (reprojection)
        ctx.feature_maps, ctx.masks, fg_self,
        ctx.H, ctx.W, uv_self,
        train_cfg.n_alt, train_cfg.cos_thresh,
        train_cfg.w_photo, train_cfg.w_feature, train_cfg.w_ncc,
        train_cfg.ncc_patch, train_cfg.ncc_half_pix,
        train_cfg.sample_mode, train_cfg.gaussian_sigma, radius,
        step, train_cfg.ncc_min, train_cfg.occ_mode,
        hit_bg=hit_bg if trace_cfg.bsphere_radius > 0 else None,
        w_ncc_normal=train_cfg.w_ncc_normal,
        ncc_topk=train_cfg.ncc_topk,
        ncc_color=train_cfg.ncc_color,
        ncc_grad_alpha=train_cfg.ncc_grad_alpha,
        ncc_normal_patch=train_cfg.ncc_normal_patch,
        ncc_normal_half_pix=train_cfg.ncc_normal_half_pix,
        ncc_patch_wsigma=train_cfg.ncc_patch_wsigma,
        trace_cfg=trace_cfg,
    )
    aux = dict(o=o, u=u, fg_self=fg_self, x_theta=x_theta, hit=hit,
               eik_pts=eik_pts, sdf_min=sdf_min, stats=stats)
    return E, aux


def sdf_regularizers(f, aux: dict, ctx: BAContext, train_cfg: TrainConfig,
                     trace_cfg: TraceConfig, alpha: float) -> Tensor:
    """R(θ): the SDF-validity regularisers from training, evaluated on this batch.

    Reuses the exact loss.py terms (eikonal + IDR/min-SDF silhouette + DVR mask +
    behind-hit) with the TrainConfig weights, so the θ-block keeps f a valid
    1-Lipschitz SDF rather than overfitting the cameras photometrically. Only the
    terms with weight > 0 contribute; MVS/SfM/feature terms (which need extra
    precomputed data) are intentionally out of scope for the BA pass.
    """
    dev = ctx.device
    o, u, hit, fg_self = aux["o"], aux["u"], aux["hit"], aux["fg_self"]
    reg = torch.zeros((), device=dev)
    if train_cfg.w_eikonal > 0:
        reg = reg + train_cfg.w_eikonal * eikonal_loss(
            f, aux["eik_pts"], train_cfg.n_eik_vol, dev)
    if train_cfg.w_idr_mask > 0:
        m, _ = idr_mask_loss(f, o, u, hit, fg_self, alpha,
                             train_cfg.idr_n_samples,
                             train_cfg.sil_t_near, train_cfg.sil_t_far)
        reg = reg + train_cfg.w_idr_mask * m
    if train_cfg.w_sil > 0:
        reg = reg + train_cfg.w_sil * mask_loss_min_sdf(
            aux["sdf_min"], fg_self, alpha,
            fg_offset=train_cfg.sil_fg_offset, bg_offset=train_cfg.sil_bg_offset,
            focal_gamma=train_cfg.sil_focal_gamma,
            balance_classes=train_cfg.sil_balance,
            normalize_by_alpha=train_cfg.sil_norm_alpha)
    if train_cfg.w_mask_fg > 0 or train_cfg.w_mask_bg > 0:
        mfg, mbg = dvr_mask_loss(f, o, u, fg_self, trace_cfg.t_far,
                                 train_cfg.mask_fg_margin, train_cfg.mask_bg_margin,
                                 train_cfg.n_mask_fg, train_cfg.n_mask_bg)
        reg = reg + train_cfg.w_mask_fg * mfg + train_cfg.w_mask_bg * mbg
    if train_cfg.w_behind_hit > 0:
        reg = reg + train_cfg.w_behind_hit * behind_hit_loss(
            f, aux["x_theta"], hit, u, train_cfg.behind_eps)
    return reg


def _silhouette_alpha(train_cfg: TrainConfig) -> float:
    """Converged silhouette sharpness α (matches the end of training's schedule)."""
    if train_cfg.sil_s_interval > 0:
        return train_cfg.sil_s * (2.0 ** train_cfg.sil_s_max_mults)
    return train_cfg.sil_s


# ───────────────────────────── BCD driver ──────────────────────────────────

def run_bundle_adjustment(f, ctx: BAContext, train_cfg: TrainConfig,
                          trace_cfg: TraceConfig, *, run_dir: Path | None = None,
                          device: str | None = None,
                          use_wandb: bool = False,
                          wandb_run_name: str | None = None,
                          perturb_rot_deg: float = 0.0,
                          perturb_trans_mm: float = 0.0,
                          perturb_seed: int = 0,
                          perturb_cam: int = -1,
                          free_only_cam: int = -1,
                          resume_ba: dict | None = None,
                          dtu_scale: float = 1.0) -> tuple[object, CameraParams]:
    """Block coordinate descent over (θ, φ) from a converged θ.

    Alternates `cycles` times: a φ-block (block_phi steps, minimise E, θ frozen)
    then a θ-block (block_theta steps, minimise E + R(θ), φ frozen) — or the
    reverse when phi_first is False. Each block keeps its own Adam state, so the
    two coordinate blocks never share moment estimates (block-diagonal). Returns
    the refined model and the learned CameraParams.

    Synthetic recovery test (perturb_rot_deg / perturb_trans_mm > 0): before BA,
    each non-gauge camera (i≥1) is rotated by a fixed angle about a random axis
    and translated by a fixed magnitude in a random direction. The original
    calibrated poses are treated as ground truth; per-step rotation (deg) and
    translation (mm) error vs that GT is logged so BA's ability to *recover* a
    known perturbation can be measured directly (independent of Chamfer noise).
    """
    device = device or next(f.parameters()).device
    ba = train_cfg.bundle

    # --- synthetic perturbation for the recovery test ------------------------
    gt_R = gt_t = None
    if perturb_rot_deg > 0 or perturb_trans_mm > 0:
        gt_R = ctx.c2w_base[:, :3, :3].clone()
        gt_t = ctx.c2w_base[:, :3, 3].clone()
        if ba.lock_first and perturb_cam == 0:
            raise ValueError("--perturb-cam 0 is gauge-locked (lock_first): its "
                             "pose cannot move, so nothing would be perturbed. "
                             "Pick a camera >=1 or pass --no-lock-first.")
        trans_units = perturb_trans_mm / dtu_scale if dtu_scale > 0 else perturb_trans_mm
        perturbed = _perturb_poses(ctx.c2w_base, perturb_rot_deg, trans_units,
                                   perturb_seed, lock_first=ba.lock_first,
                                   only_cam=perturb_cam)
        ctx = replace(ctx, c2w_base=perturbed)
        _which_cam = f"cam {perturb_cam} only" if perturb_cam >= 0 else "all i>=1"
        print(f"  [BA] PERTURB recovery test: rot={perturb_rot_deg:g}° "
              f"trans={perturb_trans_mm:g}mm (={trans_units:.5g} units) "
              f"seed={perturb_seed}  cam0 locked={ba.lock_first}  "
              f"perturbed: {_which_cam}", flush=True)
        if run_dir is not None:
            run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"gt_c2w": torch.stack([
                            torch.cat([torch.cat([gt_R[i], gt_t[i, :, None]], 1),
                                       torch.tensor([[0., 0., 0., 1.]])], 0)
                            for i in range(gt_R.shape[0])]),
                        "perturbed_c2w": perturbed.cpu(),
                        "perturb_rot_deg": perturb_rot_deg,
                        "perturb_trans_mm": perturb_trans_mm,
                        "perturb_seed": perturb_seed,
                        "perturb_cam": perturb_cam,
                        "dtu_scale": dtu_scale},
                       run_dir / "ba_perturb_gt.pt")

    cam_params = CameraParams(ctx.c2w_base, lock_first=ba.lock_first,
                              K_base=ctx.K_all,
                              opt_intrinsics=ba.opt_intrinsics).to(device)
    # Optionally free ONLY one camera (e.g. the perturbed one in a recovery test):
    # zero every other camera's extrinsic gradient so the rig cannot translation-
    # drift and the recovery metric is read on that camera in isolation.
    if free_only_cam >= 0:
        only = torch.zeros_like(cam_params.free_mask)
        only[free_only_cam] = 1.0
        cam_params.free_mask.copy_(only)
        print(f"  [BA] free_only_cam={free_only_cam}: optimising that camera "
              f"alone, all others frozen ({int(only.sum())}/{only.shape[0]} free)",
              flush=True)
    # φ-block optimiser: extrinsics at lr, intrinsics (if enabled) at lr_intrinsics.
    phi_groups = [{"params": [cam_params.log_rot, cam_params.dt], "lr": ba.lr}]
    if ba.opt_intrinsics:
        phi_groups.append({"params": cam_params.intrinsics_params(),
                           "lr": ba.lr_intrinsics})
    opt_phi   = torch.optim.Adam(phi_groups)
    opt_theta = torch.optim.Adam(f.parameters(),          lr=ba.lr_theta)

    # --- resume: continue a previous BA pass from its saved φ + optimiser state ---
    # The perturbation above is reproduced deterministically (same seed/args), so
    # the perturbed base and GT match the original run; here we overlay the learned
    # deltas (partial recovery) and both Adam states so descent picks up exactly
    # where it stopped — NOT a cold restart from the perturbation.
    resume_gstep = 0
    if resume_ba is not None:
        cam_params.load_state_dict(resume_ba["cam_params"])
        opt_phi.load_state_dict(resume_ba["opt_phi"])
        opt_theta.load_state_dict(resume_ba["opt_theta"])
        resume_gstep = int(resume_ba.get("step", 0))
        rr, tt = cam_params.deltas()
        print(f"  [BA] RESUME from step {resume_gstep}: |Δφ| rot={rr:.4f}° "
              f"trans={tt:.4g} (φ + Adam state restored)", flush=True)

    alpha = _silhouette_alpha(train_cfg)

    if gt_R is not None:
        gt_R = gt_R.to(device); gt_t = gt_t.to(device)

    def _pose_err_vs_gt():
        """Pose error of the current φ vs the unperturbed GT poses, as
        (mean_rot°, mean_trans_mm, cam_rot°, cam_trans_mm):

          * mean_*  — averaged over all free (non-gauge) cameras;
          * cam_*   — for the single perturbed camera (`perturb_cam`) alone, so a
            single-camera recovery test is read without the dilution of the 62
            initially-correct cameras' residual jitter. NaN when no single camera
            was targeted (perturb_cam < 0) or without a GT.
        """
        if gt_R is None:
            return float("nan"), float("nan"), float("nan"), float("nan")
        with torch.no_grad():
            c2w = cam_params()
            R, t = c2w[:, :3, :3], c2w[:, :3, 3]

            def _err(Rsel, tsel, gRsel, gtsel):
                Rrel = Rsel @ gRsel.transpose(-1, -2)
                tr = Rrel.diagonal(dim1=-2, dim2=-1).sum(-1)
                cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
                rot = (cos.arccos().mean() * 180.0 / math.pi).item()
                tmm = ((tsel - gtsel).norm(dim=-1).mean() * dtu_scale).item()
                return rot, tmm

            m = cam_params.free_mask.squeeze(-1) > 0
            rot, tmm = _err(R[m], t[m], gt_R[m], gt_t[m])
            if perturb_cam >= 0:
                i = slice(perturb_cam, perturb_cam + 1)
                crot, ctmm = _err(R[i], t[i], gt_R[i], gt_t[i])
            else:
                crot = ctmm = float("nan")
        return rot, tmm, crot, ctmm

    # The θ-block regulariser reads train_cfg.w_eikonal. This run trained with
    # w_eikonal=0, so ba.w_eikonal>0 turns eikonal back on FOR THE BA PASS ONLY —
    # keeping f a valid SDF (|∇f|≈1) while θ is refined jointly with the poses.
    if ba.w_eikonal > 0 and train_cfg.w_eikonal != ba.w_eikonal:
        train_cfg = replace(train_cfg, w_eikonal=ba.w_eikonal)
        print(f"  [BA] θ-block eikonal regulariser: w_eikonal={ba.w_eikonal:g} "
              f"(overrides run's {0.0})", flush=True)

    # --- wandb ---------------------------------------------------------------
    wandb_log = (lambda d, step=None: None)
    if use_wandb:
        try:
            import wandb
            wandb.init(project="1lip-tracer-ba",
                       name=wandb_run_name or (run_dir.name if run_dir else "ba"),
                       config={"trace": dataclasses_asdict(trace_cfg),
                               "bundle": dataclasses_asdict(ba)})
            url = getattr(getattr(wandb, "run", None), "url", None)
            if url:
                print(f"  [BA] [wandb] {url}", flush=True)
            def wandb_log(d, step=None):
                try:
                    wandb.log(d, step=step)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [BA] wandb disabled ({e})", flush=True)

    if run_dir is not None:
        run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [BA] block coordinate descent: {ba.cycles} cycles × "
          f"(φ:{ba.block_phi} + θ:{ba.block_theta})  "
          f"lr_φ={ba.lr} lr_θ={ba.lr_theta}  lock_first={ba.lock_first}")
    if ba.opt_intrinsics:
        print(f"  [BA] intrinsics K (fx,fy,cx,cy) refined in φ-block  "
              f"lr_K={ba.lr_intrinsics} (all cameras free)", flush=True)
    _trace_name = "trace_idr" if trace_cfg.grad_mode == "idr" else "trace_unrolled"
    _beta_note = ""
    if trace_cfg.grad_mode == "idr" and trace_cfg.sdf_min_beta > 0 and train_cfg.w_sil <= 0:
        _beta_note = f"  [sdf_min_beta {trace_cfg.sdf_min_beta:g}->0 for compacted trace_idr; w_sil=0 so unused]"
    print(f"  [BA] batch={ba.batch or train_cfg.batch}  α_sil={alpha:.3g}  "
          f"trace={_trace_name}(iters={trace_cfg.iters}){_beta_note}")

    gstep = resume_gstep   # global step — continues numbering across a resume

    # --- per-step metrics CSV (for CVPR plots) -------------------------------
    # One row per optimiser step: photometric E, total loss, NCC (ZNCC), the
    # fraction of textured ray-pairs kept by the photometric gate, the active
    # block's gradient L2-norm (pre-clip for θ), and the cumulative mean pose
    # delta. stdout gets a line every LOG_EVERY steps; the CSV gets every step.
    LOG_EVERY = max(1, getattr(ba, "log_every", 20))
    csv_file = csv_writer = None
    if run_dir is not None:
        import csv as _csv
        csv_file = open(run_dir / "ba_metrics.csv", "w", newline="")
        csv_writer = _csv.writer(csv_file)
        csv_writer.writerow(["gstep", "cycle", "block", "E", "loss",
                             "ncc_zncc", "ncc_kept_frac", "grad_norm",
                             "rot_deg_mean", "trans_mean", "lr",
                             "rot_err_gt_deg", "trans_err_gt_mm",
                             "rot_err_cam_deg", "trans_err_cam_mm"])
        csv_file.flush()
        print(f"  [BA] per-step metrics → {run_dir / 'ba_metrics.csv'}", flush=True)

    def _run_block(which: str, n_steps: int, cycle: int):
        """Run one coordinate block. `which` ∈ {'phi','theta'}."""
        nonlocal gstep
        train_phi = which == "phi"
        # Freeze the inactive block's parameters so no spurious grad is computed.
        for p in cam_params.parameters():
            p.requires_grad_(train_phi)
        for p in f.parameters():
            p.requires_grad_(not train_phi)
        opt = opt_phi if train_phi else opt_theta
        active_params = (list(cam_params.parameters()) if train_phi
                         else list(f.parameters()))
        lr = opt.param_groups[0]["lr"]
        last = {}
        for _ in range(n_steps):
            idx = _sample_batch(ctx)
            E, aux = photometric_objective(f, cam_params, idx, ctx,
                                           train_cfg, trace_cfg, gstep)
            loss = E if train_phi else E + sdf_regularizers(
                f, aux, ctx, train_cfg, trace_cfg, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not train_phi:
                # clip_grad_norm_ returns the *pre-clip* total norm — log that.
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    f.parameters(), max_norm=1.0))
            else:
                grad_norm = float(torch.norm(torch.stack(
                    [p.grad.norm() for p in active_params if p.grad is not None])))
            opt.step()
            gstep += 1

            st = aux["stats"]
            zncc = float(st.get("ncc_zncc", float("nan")))
            kept_frac = st.get("ncc_kept", 0) / max(st.get("ncc_textured", 0), 1)
            rot_deg, trans = cam_params.deltas()
            rot_err, trans_err, cam_rot_err, cam_trans_err = _pose_err_vs_gt()
            E_f, loss_f = float(E.detach()), float(loss.detach())
            if csv_writer is not None:
                csv_writer.writerow([gstep, cycle + 1, which,
                                     f"{E_f:.6f}", f"{loss_f:.6f}", f"{zncc:.6f}",
                                     f"{kept_frac:.4f}", f"{grad_norm:.6e}",
                                     f"{rot_deg:.6f}", f"{trans:.6e}", lr,
                                     f"{rot_err:.6f}", f"{trans_err:.6f}",
                                     f"{cam_rot_err:.6f}", f"{cam_trans_err:.6f}"])
                csv_file.flush()
            if gstep % LOG_EVERY == 0:
                if gt_R is not None and perturb_cam >= 0:
                    _errstr = (f"  cam{perturb_cam}_err[rot={cam_rot_err:.4f}° "
                               f"trans={cam_trans_err:.4f}mm]")
                elif gt_R is not None:
                    _errstr = f"  err_gt[rot={rot_err:.4f}° trans={trans_err:.4f}mm]"
                else:
                    _errstr = ""
                print(f"  [BA] c{cycle + 1} {which} step {gstep}  "
                      f"E={E_f:.4f} loss={loss_f:.4f} ncc={zncc:.3f} "
                      f"kept={kept_frac:.2f} |g|={grad_norm:.2e}  "
                      f"|Δφ|rot={rot_deg:.4f}° trans={trans:.3e}{_errstr}", flush=True)
            _wl = {f"{which}/E": E_f, f"{which}/loss": loss_f,
                   f"{which}/ncc_zncc": zncc, f"{which}/ncc_kept_frac": kept_frac,
                   f"{which}/grad_norm": grad_norm,
                   "pose/rot_deg_mean": rot_deg, "pose/trans_mean": trans,
                   "cycle": cycle + 1}
            if gt_R is not None:
                _wl["recovery/rot_err_gt_deg"] = rot_err
                _wl["recovery/trans_err_gt_mm"] = trans_err
                if perturb_cam >= 0:
                    _wl["recovery/rot_err_cam_deg"] = cam_rot_err
                    _wl["recovery/trans_err_cam_mm"] = cam_trans_err
            wandb_log(_wl, step=gstep)
            last = dict(E=E_f, loss=loss_f, stats=st, grad_norm=grad_norm)
        # restore grads so both blocks stay differentiable next round
        for p in cam_params.parameters():
            p.requires_grad_(True)
        for p in f.parameters():
            p.requires_grad_(True)
        return last

    order = ("phi", "theta") if ba.phi_first else ("theta", "phi")
    for cycle in range(ba.cycles):
        stamp = {}
        for which in order:
            n_steps = ba.block_phi if which == "phi" else ba.block_theta
            if n_steps > 0:
                stamp[which] = _run_block(which, n_steps, cycle)
        rot_deg, trans = cam_params.deltas()
        msg = f"  [BA] cycle {cycle + 1}/{ba.cycles}"
        for which in order:
            if which in stamp:
                s = stamp[which]
                ncc = s["stats"].get("ncc_zncc", float("nan"))
                msg += (f"  {which}: E={s['E']:.4f} ncc={ncc:.3f} "
                        f"|g|={s.get('grad_norm', float('nan')):.2e}")
        msg += f"  |Δφ|: rot={rot_deg:.3f}° trans={trans:.4g}"
        print(msg, flush=True)

        if run_dir is not None and (cycle + 1) % max(ba.ckpt_every, 1) == 0:
            _save_ba_checkpoint(run_dir / "ba_latest.pt", f, cam_params,
                                opt_phi, opt_theta, cycle, gstep)

    if run_dir is not None:
        _save_ba_checkpoint(run_dir / "ba_final.pt", f, cam_params,
                            opt_phi, opt_theta, ba.cycles - 1, gstep)
        print(f"  [BA] saved → {run_dir / 'ba_final.pt'}")
    if csv_file is not None:
        csv_file.close()
    return f, cam_params


def _save_ba_checkpoint(path: Path, f, cam_params: CameraParams,
                        opt_phi, opt_theta, cycle: int, gstep: int) -> None:
    payload = {
        "f": f.state_dict(),
        "cam_params": cam_params.state_dict(),
        "opt_phi": opt_phi.state_dict(),
        "opt_theta": opt_theta.state_dict(),
        "cycle": cycle, "step": gstep,
        "architecture": f.architecture, "group_size": f.group_size,
        "depth": f.depth, "activation": f.activation,
        "input_encoding": f.input_encoding, "multires": f.multires,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ─────────────────────────── checkpoint entrypoint ─────────────────────────

def _dtu_scale_from_scene(scene) -> float:
    """Read DTU scale_mat (normalized-unit → mm) so pose errors log in mm.

    cameras.npz stores scale_mat_i = diag(s,s,s,1); s is the unit→mm factor for
    that scan. Returns 1.0 (units == mm) if the file/key is absent."""
    try:
        import numpy as np
        d = np.load(Path(scene) / "cameras.npz")
        k = next((x for x in d.files if x.startswith("scale_mat_")
                  and "inv" not in x), None)
        if k is not None:
            return float(d[k][0, 0])
    except Exception:
        pass
    return 1.0


def bundle_adjust_from_checkpoint(ckpt_path: Path, cfg: Config, *,
                                  run_dir: Path | None = None,
                                  device: str | None = None,
                                  use_wandb: bool = False,
                                  perturb_rot_deg: float = 0.0,
                                  perturb_trans_mm: float = 0.0,
                                  perturb_seed: int = 0,
                                  perturb_cam: int = -1,
                                  free_only_cam: int = -1,
                                  resume_ba_path: Path | None = None) -> tuple[object, CameraParams]:
    """Load a converged θ checkpoint + data from `cfg`, then run the BCD pass.

    With `resume_ba_path`, the model weights are taken from that BA checkpoint
    (the θ refined so far) and its φ + optimiser state are restored so the pass
    continues; pass the SAME perturb args as the original run so the perturbed
    base / GT are reproduced deterministically.
    """
    from .data import load_views, load_blender_views
    from .model import make_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mc = cfg.model
    f = make_model(hidden=mc.hidden, depth=mc.depth, group_size=mc.group_size,
                   activation=mc.activation, input_encoding=mc.input_encoding,
                   multires=mc.multires, architecture=mc.architecture).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    print(f"  [BA] loaded converged θ from {ckpt_path}")

    resume_ba = None
    if resume_ba_path is not None:
        resume_ba = torch.load(resume_ba_path, map_location="cpu")
        f.load_state_dict(resume_ba["f"], strict=False)   # most-recent θ wins
        print(f"  [BA] resuming φ + θ from BA checkpoint {resume_ba_path} "
              f"(step {resume_ba.get('step', 0)})")

    if cfg.train.use_blender:
        views = load_blender_views(scene=cfg.scene, split="train", down=1)
        dtu_scale = 1.0
    else:
        views = load_views(cfg.scene)
        dtu_scale = _dtu_scale_from_scene(cfg.scene)
    ctx = prepare_context(views, cfg.train, device)
    return run_bundle_adjustment(f, ctx, cfg.train, cfg.trace,
                                 run_dir=run_dir, device=device,
                                 use_wandb=use_wandb,
                                 wandb_run_name=run_dir.name if run_dir else None,
                                 perturb_rot_deg=perturb_rot_deg,
                                 perturb_trans_mm=perturb_trans_mm,
                                 perturb_seed=perturb_seed,
                                 perturb_cam=perturb_cam,
                                 free_only_cam=free_only_cam,
                                 resume_ba=resume_ba,
                                 dtu_scale=dtu_scale)


def main() -> None:
    import argparse
    from .train import load_config_json

    ap = argparse.ArgumentParser(
        description="Block-coordinate photometric bundle adjustment (θ, φ).")
    ap.add_argument("--config", type=Path, required=True,
                    help="run config.json (defines model / trace / loss weights / scene)")
    ap.add_argument("--ckpt",   type=Path, required=True,
                    help="converged θ checkpoint to start from")
    ap.add_argument("--out",    type=Path, default=None,
                    help="output dir for BA checkpoints (default: alongside --ckpt)")
    # BA overrides (fall back to the config's BundleAdjustConfig)
    ap.add_argument("--cycles",      type=int,   default=None)
    ap.add_argument("--block-phi",   type=int,   default=None)
    ap.add_argument("--block-theta", type=int,   default=None)
    ap.add_argument("--lr-phi",      type=float, default=None)
    ap.add_argument("--lr-theta",    type=float, default=None)
    ap.add_argument("--batch",       type=int,   default=None)
    ap.add_argument("--w-eikonal",   type=float, default=None,
                    help="eikonal weight for the θ-block ONLY (keeps f a valid SDF "
                         "when refining θ jointly; needed when the run trained w_eikonal=0)")
    ap.add_argument("--log-every",   type=int,   default=None,
                    help="stdout per-step log cadence (CSV always logs every step)")
    ap.add_argument("--wandb",       action="store_true",
                    help="log per-step metrics to Weights & Biases (project 1lip-tracer-ba)")
    ap.add_argument("--theta-first", action="store_true",
                    help="start each cycle with the θ-block instead of φ")
    ap.add_argument("--no-lock-first", action="store_true",
                    help="do not gauge-lock camera 0 (allows global drift)")
    ap.add_argument("--opt-intrinsics", action="store_true",
                    help="also refine per-camera intrinsics K (fx,fy,cx,cy) in the φ-block")
    ap.add_argument("--lr-intrinsics", type=float, default=None,
                    help="intrinsics learning rate (dimensionless delta; default from config)")
    # --- synthetic perturbation-recovery test ---
    ap.add_argument("--perturb-rot-deg", type=float, default=0.0,
                    help="inject a fixed rotation (deg, random axis) into each "
                         "non-gauge camera before BA; logs rot/trans error vs the "
                         "original calibrated poses so recovery can be measured")
    ap.add_argument("--perturb-trans-mm", type=float, default=0.0,
                    help="inject a fixed translation (mm, random direction) per camera")
    ap.add_argument("--perturb-seed", type=int, default=0,
                    help="RNG seed for the injected perturbation (reproducible)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue a previous BA pass from its ba_final.pt/ba_latest.pt "
                         "(restores φ + both Adam states + step). Pass the SAME perturb "
                         "args so the perturbed base / GT are reproduced.")
    ap.add_argument("--free-only-cam", type=int, default=-1,
                    help="optimise ONLY this camera; freeze all others (and cam0). "
                         "For a clean single-camera recovery test so the rest of the "
                         "rig cannot translation-drift. -1 = all i>=1 free (default).")
    ap.add_argument("--perturb-cam", type=int, default=-1,
                    help="perturb ONLY this camera index (>=1; cam 0 is gauge-locked); "
                         "all others stay at GT — the cleanest minimal recovery test. "
                         "Default -1 perturbs every non-gauge camera.")
    args = ap.parse_args()

    cfg = load_config_json(args.config)
    ba = cfg.train.bundle
    ba.enabled = True
    if args.cycles      is not None: ba.cycles      = args.cycles
    if args.block_phi   is not None: ba.block_phi   = args.block_phi
    if args.block_theta is not None: ba.block_theta = args.block_theta
    if args.lr_phi      is not None: ba.lr          = args.lr_phi
    if args.lr_theta    is not None: ba.lr_theta    = args.lr_theta
    if args.batch       is not None: ba.batch       = args.batch
    if args.w_eikonal   is not None: ba.w_eikonal   = args.w_eikonal
    if args.log_every   is not None: ba.log_every   = args.log_every
    if args.theta_first:   ba.phi_first  = False
    if args.no_lock_first: ba.lock_first = False
    if args.opt_intrinsics: ba.opt_intrinsics = True
    if args.lr_intrinsics is not None: ba.lr_intrinsics = args.lr_intrinsics

    run_dir = args.out or (args.ckpt.parent / "bundle_adjust")
    bundle_adjust_from_checkpoint(args.ckpt, cfg, run_dir=run_dir,
                                  use_wandb=args.wandb,
                                  perturb_rot_deg=args.perturb_rot_deg,
                                  perturb_trans_mm=args.perturb_trans_mm,
                                  perturb_seed=args.perturb_seed,
                                  perturb_cam=args.perturb_cam,
                                  free_only_cam=args.free_only_cam,
                                  resume_ba_path=args.resume)


if __name__ == "__main__":
    main()
