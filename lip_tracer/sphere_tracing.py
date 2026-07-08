"""Differentiable and non-differentiable sphere tracing for 1-Lip SDFs."""
from __future__ import annotations

import torch
import torch.utils.checkpoint as _chk
from torch import Tensor

from .config import TraceConfig
from .model import FTheta

_DEFAULT_TRACE = TraceConfig()
_LAST_TRACE_STATS: dict[str, object] = {}


def _needs_bracket(f: FTheta) -> bool:
    # Fire bracket+root-find whenever the field is not 1-Lipschitz in world space:
    # NeuS (Softplus) or any model using positional encoding. PE voids |∇f|≤1 in
    # world coords, so sphere-tracing can overshoot and leave a sign flip on
    # (t_n, t_{n+1}); bracketing converts the overshoot into a clean root find.
    if getattr(f, "architecture", None) == "neus":
        return True
    return getattr(f, "input_encoding", "identity") == "pe"


def get_last_trace_stats() -> dict[str, object]:
    return dict(_LAST_TRACE_STATS)


def _record_trace_stats(
    name: str,
    use_bracket: bool,
    bracketed: Tensor,
    bracket_lo: Tensor,
    bracket_hi: Tensor,
    mean_iters: float | None = None,
) -> None:
    with torch.no_grad():
        count = bracketed.sum().detach()
        batch = max(int(bracketed.numel()), 1)
        width = (bracket_hi - bracket_lo).abs()
        masked_width = torch.where(bracketed, width, torch.zeros_like(width))
        denom = count.clamp(min=1)
        _LAST_TRACE_STATS.clear()
        _LAST_TRACE_STATS.update({
            "trace": name,
            "neus_bracket_enabled": use_bracket,
            "neus_bracket_count": count,
            "neus_bracket_frac": count / batch,
            "neus_bracket_width_mean": masked_width.sum().detach() / denom,
            "neus_bracket_width_max": masked_width.max().detach() if masked_width.numel() > 0 else count,
        })
        if mean_iters is not None:
            _LAST_TRACE_STATS["mean_iters"] = float(mean_iters)


@torch.no_grad()
def _refine_bracketed_roots(
    f: FTheta,
    o: Tensor,
    d: Tensor,
    t_lo: Tensor,
    t_hi: Tensor,
    mask: Tensor,
    steps: int,
) -> Tensor:
    """Newton refinement constrained to a sign-change interval."""
    if steps <= 0 or not mask.any():
        return 0.5 * (t_lo + t_hi)

    lo = torch.minimum(t_lo, t_hi).clone()
    hi = torch.maximum(t_lo, t_hi).clone()
    flo = f(o + lo.unsqueeze(-1) * d)
    t = 0.5 * (lo + hi)
    eps_fd = 1e-3

    for _ in range(steps):
        x = o + t.unsqueeze(-1) * d
        fval = f(x)
        ddir = (f(x + eps_fd * d) - f(x - eps_fd * d)) / (2.0 * eps_fd)
        sign = torch.where(ddir != 0, ddir.sign(), torch.ones_like(ddir))
        ddir_safe = sign * ddir.abs().clamp(min=1e-6)
        newton = t - fval / ddir_safe
        mid = 0.5 * (lo + hi)
        inside = (newton > lo) & (newton < hi) & torch.isfinite(newton) & (ddir.abs() > 1e-6)
        cand = torch.where(inside, newton, mid)
        fc = f(o + cand.unsqueeze(-1) * d)

        left_contains_root = flo * fc <= 0
        hi = torch.where(mask & left_contains_root, cand, hi)
        lo = torch.where(mask & ~left_contains_root, cand, lo)
        flo = torch.where(mask & ~left_contains_root, fc, flo)
        t = torch.where(mask, cand, t)

    return t


def ray_sphere_exit(o: Tensor, d: Tensor, R: float) -> Tensor:
    """Per-ray exit distance for the bounding sphere of radius R centred at origin.

    Solves |o + t*d|² = R² for the larger (exit) root.
    If the ray misses the sphere (disc < 0, shouldn't happen when cameras surround
    the object inside the sphere) falls back to a very large value.

    o : (B, 3)  ray origins
    d : (B, 3)  unit ray directions
    Returns (B,) exit depths, always ≥ 0.
    """
    b    = (o * d).sum(-1)           # o·d
    c    = (o * o).sum(-1) - R * R   # |o|² − R²
    disc = b * b - c                 # discriminant (|d|=1 assumed)
    t    = -b + disc.clamp(min=0.0).sqrt()
    return t.clamp(min=0.0)


def ray_sphere_entry(o: Tensor, d: Tensor, R: float) -> Tensor:
    """Per-ray NEAR intersection distance with the bounding sphere of radius R
    centred at the origin (|d|=1). Used to START the trace on the sphere, skipping
    the empty camera→object gap.

    Returns t_entry ≥ 0 for rays that enter the sphere from outside; 0 for rays
    that miss it or start inside (nothing to skip). center=origin assumes the
    object is origin-normalised (DTU/IDR); do not use on un-centred rigs.
    """
    b    = (o * d).sum(-1)
    c    = (o * o).sum(-1) - R * R
    disc = b * b - c
    t    = -b - disc.clamp(min=0.0).sqrt()
    hits = (disc > 0) & (t > 0)
    return torch.where(hits, t, torch.zeros_like(t))


def _trace_t0(o: Tensor, d: Tensor, cfg: TraceConfig) -> Tensor:
    """Initial ray parameter t for a trace: the bounding-sphere entry when
    cfg.bsphere_start_radius > 0, else 0 (start at the camera). Detached — t0 is a
    constant offset that just skips empty space; gradients flow via later f-evals."""
    if getattr(cfg, "bsphere_start_radius", 0.0) > 0:
        return ray_sphere_entry(o.detach(), d.detach(), cfg.bsphere_start_radius)
    return torch.zeros(o.shape[0], device=o.device)


def _newton_step(
    f: FTheta, o: Tensor, d: Tensor, t: Tensor, gate: Tensor, eps: float
) -> Tensor:
    """One safeguarded Newton step on g(t)=f(o+t·d).

    Uses the *analytic* directional derivative g' = ∇f·d (exact, no finite-
    difference bias/noise) and clamps the correction to ±k·eps so a near-grazing
    ray (|∇f·d|→0 on a 1-Lipschitz field) cannot overshoot the surface.
    Returns the new t (only `gate` rays are moved).
    """
    K = 3.0
    with torch.enable_grad():
        x = (o + t.unsqueeze(-1) * d).detach().requires_grad_(True)
        fx = f(x)
        g = torch.autograd.grad(fx.sum(), x)[0]
    fval = fx.detach()
    ddir = (g.detach() * d).sum(-1)
    sign = torch.where(ddir >= 0, torch.ones_like(ddir), -torch.ones_like(ddir))
    ddir = sign * ddir.abs().clamp(min=1e-6)
    delta = (fval / ddir).clamp(-K * eps, K * eps)
    return t - torch.where(gate, delta, torch.zeros_like(t))


def trace_unrolled(
    f: FTheta, o: Tensor, d: Tensor,
    cfg: TraceConfig = _DEFAULT_TRACE,
    collect_eik: bool = True,
    diff_normal: bool = False,
    attach_normal_point: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Differentiable sphere tracing — exact gradients through unrolled iterations.

    Each iteration evaluates f with grad enabled; the accumulated t carries
    gradients through the full chain of f evaluations.  Boolean masks (converged,
    escaped) are detached so they don't inject non-differentiable ops.

    Collects one sample point every cfg.eik_stride iterations (detached) for
    eikonal regularisation along the trace, not just at the final hit.

    When cfg.bsphere_radius > 0, each ray uses its own per-ray t_far = exit depth
    of the bounding sphere.  Rays that reach this exit without converging are flagged
    hit_bg=True: x_theta lands on the sphere surface, which is photometrically
    inconsistent across views, providing a gradient signal that pulls the surface
    inward to fill holes.

    Returns:
        x_theta : (B, 3) surface hit points (differentiable)
        t       : (B,)   ray distances
        hit     : (B,)   bool — real convergence (|f| < eps, within t_far)
        eik_pts : (E, 3) detached sample points for eikonal loss
        n_raw   : (B, 3) surface normals (detached)
        sdf_min : (B,)   minimum SDF along each ray — for mask loss σ(−α·sdf_min)
        hit_bg  : (B,)   bool — reached bounding sphere without converging
    """
    B = o.shape[0]
    n_eik = ((cfg.iters - 1) // cfg.eik_stride + 1) if collect_eik else 0
    eik_buf = torch.empty(n_eik * B, 3, device=o.device) if n_eik > 0 else None
    eik_slot = 0

    # Per-ray t_far: bounding-sphere exit or global constant.
    if cfg.bsphere_radius > 0:
        t_far_ray = ray_sphere_exit(o.detach(), d.detach(), cfg.bsphere_radius)
    else:
        t_far_ray = torch.full((B,), cfg.t_far, device=o.device)

    t = _trace_t0(o, d, cfg)
    sdf = torch.zeros(B, device=o.device)
    sdf_min = torch.full((B,), float("inf"), device=o.device)
    sdf_iters: list[Tensor] = []                          # for soft-min logsumexp
    converged = torch.zeros(B, dtype=torch.bool, device=o.device)
    escaped   = torch.zeros(B, dtype=torch.bool, device=o.device)
    use_bracket = _needs_bracket(f)
    prev_t = torch.zeros(B, device=o.device)
    prev_sdf = torch.zeros(B, device=o.device)
    have_prev = torch.zeros(B, dtype=torch.bool, device=o.device)
    bracketed = torch.zeros(B, dtype=torch.bool, device=o.device)
    bracket_lo = torch.zeros(B, device=o.device)
    bracket_hi = torch.zeros(B, device=o.device)
    for i in range(cfg.iters):
        active = ~(converged | escaped)
        if not active.any():
            break
        x = o + t.unsqueeze(-1) * d
        if eik_buf is not None and i % cfg.eik_stride == 0:
            eik_buf[eik_slot * B:(eik_slot + 1) * B] = x.detach()
            eik_slot += 1
        sdf = _chk.checkpoint(f, x, use_reentrant=False) if torch.is_grad_enabled() else f(x)
        sdf_min = torch.minimum(sdf_min, sdf)
        if cfg.sdf_min_beta > 0:
            sdf_iters.append(sdf)
        with torch.no_grad():
            crossing = active & have_prev & (prev_sdf * sdf.detach() < 0) if use_bracket else torch.zeros_like(converged)
            bracket_lo = torch.where(crossing, prev_t, bracket_lo)
            bracket_hi = torch.where(crossing, t.detach(), bracket_hi)
            bracketed = bracketed | crossing
            converged = (sdf.detach().abs() < cfg.eps) | bracketed
            escaped   = t >= t_far_ray
            keep_prev = active & ~(converged | escaped)
            prev_t = torch.where(keep_prev, t.detach(), prev_t)
            prev_sdf = torch.where(keep_prev, sdf.detach(), prev_sdf)
            have_prev = have_prev | keep_prev
        step = torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
        t = t + step

    if cfg.sdf_min_beta > 0 and sdf_iters:
        # soft-min: -1/β · logsumexp(-β · sdf_k). Gradient flows through every
        # iteration weighted by proximity to the minimum (vs. hard min where
        # gradient flows only through the argmin iteration).
        sdf_stack = torch.stack(sdf_iters, dim=0)        # (n_iters, B)
        sdf_min = -torch.logsumexp(-cfg.sdf_min_beta * sdf_stack, dim=0) / cfg.sdf_min_beta

    # Reuse the last loop's sdf instead of a separate forward pass.
    if use_bracket and bracketed.any():
        t_br = _refine_bracketed_roots(f, o.detach(), d.detach(), bracket_lo, bracket_hi,
                                       bracketed, cfg.newton_steps)
        t = torch.where(bracketed, t_br, t)
    _record_trace_stats("unrolled", use_bracket, bracketed, bracket_lo, bracket_hi)

    hit    = ((sdf.detach().abs() < cfg.eps) | bracketed) & (t.detach() < t_far_ray) & (t.detach() >= 0)
    hit_bg = (~hit) & (t.detach() >= t_far_ray - cfg.eps)

    # Newton refinement — analytic directional derivative + clamped step.
    # delta is computed detached so t keeps its grad_fn (IDR-style gradient).
    newton_gate = hit & ~bracketed
    for _ in range(cfg.newton_steps):
        with torch.no_grad():
            t_ref = _newton_step(f, o.detach(), d, t.detach(), newton_gate, cfg.eps)
            delta = t.detach() - t_ref
        t = t - delta  # outside no_grad: t keeps its grad_fn

    # Compute normals here — reuses final position, saves one f-call in train loop.
    # diff_normal=True keeps the double-backward graph so n_raw = ∇f(x*) carries
    # gradient w.r.t. θ (x* stays detached → normal-only branch of the PMVS loss).
    xr = (o + t.detach().unsqueeze(-1) * d).detach().requires_grad_(True)
    with torch.enable_grad():
        n_raw = torch.autograd.grad(f(xr).sum(), xr, create_graph=diff_normal)[0]
    if not diff_normal:
        n_raw = n_raw.detach()

    x_theta = o + t.unsqueeze(-1) * d
    # "Detach nothing": re-evaluate the normal at the differentiable x_theta so the
    # loss also carries the position→normal edge ∂n/∂x·∂x_theta/∂θ.
    if diff_normal and attach_normal_point:
        with torch.enable_grad():
            n_raw = torch.autograd.grad(f(x_theta).sum(), x_theta,
                                        create_graph=True)[0]
    eik_out = eik_buf[:eik_slot * B] if eik_buf is not None else torch.empty(0, 3, device=o.device)
    return x_theta, t, hit, eik_out, n_raw, sdf_min, hit_bg


def trace_idr(
    f: FTheta, o: Tensor, d: Tensor,
    cfg: TraceConfig = _DEFAULT_TRACE,
    collect_eik: bool = True,
    diff_normal: bool = False,
    attach_normal_point: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Differentiable sphere tracing via IDR implicit gradient (Yariv et al. 2020).

    Runs a no-grad sphere trace to convergence, then applies the implicit
    function theorem to attach gradients w.r.t. network parameters θ without
    backpropagating through the full trace loop:

        x_θ = x* − f(x*) / (n · d) · d

    where x* is the detached hit point, n = ∇_x f(x*), and f(x*) is
    re-evaluated with gradient tracking for θ only.  At convergence f(x*) ≈ 0
    so x_θ ≈ x* geometrically, but ∂x_θ/∂θ = −d/(n·d) · ∂f(x*)/∂θ, which is
    exactly the implicit differentiation gradient.

    When cfg.bsphere_radius > 0, rays that exit the bounding sphere without
    converging are marked hit_bg=True.  x_theta lands on the sphere surface.
    The IDR correction is zeroed for hit_bg rays (x* is far from any surface so
    the correction would be unreliable), but x_theta still participates in the
    photo loss — photometric inconsistency at the sphere exit provides gradient
    signal indirectly via the mask / sdf_min losses.

    Same return signature as trace_unrolled (7-tuple).
    """
    B = o.shape[0]
    n_eik = ((cfg.iters - 1) // cfg.eik_stride + 1) if collect_eik else 0
    eik_buf = torch.empty(n_eik * B, 3, device=o.device) if n_eik > 0 else None
    eik_slot = 0

    # Per-ray t_far: bounding-sphere exit or global constant.
    if cfg.bsphere_radius > 0:
        t_far_ray = ray_sphere_exit(o.detach(), d.detach(), cfg.bsphere_radius)
    else:
        t_far_ray = torch.full((B,), cfg.t_far, device=o.device)

    # --- no-grad trace ---
    with torch.no_grad():
        t = _trace_t0(o, d, cfg)
        sdf = torch.zeros(B, device=o.device)
        sdf_min = torch.full((B,), float("inf"), device=o.device)
        x_iters: list[Tensor] = []                    # detached x for differentiable sdf_min
        converged = torch.zeros(B, dtype=torch.bool, device=o.device)
        escaped   = torch.zeros(B, dtype=torch.bool, device=o.device)
        use_bracket = _needs_bracket(f)
        prev_t = torch.zeros(B, device=o.device)
        prev_sdf = torch.zeros(B, device=o.device)
        have_prev = torch.zeros(B, dtype=torch.bool, device=o.device)
        bracketed = torch.zeros(B, dtype=torch.bool, device=o.device)
        bracket_lo = torch.zeros(B, device=o.device)
        bracket_hi = torch.zeros(B, device=o.device)
        iters_per_ray = torch.zeros(B, device=o.device)
        # Active-ray compaction: each iter evaluates f only on the unconverged subset.
        # Per-ray state stays full-B and is index-updated; only the f(x) input/output shrinks.
        for i in range(cfg.iters):
            escaped = t >= t_far_ray
            active = ~(converged | escaped)
            # The compacted loop already materialises idx via nonzero (a device
            # sync); break on its size instead of a separate active.any() → one
            # fewer host sync per iteration, and never feed an empty batch to f.
            idx   = active.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break
            iters_per_ray = iters_per_ray + active.float()
            if eik_buf is not None and i % cfg.eik_stride == 0:
                eik_buf[eik_slot * B:(eik_slot + 1) * B] = o + t.unsqueeze(-1) * d
                eik_slot += 1
            t_a   = t[idx]
            sdf_a = f(o[idx] + t_a.unsqueeze(-1) * d[idx])
            sdf_min[idx] = torch.minimum(sdf_min[idx], sdf_a)
            if cfg.sdf_min_beta > 0:
                # NOTE: sdf_min_beta>0 path not maintained under compaction — soft-min
                # over per-iter x_iters needs full-B snapshots. Re-add full-B branch if
                # you turn this on again.
                raise NotImplementedError("sdf_min_beta>0 not supported with compacted trace_idr")
            if use_bracket:
                crossing_a      = have_prev[idx] & (prev_sdf[idx] * sdf_a < 0)
                bracket_lo[idx] = torch.where(crossing_a, prev_t[idx], bracket_lo[idx])
                bracket_hi[idx] = torch.where(crossing_a, t_a,         bracket_hi[idx])
                bracketed[idx]  = bracketed[idx] | crossing_a
                new_conv_a      = (sdf_a.abs() < cfg.eps) | bracketed[idx]
            else:
                new_conv_a      = sdf_a.abs() < cfg.eps
            converged[idx]  = converged[idx] | new_conv_a
            keep_a          = ~new_conv_a                # escaped[idx]=False on active
            prev_t[idx]     = torch.where(keep_a, t_a,   prev_t[idx])
            prev_sdf[idx]   = torch.where(keep_a, sdf_a, prev_sdf[idx])
            have_prev[idx]  = have_prev[idx] | keep_a
            t[idx]          = t_a + torch.where(new_conv_a, torch.zeros_like(sdf_a), sdf_a)

        if use_bracket and bracketed.any():
            t = torch.where(
                bracketed,
                _refine_bracketed_roots(f, o, d, bracket_lo, bracket_hi, bracketed, cfg.newton_steps),
                t,
            )
        _record_trace_stats("idr", use_bracket, bracketed, bracket_lo, bracket_hi,
                            mean_iters=iters_per_ray.mean().item())

        # converged already encodes (|sdf|<eps | bracketed) from the compacted loop.
        hit    = converged & (t < t_far_ray) & (t >= 0)
        hit_bg = (~hit) & (t >= t_far_ray - cfg.eps)

        # Newton refinement — real hits only. Keep the directional derivative's
        # sign and clamp the correction: otherwise a near-grazing hit can jump
        # far beyond t_far while remaining marked as a valid photo-loss hit.
        newton_gate = hit & ~bracketed
        for _ in range(cfg.newton_steps):
            t = _newton_step(f, o, d, t, newton_gate, cfg.eps)

    # --- differentiable sdf_min via soft-min over detached trace points ---
    # IDR's no-grad trace gives a detached sdf_min, which kills mask_loss_min_sdf.
    # Re-evaluate f at the trace points outside no_grad in one batched call so
    # sdf_min carries gradient w.r.t. θ. The x positions stay detached, so the
    # gradient is purely "make f smaller at these specific points" — first-order,
    # no coupling through the trace path. One flat forward+backward of width
    # n_iters·B (vs backprop's 36-deep checkpointed chain).
    if cfg.sdf_min_beta > 0 and x_iters:
        x_stack = torch.stack(x_iters, dim=0)              # (n_iters, B, 3)
        n_it    = x_stack.shape[0]
        with torch.enable_grad():
            sdf_grad = f(x_stack.reshape(-1, 3)).reshape(n_it, B)
        sdf_min = -torch.logsumexp(-cfg.sdf_min_beta * sdf_grad, dim=0) / cfg.sdf_min_beta

    # --- normals at converged point ---
    # diff_normal=True keeps the double-backward graph so the returned n_raw
    # carries gradient w.r.t. θ; the IDR correction below always uses a detached
    # normal so x_theta's gradient stays purely the f_star implicit term.
    x_star = (o + t.unsqueeze(-1) * d).detach()
    xr = x_star.requires_grad_(True)
    with torch.enable_grad():
        n_raw = torch.autograd.grad(f(xr).sum(), xr, create_graph=diff_normal)[0]
    if not diff_normal:
        n_raw = n_raw.detach()
    n_geo = n_raw.detach()

    # --- IDR implicit gradient ---
    # n · d: directional derivative of f along the ray.  Clamp away from zero
    # (grazing rays) while preserving sign so the correction stays on the ray.
    n_dot_d = (n_geo * d).sum(-1)                             # (B,)
    sign    = torch.where(n_dot_d != 0, n_dot_d.sign(), torch.ones_like(n_dot_d))
    n_dot_d_safe = sign * n_dot_d.abs().clamp(min=1e-3)        # (B,)

    # f(x*) with grad only for θ — x* is detached so ∂/∂x doesn't flow
    with torch.enable_grad():
        f_star = f(x_star)                                     # (B,)

    # IDR correction only for real hits; bg hits use x* directly (no correction —
    # the sphere exit is far from any surface, so f*/n·d would be unreliable).
    correction = torch.where(hit, f_star / n_dot_d_safe.detach(), torch.zeros_like(f_star))
    x_theta = x_star + correction.unsqueeze(-1) * (-d)        # x* - (f*/n·d)*d

    # "Detach nothing": re-evaluate the returned normal at the DIFFERENTIABLE
    # x_theta so the loss also carries the position→normal edge
    # ∂n/∂x·∂x_theta/∂θ (curvature × surface motion). The IDR correction above
    # keeps using n_geo (detached x_star), so x_theta's own gradient is unchanged.
    if diff_normal and attach_normal_point:
        with torch.enable_grad():
            n_raw = torch.autograd.grad(f(x_theta).sum(), x_theta,
                                        create_graph=True)[0]

    eik_out = eik_buf[:eik_slot * B] if eik_buf is not None else torch.empty(0, 3, device=o.device)
    return x_theta, t, hit, eik_out, n_raw, sdf_min, hit_bg


@torch.no_grad()
def trace_nograd(
    f: FTheta, o: Tensor, d: Tensor,
    cfg: TraceConfig = _DEFAULT_TRACE,
    return_diag: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Non-differentiable sphere trace — occlusion checks and rendering.

    Returns:
        x_hit : (B, 3)
        t     : (B,)
        hit   : (B,) bool — real convergence only (hit_bg excluded)
        f_pre : (B,) |f| at the hit *before* the Newton loop — only when
                return_diag=True; for diagnosing Newton de-convergence.
    """
    B = o.shape[0]

    if cfg.bsphere_radius > 0:
        t_far_ray = ray_sphere_exit(o, d, cfg.bsphere_radius)
    else:
        t_far_ray = torch.full((B,), cfg.t_far, device=o.device)

    t = _trace_t0(o, d, cfg)
    sdf = torch.zeros(B, device=o.device)
    converged = torch.zeros(B, dtype=torch.bool, device=o.device)
    use_bracket = _needs_bracket(f)
    prev_t = torch.zeros(B, device=o.device)
    prev_sdf = torch.zeros(B, device=o.device)
    have_prev = torch.zeros(B, dtype=torch.bool, device=o.device)
    bracketed = torch.zeros(B, dtype=torch.bool, device=o.device)
    bracket_lo = torch.zeros(B, device=o.device)
    bracket_hi = torch.zeros(B, device=o.device)
    # Active-ray compaction: each iter evaluates f only on the unconverged subset.
    # Per-ray state (t, prev_*, bracketed, converged) stays full-B and is
    # index-updated; only the f(x) input/output shrinks.
    for _ in range(cfg.iters):
        escaped = t >= t_far_ray
        active  = ~(converged | escaped)
        # break on the compacted index size (already materialised by nonzero)
        # instead of a separate active.any() → one fewer host sync per iteration.
        idx   = active.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        t_a   = t[idx]
        sdf_a = f(o[idx] + t_a.unsqueeze(-1) * d[idx])
        if use_bracket:
            crossing_a    = have_prev[idx] & (prev_sdf[idx] * sdf_a < 0)
            bracket_lo[idx] = torch.where(crossing_a, prev_t[idx], bracket_lo[idx])
            bracket_hi[idx] = torch.where(crossing_a, t_a,         bracket_hi[idx])
            bracketed[idx]  = bracketed[idx] | crossing_a
            new_conv_a = (sdf_a.abs() < cfg.eps) | bracketed[idx]
        else:
            new_conv_a = sdf_a.abs() < cfg.eps
        converged[idx] = converged[idx] | new_conv_a
        keep_a = ~new_conv_a                              # escaped[idx]=False on active
        prev_t[idx]    = torch.where(keep_a, t_a,   prev_t[idx])
        prev_sdf[idx]  = torch.where(keep_a, sdf_a, prev_sdf[idx])
        have_prev[idx] = have_prev[idx] | keep_a
        t[idx] = t_a + torch.where(new_conv_a, torch.zeros_like(sdf_a), sdf_a)
    sdf = f(o + t.unsqueeze(-1) * d)
    if use_bracket and bracketed.any():
        t = torch.where(
            bracketed,
            _refine_bracketed_roots(f, o, d, bracket_lo, bracket_hi, bracketed, cfg.newton_steps),
            t,
        )
        sdf = f(o + t.unsqueeze(-1) * d)
    _record_trace_stats("nograd", use_bracket, bracketed, bracket_lo, bracket_hi)
    hit = (converged | (sdf.abs() < cfg.eps) | bracketed) & (t < t_far_ray) & (t >= 0)
    f_pre = sdf.detach().abs()  # |f| at the hit before Newton refinement

    # Newton refinement — analytic directional derivative + clamped step.
    newton_gate = hit & ~bracketed
    for _ in range(cfg.newton_steps):
        t = _newton_step(f, o, d, t, newton_gate, cfg.eps)

    x_hit = o + t.unsqueeze(-1) * d
    if return_diag:
        return x_hit, t, hit, f_pre
    return x_hit, t, hit
