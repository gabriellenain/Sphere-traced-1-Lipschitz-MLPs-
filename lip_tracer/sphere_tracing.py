"""Differentiable and non-differentiable sphere tracing for 1-Lip SDFs."""
from __future__ import annotations

import torch
import torch.utils.checkpoint as _chk
from torch import Tensor

from .config import TraceConfig
from .model import FTheta

_DEFAULT_TRACE = TraceConfig()


def trace_unrolled(
    f: FTheta, o: Tensor, d: Tensor,
    cfg: TraceConfig = _DEFAULT_TRACE,
    collect_eik: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Differentiable sphere tracing — exact gradients through unrolled iterations.

    Each iteration evaluates f with grad enabled; the accumulated t carries
    gradients through the full chain of f evaluations.  Boolean masks (converged,
    escaped) are detached so they don't inject non-differentiable ops.

    Collects one sample point every cfg.eik_stride iterations (detached) for
    eikonal regularisation along the trace, not just at the final hit.

    Returns:
        x_theta : (B, 3) surface hit points (differentiable)
        t       : (B,)   ray distances
        hit     : (B,)   bool — converged & within t_far
        eik_pts : (E, 3) detached sample points for eikonal loss
        n_raw   : (B, 3) surface normals (detached)
        sdf_min : (B,)   minimum SDF along each ray — for mask loss σ(−α·sdf_min)
    """
    B = o.shape[0]
    n_eik = cfg.iters // cfg.eik_stride if collect_eik else 0
    eik_buf = torch.empty(n_eik * B, 3, device=o.device) if n_eik > 0 else None
    eik_slot = 0

    t = torch.zeros(B, device=o.device)
    sdf = torch.zeros(B, device=o.device)
    sdf_min = torch.full((B,), float("inf"), device=o.device)
    converged = torch.zeros(B, dtype=torch.bool, device=o.device)
    escaped   = torch.zeros(B, dtype=torch.bool, device=o.device)
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
        with torch.no_grad():
            converged = sdf.detach().abs() < cfg.eps
            escaped   = t >= cfg.t_far
        step = torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
        t = t + step

    # Reuse the last loop's sdf instead of a separate forward pass.
    hit = (sdf.detach().abs() < cfg.eps) & (t.detach() < cfg.t_far) & (t.detach() >= 0)

    # Newton refinement via finite differences — delta detached, t grad_fn preserved
    eps_fd = 1e-3
    for _ in range(cfg.newton_steps):
        with torch.no_grad():
            x    = o + t.detach().unsqueeze(-1) * d
            fval = f(x)
            ddir = (f(x + eps_fd * d) - fval) / eps_fd
            ddir = ddir.abs().clamp(min=1e-6)
            delta = torch.where(hit, fval / ddir, torch.zeros_like(t.detach()))
        t = t - delta  # outside no_grad: t keeps its grad_fn

    # Compute normals here — reuses final position, saves one f-call in train loop
    xr = (o + t.detach().unsqueeze(-1) * d).detach().requires_grad_(True)
    with torch.enable_grad():
        n_raw = torch.autograd.grad(f(xr).sum(), xr, create_graph=False)[0].detach()

    x_theta = o + t.unsqueeze(-1) * d
    eik_out = eik_buf[:eik_slot * B] if eik_buf is not None else torch.empty(0, 3, device=o.device)
    return x_theta, t, hit, eik_out, n_raw, sdf_min


@torch.no_grad()
def trace_nograd(
    f: FTheta, o: Tensor, d: Tensor,
    cfg: TraceConfig = _DEFAULT_TRACE,
) -> tuple[Tensor, Tensor, Tensor]:
    """Non-differentiable sphere trace — occlusion checks and rendering.

    Returns:
        x_hit : (B, 3)
        t     : (B,)
        hit   : (B,) bool
    """
    B = o.shape[0]
    t = torch.zeros(B, device=o.device)
    sdf = torch.zeros(B, device=o.device)
    converged = torch.zeros(B, dtype=torch.bool, device=o.device)
    for _ in range(cfg.iters):
        escaped = t >= cfg.t_far
        active  = ~(converged | escaped)
        if not active.any():
            break
        sdf = f(o + t.unsqueeze(-1) * d)
        converged = sdf.abs() < cfg.eps
        t = t + torch.where(converged | escaped, torch.zeros_like(t), sdf)
    sdf = f(o + t.unsqueeze(-1) * d)
    hit = (sdf.abs() < cfg.eps) & (t < cfg.t_far) & (t >= 0)

    # Newton refinement via finite differences — stays inside @torch.no_grad()
    eps_fd = 1e-3
    for _ in range(cfg.newton_steps):
        x    = o + t.unsqueeze(-1) * d
        fval = f(x)
        ddir = (f(x + eps_fd * d) - fval) / eps_fd
        ddir = ddir.abs().clamp(min=1e-6)
        t    = t - torch.where(hit, fval / ddir, torch.zeros_like(t))

    return o + t.unsqueeze(-1) * d, t, hit
