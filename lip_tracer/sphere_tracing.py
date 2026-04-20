"""Differentiable and non-differentiable sphere tracing for 1-Lip SDFs."""
from __future__ import annotations

import torch
from torch import Tensor

from .config import TraceConfig
from .model import FTheta

_DEFAULT_TRACE = TraceConfig()


def trace_unrolled(
    f: FTheta, o: Tensor, d: Tensor,
    cfg: TraceConfig = _DEFAULT_TRACE,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    """
    t = torch.zeros(o.shape[0], device=o.device)
    eik_pts: list[Tensor] = []
    for i in range(cfg.iters):
        x = o + t.unsqueeze(-1) * d
        if i % cfg.eik_stride == 0:
            eik_pts.append(x.detach())
        sdf = f(x)
        with torch.no_grad():
            converged = sdf.abs() < cfg.eps
            escaped   = t >= cfg.t_far
        step = torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
        t = t + step

    with torch.no_grad():
        sdf_final = f(o + t.detach().unsqueeze(-1) * d)
        hit = (sdf_final.abs() < cfg.eps) & (t.detach() < cfg.t_far) & (t.detach() >= 0)
    x_theta = o + t.unsqueeze(-1) * d
    return x_theta, t, hit, torch.cat(eik_pts, dim=0)


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
    t = torch.zeros(o.shape[0], device=o.device)
    for _ in range(cfg.iters):
        sdf = f(o + t.unsqueeze(-1) * d)
        converged = sdf.abs() < cfg.eps
        t = t + torch.where(converged | (t >= cfg.t_far), torch.zeros_like(t), sdf)
    sdf = f(o + t.unsqueeze(-1) * d)
    hit = (sdf.abs() < cfg.eps) & (t < cfg.t_far) & (t >= 0)
    return o + t.unsqueeze(-1) * d, t, hit
