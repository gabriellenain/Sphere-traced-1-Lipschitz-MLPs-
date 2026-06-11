"""Monocular depth/normal consistency losses (MonoSDF-style).

Faithful, minimal re-implementation of the two priors used by MonoSDF
(Yu et al., 2022) — *not* a copy of their codebase. Adapted to this
sphere-tracing pipeline: we already have, per ray, a hit depth and a
surface normal (∇f/|∇f| at the hit point), so no volume-rendering
integration is needed — the losses act directly on per-ray quantities.

Two terms:

  • depth   — scale-and-shift invariant (SSI) L2. Monocular depth is only
              defined up to a per-image affine transform, so before the L2
              we solve, *per source image*, the least-squares
              (scale, shift) that best maps the prediction onto the
              rendered depth. Mixing rays from several images in one batch
              and aligning globally is the classic mistake — hence the
              per-image grouping via `image_ids`.

  • normal  — L1 + angular (1 − cos) between the rendered normal and the
              monocular normal. Both must be unit and in the *same frame*;
              the caller is responsible for bringing the monocular normals
              (camera space, as Omnidata outputs them) into world space.

The monocular predictions themselves (depth + normal maps) are assumed
precomputed offline (e.g. Omnidata) and sampled by the caller at the ray
pixels, exactly like the precomputed feature maps.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "ssi_depth_loss",
    "normal_consistency_loss",
    "mono_sdf_loss",
]


def _solve_scale_shift(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Closed-form least squares for (s, t) minimising ‖s·pred + t − target‖².

    2×2 normal equations:
        [Σpred²  Σpred] [s]   [Σpred·target]
        [Σpred   N    ] [t] = [Σtarget     ]
    Returns scalars (s, t); falls back to (1, 0) if degenerate.
    """
    n = pred.numel()
    if n == 0:
        return pred.new_tensor(1.0), pred.new_tensor(0.0)
    s_pp = (pred * pred).sum()
    s_p = pred.sum()
    s_pt = (pred * target).sum()
    s_t = target.sum()
    det = s_pp * n - s_p * s_p
    if det.abs() < 1e-9:
        return pred.new_tensor(1.0), pred.new_tensor(0.0)
    s = (s_pt * n - s_p * s_t) / det
    t = (s_pp * s_t - s_p * s_pt) / det
    return s, t


def ssi_depth_loss(
    pred_depth: Tensor,        # (B,) rendered/sphere-traced hit depth
    mono_depth: Tensor,        # (B,) monocular depth at the same rays
    mask: Tensor | None = None,  # (B,) bool — valid (hit & in mono support)
    image_ids: Tensor | None = None,  # (B,) source-image index per ray
) -> Tensor:
    """Scale-and-shift invariant L2 depth loss, aligned per source image.

    The affine (s, t) is solved on the *detached* tensors so the alignment
    does not itself receive gradient — only the residual does, matching the
    MonoSDF formulation.
    """
    if mask is not None:
        pred_depth = pred_depth[mask]
        mono_depth = mono_depth[mask]
        image_ids = None if image_ids is None else image_ids[mask]
    if pred_depth.numel() == 0:
        return pred_depth.new_zeros(())

    if image_ids is None:
        groups = [torch.ones_like(pred_depth, dtype=torch.bool)]
    else:
        groups = [(image_ids == i) for i in torch.unique(image_ids)]

    terms: list[Tensor] = []
    for g in groups:
        p = pred_depth[g]
        m = mono_depth[g]
        if p.numel() < 3:                      # too few rays to fit (s, t)
            continue
        s, t = _solve_scale_shift(m.detach(), p.detach())
        aligned = s * m + t                    # mono → rendered-depth scale
        terms.append(((p - aligned) ** 2).mean())
    if not terms:
        return pred_depth.new_zeros(())
    return torch.stack(terms).mean()


def normal_consistency_loss(
    pred_normal: Tensor,       # (B, 3) rendered normal ∇f/|∇f| (world)
    mono_normal: Tensor,       # (B, 3) monocular normal (world, unit)
    mask: Tensor | None = None,  # (B,) bool
) -> Tensor:
    """MonoSDF normal prior: L1 + angular, both in the same world frame."""
    if mask is not None:
        pred_normal = pred_normal[mask]
        mono_normal = mono_normal[mask]
    if pred_normal.numel() == 0:
        return pred_normal.new_zeros(())
    p = torch.nn.functional.normalize(pred_normal, dim=-1)
    m = torch.nn.functional.normalize(mono_normal, dim=-1)
    l1 = (p - m).abs().sum(-1).mean()
    angular = (1.0 - (p * m).sum(-1)).mean()
    return l1 + angular


def mono_sdf_loss(
    pred_depth: Tensor,
    mono_depth: Tensor,
    pred_normal: Tensor,
    mono_normal: Tensor,
    mask: Tensor | None = None,
    image_ids: Tensor | None = None,
    w_depth: float = 1.0,
    w_normal: float = 1.0,
) -> tuple[Tensor, dict]:
    """Combined MonoSDF prior. Returns (loss, per-term stats for logging)."""
    d = ssi_depth_loss(pred_depth, mono_depth, mask, image_ids)
    n = normal_consistency_loss(pred_normal, mono_normal, mask)
    loss = w_depth * d + w_normal * n
    return loss, {"mono_depth": float(d.detach()),
                  "mono_normal": float(n.detach())}
