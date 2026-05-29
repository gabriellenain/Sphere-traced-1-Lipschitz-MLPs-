"""Joint photometric bundle adjustment.

Minimal port of Furukawa-Ponce-style BA adapted to a differentiable renderer:
camera extrinsics are nn.Parameters refined by the existing photometric/NCC
loss, no separate feature matching or reprojection objective. Intrinsics are
kept fixed (DTU calibration is already trustworthy).

Parametrization: each camera's rotation is
    R = exp(skew(log_rot_v)) @ R_base_v
i.e. a small left-multiplied so(3) delta starting at zero, so the optimum at
step 0 is the identity perturbation. Translation is c2w[:3, 3] = t_base + dt.
Camera 0 is gauge-locked (its delta is masked to zero) so BA cannot drift
the whole scene rigidly.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


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


class CameraParams(nn.Module):
    """Learnable extrinsics delta on top of a frozen base c2w."""
    def __init__(self, c2w_base: Tensor, lock_first: bool = True):
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


def rays_from_pixels(c2w: Tensor, K: Tensor, px: Tensor, py: Tensor,
                     vi: Tensor) -> tuple[Tensor, Tensor]:
    """Rebuild (o, d) for a batch of rays from current camera params.

    c2w: (V, 4, 4)   K: (V, 3, 3)   px, py: (B,) float pixel coords   vi: (B,) long
    Returns o: (B, 3), d: (B, 3) unit-norm — both differentiable wrt c2w.
    """
    Kb = K[vi]; cw = c2w[vi]                                # (B, 3, 3), (B, 4, 4)
    x = (px - Kb[:, 0, 2]) / Kb[:, 0, 0]
    y = (py - Kb[:, 1, 2]) / Kb[:, 1, 1]
    d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (B, 3)
    d_w   = torch.einsum("bij,bj->bi", cw[:, :3, :3], d_cam)
    d_w   = d_w / d_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    o     = cw[:, :3, 3]
    return o, d_w
