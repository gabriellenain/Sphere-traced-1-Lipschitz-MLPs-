"""Coarse-to-fine surface refinement via normal deformation fields.

Pipeline:
  1. Sphere trace on frozen fθ  →  x ∈ Sθ
  2. nθ(x) = ∇fθ(x) / ‖∇fθ(x)‖   (autograd, detached)
  3. xψ = x + δψ(x) · nθ(x)        (gradient flows only through δψ)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import FTheta
from .positional_encoding import PositionalEncoding


class DeformationField(nn.Module):
    """Scalar displacement MLP δψ : R³ → R, bounded by δmax via tanh.

    NeuS-style: Softplus activations + skip connection at mid-depth.
    Input is Fourier PE(x, L) to capture high-frequency details.
    """

    def __init__(self, hidden: int = 256, depth: int = 8, delta_max: float = 0.01,
                 skip_at: int = 4, multires: int = 6) -> None:
        super().__init__()
        self.delta_max = delta_max
        self.skip_at   = skip_at
        self.pe        = PositionalEncoding(multires=multires, input_dims=3)
        pe_dim         = self.pe.out_dim   # 3*(2*6+1) = 39

        self.layers = nn.ModuleList()
        for i in range(depth):
            in_dim = (pe_dim if i == 0 else hidden) + (pe_dim if i == skip_at else 0)
            self.layers.append(nn.Linear(in_dim, hidden))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.pe(x)
        pe_x = h
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, pe_x], dim=-1)
            h = F.softplus(layer(h))
        return self.delta_max * torch.tanh(self.out(h).squeeze(-1))


def deform_hits(
    f: FTheta,
    x: Tensor,
    psi: DeformationField,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply normal deformation to surface hit points.

    Args:
        f   : frozen coarse field fθ
        x   : (N, 3) hit points on Sθ (from sphere tracing, detached)
        psi : learned deformation field δψ

    Returns:
        x_psi   : (N, 3) deformed surface points xψ = x + δψ(x)·nθ(x)
        delta   : (N,)   scalar displacements δψ(x)
        n_theta : (N, 3) frozen coarse normals nθ(x)
    """
    xr = x.detach().requires_grad_(True)
    with torch.enable_grad():
        sdf_val = f(xr).sum()
        n_theta = torch.autograd.grad(sdf_val, xr)[0]
    n_theta = F.normalize(n_theta.detach(), dim=-1)

    delta = psi(x)                          # (N,)  — gradient flows through ψ
    x_psi = x.detach() + delta.unsqueeze(-1) * n_theta
    return x_psi, delta, n_theta
