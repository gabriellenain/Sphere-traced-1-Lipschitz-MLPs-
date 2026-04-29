"""Positional encodings for 1-Lipschitz SDF networks."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def gamma_lipschitz_constant(multires: int) -> float:
    """Closed-form Lipschitz constant of the unnormalized NeuS encoder."""
    if multires < 0:
        raise ValueError(f"multires must be non-negative, got {multires}")
    return math.sqrt((4 ** multires + 2) / 3)


class NormalizedPositionalEncoding(nn.Module):
    """NeuS-style positional encoding (unnormalized — CPL spectral norm absorbs Lipschitz constant)."""

    def __init__(self, multires: int, input_dims: int = 3) -> None:
        super().__init__()
        if multires < 0:
            raise ValueError(f"multires must be non-negative, got {multires}")
        if input_dims <= 0:
            raise ValueError(f"input_dims must be positive, got {input_dims}")

        self.multires = multires
        self.input_dims = input_dims
        self.out_dim = input_dims * (2 * multires + 1)

        freq_bands = 2.0 ** torch.arange(multires, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dims:
            raise ValueError(
                f"expected last dim {self.input_dims}, got {x.shape[-1]}"
            )
        if self.multires == 0:
            return x

        xb = x.unsqueeze(-2) * self.freq_bands.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        sin_cos = torch.stack((torch.sin(xb), torch.cos(xb)), dim=-2)
        sin_cos = sin_cos.reshape(*x.shape[:-1], 2 * self.multires * self.input_dims)
        return torch.cat((x, sin_cos), dim=-1)


def get_normalized_embedder(multires: int, input_dims: int = 2):
    """Return an embedder callable and its output dimension."""
    embedder = NormalizedPositionalEncoding(multires=multires, input_dims=input_dims)
    return embedder, embedder.out_dim


class LipschitzPositionalEncoding(nn.Module):
    """Intrinsically 1-Lipschitz PE: Γ_{ρ,L}(x) = (ρ·x, √(1−ρ²)·normalized_PE(x)).

    Each feature is divided by √L·π·2^k so the PE Jacobian has orthogonal
    columns each of norm 1, giving spectral norm 1.  Weights (ρ, √(1−ρ²)) lie
    on the unit circle, so ‖J_Γ·v‖² = ρ²+( 1−ρ²)·1 = 1 for any unit v —
    the Lipschitz constant is exactly 1 for all L.
    Output dim: input_dims * (2*multires + 1)  (same shape as NeuS encoder).
    """

    def __init__(self, multires: int, input_dims: int = 3, rho: float = 0.5) -> None:
        super().__init__()
        if multires < 1:
            raise ValueError(f"multires must be ≥ 1, got {multires}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        self.multires    = multires
        self.input_dims  = input_dims
        self.rho         = rho
        self.rho_pe      = math.sqrt(1.0 - rho ** 2)   # √(1−ρ²)
        self.out_dim     = input_dims * (2 * multires + 1)

        freq_bands = math.pi * (2.0 ** torch.arange(multires, dtype=torch.float32))  # (L,)
        norms      = 1.0 / (math.sqrt(multires) * freq_bands)                        # 1/(√L·π·2^k)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("norms",      norms,      persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected last dim {self.input_dims}, got {x.shape[-1]}")
        freq = self.freq_bands.to(dtype=x.dtype, device=x.device)  # (L,)
        nrm  = self.norms.to(dtype=x.dtype, device=x.device)       # (L,)
        xb   = x.unsqueeze(-2) * freq.unsqueeze(-1)                 # (..., L, D)
        sc   = torch.stack([torch.sin(xb), torch.cos(xb)], dim=-2) * nrm.unsqueeze(-1).unsqueeze(-1)
        pe   = sc.reshape(*x.shape[:-1], 2 * self.multires * self.input_dims)
        return torch.cat([self.rho * x, self.rho_pe * pe], dim=-1)
