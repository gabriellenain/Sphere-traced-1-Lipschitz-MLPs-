"""Positional encoding for 1-Lipschitz SDF networks."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Plain Fourier PE: [x, sin(2^0 x), cos(2^0 x), ..., sin(2^(L-1) x), cos(2^(L-1) x)].

    Output dim: input_dims * (2 * multires + 1).
    """

    def __init__(self, multires: int, input_dims: int = 3) -> None:
        super().__init__()
        if multires < 0:
            raise ValueError(f"multires must be non-negative, got {multires}")
        if input_dims <= 0:
            raise ValueError(f"input_dims must be positive, got {input_dims}")
        self.multires   = multires
        self.input_dims = input_dims
        self.out_dim    = input_dims * (2 * multires + 1)
        freq_bands = 2.0 ** torch.arange(multires, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected last dim {self.input_dims}, got {x.shape[-1]}")
        if self.multires == 0:
            return x
        xb = x.unsqueeze(-2) * self.freq_bands.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        sin_cos = torch.stack((torch.sin(xb), torch.cos(xb)), dim=-2)
        sin_cos = sin_cos.reshape(*x.shape[:-1], 2 * self.multires * self.input_dims)
        return torch.cat((x, sin_cos), dim=-1)
