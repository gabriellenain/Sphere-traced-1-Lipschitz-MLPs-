"""Positional encoding for 1-Lipschitz SDF networks."""
from __future__ import annotations

import torch
from torch import Tensor, nn


_LIP_MODES = (None, "uniform", "per_band")


class PositionalEncoding(nn.Module):
    """Plain Fourier PE: [x, sin(2^0 x), cos(2^0 x), ..., sin(2^(L-1) x), cos(2^(L-1) x)].

    Output dim: input_dims * (2 * multires + 1).

    ``lipschitz_mode`` controls whether the encoding γ is rescaled so the
    composition (γ then a 1-Lipschitz MLP) preserves |∇f| ≤ 1 in world space:

    - ``None`` (default): raw PE, not 1-Lipschitz.
    - ``"uniform"``: divide the whole output by ``K = sqrt((4^L + 2)/3)``.
      Tight, but ~all of the gradient budget lands on the top octave; the
      passthrough and low bands are crushed by 1/K.
    - ``"per_band"``: divide each band k by ``2^k``, then divide the whole
      output by ``sqrt(L + 1)``. Same overall Lipschitz constant of 1, but
      the gradient budget is split evenly across passthrough and all bands
      (each contributes ``1/sqrt(L+1)`` to ‖∂γ/∂x_i‖). Better default for
      SDFs where coarse shape supervision matters.
    """

    def __init__(
        self,
        multires: int,
        input_dims: int = 3,
        lipschitz_mode: str | None = None,
    ) -> None:
        super().__init__()
        if multires < 0:
            raise ValueError(f"multires must be non-negative, got {multires}")
        if input_dims <= 0:
            raise ValueError(f"input_dims must be positive, got {input_dims}")
        if lipschitz_mode not in _LIP_MODES:
            raise ValueError(
                f"lipschitz_mode must be one of {_LIP_MODES}, got {lipschitz_mode!r}"
            )
        self.multires       = multires
        self.input_dims     = input_dims
        self.out_dim        = input_dims * (2 * multires + 1)
        self.lipschitz_mode = lipschitz_mode
        freq_bands = 2.0 ** torch.arange(multires, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected last dim {self.input_dims}, got {x.shape[-1]}")
        if self.multires == 0:
            return x
        freq = self.freq_bands.to(dtype=x.dtype, device=x.device)
        xb = x.unsqueeze(-2) * freq.unsqueeze(-1)
        sin_cos = torch.stack((torch.sin(xb), torch.cos(xb)), dim=-2)

        if self.lipschitz_mode == "per_band":
            # Divide each band k by 2^k so every (sin, cos) pair contributes 1
            # to the column-gradient norm². stack(dim=-2) yields shape
            # (..., L, 2, D); broadcast freq as (L, 1, 1) over the last three dims.
            sin_cos = sin_cos / freq.view(self.multires, 1, 1)

        sin_cos = sin_cos.reshape(*x.shape[:-1], 2 * self.multires * self.input_dims)
        out = torch.cat((x, sin_cos), dim=-1)

        if self.lipschitz_mode == "uniform":
            # K = sqrt(1 + sum_{k=0}^{L-1} 4^k) = sqrt((4^L + 2)/3).
            k = ((4.0 ** self.multires + 2.0) / 3.0) ** 0.5
            out = out / k
        elif self.lipschitz_mode == "per_band":
            # After per-band weighting, column-norm² = 1 + L → divide by sqrt(L+1).
            out = out / (self.multires + 1) ** 0.5
        return out
