"""
N-Activation: a learnable 1-Lipschitz piecewise-linear activation.
Reference: Prach & Lampert, "1-Lipschitz Neural Networks are more expressive
with N-Activations" (arXiv:2311.06103).
https://github.com/berndprach/NActivation
"""

import torch
import torch.nn as nn


def n_activation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Piecewise-linear 1-Lipschitz activation parameterised by two breakpoints.

    x     : [..., C, ...]  — any shape, channel dim is dim 1
    theta : [C, 2]         — per-channel breakpoints (sorted internally)
    """
    # min/max instead of torch.sort: identical (theta has exactly 2 columns) but
    # stays on the Triton codegen path — sort forces an inductor C++ fallback.
    t0 = torch.minimum(theta[:, 0], theta[:, 1])   # == theta_min
    t1 = torch.maximum(theta[:, 0], theta[:, 1])   # == theta_max
    for _ in range(len(x.shape) - 2):
        t0 = t0[..., None]                     # broadcast to spatial dims
        t1 = t1[..., None]

    out = torch.where(x < t0, x - 2 * t0,
          torch.where(x < t1, -x,
                               x - 2 * t1))
    return out


class NActivation(nn.Module):
    """
    Drop-in 1-Lipschitz replacement for ReLU with learnable breakpoints.

    Args:
        in_channels : number of channels / features (C)
        init        : (theta0, theta1) uniform breakpoints, default (-1, 0) — the
                      SDF-appropriate default (every channel is a real "N", no
                      sign-destroying |x|). Pass "absid" for the paper's
                      classification init (alternating Abs / Identity channels,
                      arXiv:2311.06103) — worse here, see model.py.
        trainable   : whether theta is a learnable parameter
        lr_factor   : scale learning rate for theta independently
    """

    def __init__(
        self,
        in_channels: int,
        init: "str | tuple[float, float]" = (-1.0, 0.0),
        trainable: bool = True,
        lr_factor: float = 1.0,
    ):
        super().__init__()
        self._scale = lr_factor ** 0.5
        if init == "absid":
            theta = torch.zeros(in_channels, 2)
            theta[0::2, 0] = -100.0          # even channels → (-100, 0) = Abs
            # odd channels remain (0, 0)     # odd  channels → Identity
        else:
            theta = torch.tensor(init).expand(in_channels, -1).clone()
        self.theta = nn.Parameter(theta / self._scale, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return n_activation(x, self.theta * self._scale)

    def extra_repr(self) -> str:
        return f"in_channels={self.theta.shape[0]}"
