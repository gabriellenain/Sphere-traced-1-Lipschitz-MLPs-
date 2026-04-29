"""
N-Activation: a learnable 1-Lipschitz piecewise-linear activation.
Reference: Prach & Lampert, "Almost-Orthogonal Layers for Efficient
General-Purpose Lipschitz Networks" (2022).
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
    t, _ = torch.sort(theta, dim=1)           # t[:, 0] <= t[:, 1]
    for _ in range(len(x.shape) - 2):
        t = t[..., None]                       # broadcast to spatial dims

    t0, t1 = t[:, 0], t[:, 1]
    out = torch.where(x < t0, x - 2 * t0,
          torch.where(x < t1, -x,
                               x - 2 * t1))
    return out


class NActivation(nn.Module):
    """
    Drop-in 1-Lipschitz replacement for ReLU with learnable breakpoints.

    Args:
        in_channels : number of channels / features (C)
        init        : (theta0, theta1) initial breakpoints, default (-1, 0)
        trainable   : whether theta is a learnable parameter
        lr_factor   : scale learning rate for theta independently
    """

    def __init__(
        self,
        in_channels: int,
        init: tuple[float, float] = (-1.0, 0.0),
        trainable: bool = True,
        lr_factor: float = 1.0,
    ):
        super().__init__()
        self._scale = lr_factor ** 0.5
        theta = torch.tensor(init).expand(in_channels, -1).clone() / self._scale
        self.theta = nn.Parameter(theta, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return n_activation(x, self.theta * self._scale)

    def extra_repr(self) -> str:
        return f"in_channels={self.theta.shape[0]}"
