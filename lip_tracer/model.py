"""1-Lipschitz SDF network: CPL layers with MaxMin / GroupSort activation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvexPotentialLayer(nn.Module):
    """1-Lipschitz CPL: l(x) = x − (2/‖W‖²) Wᵀ ReLU(Wx + b)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        nn.init.orthogonal_(self.weight)
        self.register_buffer("_u", F.normalize(torch.randn(dim), dim=0, eps=1e-12))

    def _sigma_sq(self, update_u: bool = True) -> Tensor:
        w = self.weight
        if update_u:
            with torch.no_grad():
                u = self._u
                v = F.normalize(w.t() @ u, dim=0, eps=1e-12)
                u = F.normalize(w @ v, dim=0, eps=1e-12)
                self._u.copy_(u)
        u_d = self._u.detach().clone()
        v = F.normalize(w.t() @ u_d, dim=0, eps=1e-12).detach()
        sigma = torch.dot(u_d, w @ v)
        return sigma * sigma

    def forward(self, x: Tensor) -> Tensor:
        # Do not update the power-iteration buffer during no_grad tracing passes.
        sigma_sq = self._sigma_sq(update_u=torch.is_grad_enabled()).clamp(min=1e-12)
        y = F.linear(x, self.weight, self.bias)
        y = F.relu(y)
        y = F.linear(y, self.weight.t())
        return x - (2.0 / sigma_sq) * y


class MaxMin(nn.Module):
    """GroupSort-2 — 1-Lipschitz. Default activation."""

    def forward(self, x: Tensor) -> Tensor:
        pairs = x.unflatten(-1, (-1, 2))
        return torch.cat([pairs.max(-1, keepdim=True).values,
                          pairs.min(-1, keepdim=True).values], dim=-1).flatten(-2, -1)


class GroupSort(nn.Module):
    """GroupSort-N — 1-Lipschitz, N>2 for more expressivity (Prach & Lampert 2022)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (-1, self.n)).sort(-1).values.flatten(-2, -1)


class FTheta(nn.Module):
    """1-Lipschitz SDF network: CPL stack with GroupSort activation.

    group_size=2  → MaxMin (default, backward compatible with old checkpoints).
    group_size>2  → GroupSort-N (more expressive, new checkpoints only).
    """

    def __init__(self, hidden: int = 512, depth: int = 12, group_size: int = 2) -> None:
        super().__init__()
        assert hidden % group_size == 0
        self.hidden     = hidden
        self.depth      = depth
        self.group_size = group_size
        blocks: list[nn.Module] = []
        for i in range(depth):
            blocks.append(ConvexPotentialLayer(hidden))
            if i < depth - 1:
                blocks.append(MaxMin() if group_size == 2 else GroupSort(group_size))
        self.net        = nn.Sequential(*blocks)
        self.head_weight = nn.Parameter(torch.empty(hidden))
        self.head_bias   = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.head_weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        h = F.pad(x, (0, self.hidden - x.shape[-1]))
        h = self.net(h)
        w = self.head_weight / torch.linalg.vector_norm(self.head_weight).clamp(min=1e-6)
        return (h * w).sum(-1) + self.head_bias.squeeze(-1)
