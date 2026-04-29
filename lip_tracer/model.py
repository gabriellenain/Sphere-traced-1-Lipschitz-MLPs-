"""1-Lipschitz SDF network: CPL layers with MaxMin / GroupSort / NActivation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lip_tracer.n_activation import NActivation
from lip_tracer.positonal_encoding import LipschitzPositionalEncoding, NormalizedPositionalEncoding


class ConvexPotentialLayer(nn.Module):
    """1-Lipschitz CPL: l(x) = x − (2/‖W‖²) Wᵀ ReLU(Wx + b)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        nn.init.orthogonal_(self.weight)
        self.register_buffer("_u", F.normalize(torch.randn(dim), dim=0, eps=1e-12))
        self.register_buffer("_sigma_sq_buf", torch.ones(1))

    def _sigma_sq(self, update_u: bool = True) -> Tensor:
        if not update_u:
            return self._sigma_sq_buf  # cached from last training step — free lookup
        w = self.weight
        with torch.no_grad():
            u = self._u
            v = F.normalize(w.t() @ u, dim=0, eps=1e-12)
            u = F.normalize(w @ v, dim=0, eps=1e-12)
            self._u.copy_(u)
        u_d = self._u.detach().clone()
        v = F.normalize(w.t() @ u_d, dim=0, eps=1e-12).detach()
        sigma = torch.dot(u_d, w @ v)
        sigma_sq = sigma * sigma
        self._sigma_sq_buf.copy_(sigma_sq.detach())
        return sigma_sq

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
        pairs = x.view(*x.shape[:-1], -1, 2)
        return torch.stack([pairs.max(-1).values, pairs.min(-1).values], dim=-1).view(x.shape)


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

    def __init__(
        self,
        hidden: int = 512,
        depth: int = 12,
        group_size: int = 2,
        activation: str = "groupsort",
        input_encoding: str = "identity",
        multires: int = 6,
        pe_rho: float = 0.5,
    ) -> None:
        super().__init__()
        assert activation in ("groupsort", "nact"), f"unknown activation {activation!r}"
        assert input_encoding in ("identity", "neus", "lip_pe"), \
            f"unknown input_encoding {input_encoding!r}"
        if activation == "groupsort":
            assert hidden % group_size == 0
        self.hidden       = hidden
        self.depth        = depth
        self.group_size   = group_size
        self.activation   = activation
        self.architecture = "cpl"
        self.input_encoding = input_encoding
        self.multires = multires
        self.pe_rho   = pe_rho
        if input_encoding == "neus":
            self.encoder = NormalizedPositionalEncoding(multires=multires, input_dims=3)
            from .positonal_encoding import gamma_lipschitz_constant
            self.lip_scale = 1.0 / gamma_lipschitz_constant(multires)
            if self.encoder.out_dim > hidden:
                raise ValueError(
                    f"encoded dim {self.encoder.out_dim} exceeds hidden dim {hidden}"
                )
        elif input_encoding == "lip_pe":
            self.encoder = LipschitzPositionalEncoding(multires=multires, input_dims=3, rho=pe_rho)
            self.lip_scale = 1.0  # already 1-Lipschitz by construction
            if self.encoder.out_dim > hidden:
                raise ValueError(
                    f"encoded dim {self.encoder.out_dim} exceeds hidden dim {hidden}"
                )
        else:
            self.encoder = None
            self.lip_scale = 1.0
        blocks: list[nn.Module] = []
        for i in range(depth):
            blocks.append(ConvexPotentialLayer(hidden))
            if i < depth - 1:
                if activation == "nact":
                    blocks.append(NActivation(hidden))
                elif group_size == 2:
                    blocks.append(MaxMin())
                else:
                    blocks.append(GroupSort(group_size))
        self.net        = nn.Sequential(*blocks)
        self.head_weight = nn.Parameter(torch.empty(hidden))
        self.head_bias   = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.head_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.register_buffer("_head_w_buf",
                                 self.head_weight / self.head_weight.norm().clamp(min=1e-6))

    def forward(self, x: Tensor) -> Tensor:
        if self.encoder is None:
            h = F.pad(x, (0, self.hidden - x.shape[-1]))
        else:
            h = F.pad(self.encoder(x), (0, self.hidden - self.encoder.out_dim))
        h = self.net(h)
        if torch.is_grad_enabled():
            w = self.head_weight / torch.linalg.vector_norm(self.head_weight).clamp(min=1e-6)
            self._head_w_buf.copy_(w.detach())
        else:
            w = self._head_w_buf
        return (h * w).sum(-1) + self.head_bias.squeeze(-1)

    def sdf(self, x: Tensor) -> Tensor:
        """Metric SDF value. For NeuS PE this is the raw network output divided by K."""
        return self.forward(x) * self.lip_scale


class RegularMLP(nn.Module):
    """Unconstrained MLP baseline: same interface as FTheta, no Lipschitz guarantee."""

    def __init__(self, hidden: int = 256, depth: int = 8,
                 input_encoding: str = "identity", multires: int = 6,
                 pe_rho: float = 0.5) -> None:
        super().__init__()
        self.hidden         = hidden
        self.depth          = depth
        self.input_encoding = input_encoding
        self.multires       = multires
        self.lip_scale      = 1.0
        self.group_size     = 2
        self.activation     = "relu"
        self.architecture   = "mlp"
        self.pe_rho         = pe_rho
        if input_encoding == "neus":
            self.encoder = NormalizedPositionalEncoding(multires=multires, input_dims=3)
        elif input_encoding == "lip_pe":
            self.encoder = LipschitzPositionalEncoding(multires=multires, input_dims=3, rho=pe_rho)
        else:
            self.encoder = None
        in_dim = self.encoder.out_dim if self.encoder is not None else 3
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x) if self.encoder is not None else x
        return self.net(h).squeeze(-1)

    def sdf(self, x: Tensor) -> Tensor:
        return self.forward(x)


def make_model(hidden: int, depth: int, group_size: int = 2,
               activation: str = "groupsort", input_encoding: str = "identity",
               multires: int = 6, pe_rho: float = 0.5,
               architecture: str = "cpl") -> "FTheta | RegularMLP":
    if architecture == "mlp":
        return RegularMLP(hidden=hidden, depth=depth,
                          input_encoding=input_encoding, multires=multires, pe_rho=pe_rho)
    return FTheta(hidden=hidden, depth=depth, group_size=group_size,
                  activation=activation, input_encoding=input_encoding,
                  multires=multires, pe_rho=pe_rho)
