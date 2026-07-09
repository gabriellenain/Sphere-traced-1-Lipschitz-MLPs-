"""1-Lipschitz SDF network: CPL layers with MaxMin / GroupSort / NActivation."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lip_tracer.n_activation import NActivation
from lip_tracer.positional_encoding import PositionalEncoding


class ConvexPotentialLayer(nn.Module):
    """CPL block: l(x) = x − (2/‖W‖²) Wᵀ act(Wx + b)."""

    def __init__(self, dim: int, activation: str = "relu") -> None:
        super().__init__()
        assert activation in ("relu", "softplus", "softmax"), f"unknown CPL activation {activation!r}"
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        nn.init.orthogonal_(self.weight)
        self.register_buffer("_u", F.normalize(torch.randn(dim), dim=0, eps=1e-12))
        self.register_buffer("_sigma_sq_buf", torch.ones(1))

    def _sigma_sq(self, update_u: bool = True) -> Tensor:
        if not update_u:
            return self._sigma_sq_buf
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
        if self.activation == "softplus":
            y = F.softplus(y, beta=100)
        elif self.activation == "softmax":
            y = F.softmax(y, dim=-1)
        else:
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


class CenteredSoftplus(nn.Module):
    """NeuS Softplus(beta=100), shifted so activation(0) == 0."""

    def __init__(self, beta: float = 100.0) -> None:
        super().__init__()
        self.beta = beta
        self.offset = math.log(2.0) / beta
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, x: Tensor) -> Tensor:
        return self.softplus(x) - self.offset


class FTheta(nn.Module):
    """1-Lipschitz SDF network: CPL stack with GroupSort activation.

    group_size=2  → MaxMin (default, backward compatible with old checkpoints).
    group_size>2  → GroupSort-N (more expressive, new checkpoints only).
    activation="softplus" swaps the inter-layer nonlinearity for NeuS'
    nn.Softplus(beta=100), intentionally breaking the 1-Lipschitz guarantee.
    activation="centered_softplus" uses Softplus_beta(z) - log(2)/beta.
    activation="softplus_cpl" also replaces the ReLU inside each CPL block.
    activation="softmax_cpl" uses softmax as the ReLU replacement inside each
    CPL block while keeping MaxMin (GroupSort-2) as the inter-layer activation.
    activation="softplus_cpl_maxmin" uses Softplus(beta=100) inside each CPL
    block (like softplus_cpl) but keeps MaxMin (GroupSort-2) between layers.
    """

    def __init__(
        self,
        hidden: int = 512,
        depth: int = 12,
        group_size: int = 2,
        activation: str = "groupsort",
        input_encoding: str = "identity",
        multires: int = 6,
        lipschitz_mode: str = "none",
    ) -> None:
        super().__init__()
        assert activation in ("groupsort", "nact", "softplus", "centered_softplus", "softplus_cpl", "softmax_cpl", "softplus_cpl_maxmin"), \
            f"unknown activation {activation!r}"
        assert input_encoding in ("identity", "pe"), \
            f"unknown input_encoding {input_encoding!r}"
        assert lipschitz_mode in ("none", "uniform", "per_band"), \
            f"unknown lipschitz_mode {lipschitz_mode!r}"
        if activation in ("groupsort", "softmax_cpl", "softplus_cpl_maxmin"):
            assert hidden % group_size == 0
        self.hidden         = hidden
        self.depth          = depth
        self.group_size     = group_size
        self.activation     = activation
        self.architecture   = "cpl"
        self.input_encoding = input_encoding
        self.multires       = multires
        self.lipschitz_mode = lipschitz_mode
        if input_encoding == "pe":
            self.encoder = PositionalEncoding(
                multires=multires, input_dims=3,
                lipschitz_mode=None if lipschitz_mode == "none" else lipschitz_mode,
            )
            if self.encoder.out_dim > hidden:
                raise ValueError(
                    f"encoded dim {self.encoder.out_dim} exceeds hidden dim {hidden}"
                )
        else:
            self.encoder = None
        blocks: list[nn.Module] = []
        if activation in ("softplus_cpl", "softplus_cpl_maxmin"):
            cpl_activation = "softplus"
        elif activation == "softmax_cpl":
            cpl_activation = "softmax"
        else:
            cpl_activation = "relu"
        for i in range(depth):
            blocks.append(ConvexPotentialLayer(hidden, activation=cpl_activation))
            if i < depth - 1:
                if activation == "nact":
                    # Uniform (-1,0) init + lr_factor=1.0. The paper's AbsId init +
                    # lr_factor=0.1 (arXiv:2311.06103) are tuned for certified-robust
                    # *classification*; on this signed-distance task AbsId is harmful
                    # (half the channels become sign-destroying |x|, half pure
                    # identity), pushing |∇f| below 1 and biasing the field positive.
                    blocks.append(NActivation(hidden))
                elif activation == "softplus":
                    blocks.append(nn.Softplus(beta=100))
                elif activation == "centered_softplus":
                    blocks.append(CenteredSoftplus(beta=100))
                elif activation == "softplus_cpl":
                    blocks.append(nn.Softplus(beta=100))
                elif group_size == 2:
                    blocks.append(MaxMin())
                else:
                    blocks.append(GroupSort(group_size))
        self.net         = nn.Sequential(*blocks)
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
        return self.forward(x)

    @torch.no_grad()
    def geometric_init(self, radius: float = 0.5, n_samples: int = 16384,
                       seed: int = 1234) -> None:
        """Closed-form SAL/IGR-style sphere init: set f ≈ ‖x‖ − radius with no SGD.

        SAL's original trick (a wide ReLU layer whose units are *averaged* by an
        output weight √(π/n), using E_w[relu(w·x)] ∝ ‖x‖) needs an expansive,
        √n-Lipschitz intermediate map — structurally impossible here, since every
        CPL block and MaxMin layer is 1-Lipschitz. The 1-Lipschitz-compatible route
        to ‖x‖ is the *max* identity ‖x‖ = maxᵤ ⟨u,x⟩ (a max of 1-Lipschitz maps is
        1-Lipschitz), which the random CPL stack + GroupSort already realises: with
        random 1-Lipschitz weights the frozen features h(x) = net(pad(x)) move at
        ~unit speed and span many directions, so a *single* unit-norm head can be
        aligned with the radial direction. We leave the CPL blocks at their standard
        random 1-Lipschitz init (the "random directions") and solve only the head
        weight + bias in closed form (one least-squares over points in a ball) so
        that ⟨h(x), w⟩ + b ≈ ‖x‖ − radius.

        The head weight the network applies is unit-normalised, so the raw solution
        norm ‖w‖ is a diagnostic: ‖w‖ ≈ 1 ⇔ the field is eikonal (|∇f| ≈ 1) and the
        sphere is faithful. It only holds for the genuinely 1-Lipschitz,
        identity-encoding model (activation ∈ {groupsort, nact}); the intentionally
        non-1-Lipschitz softplus/softmax CPL variants and PE encoding dilute the
        radial direction and this closed form degrades (‖w‖ strays from 1) — use the
        SGD fit_sphere_init for those instead.
        """
        device = self.head_weight.device
        bound  = radius * 1.5   # match fit_sphere_init's sampling box
        g      = torch.Generator(device="cpu").manual_seed(seed)
        x      = ((2 * torch.rand(n_samples, 3, generator=g) - 1) * bound).to(device)
        if self.encoder is None:
            h = F.pad(x, (0, self.hidden - 3))
        else:
            h = F.pad(self.encoder(x), (0, self.hidden - self.encoder.out_dim))
        h      = self.net(h)                             # frozen 1-Lipschitz features
        target = x.norm(dim=-1) - radius
        A      = torch.cat([h, torch.ones(n_samples, 1, device=device)], dim=-1)
        sol    = torch.linalg.lstsq(A, target).solution
        w, b   = sol[:-1], sol[-1]
        wn     = w.norm().clamp(min=1e-6)
        self.head_weight.copy_(w)
        self._head_w_buf.copy_(w / wn)
        self.head_bias.copy_(b.reshape(1))
        f0     = self.forward(torch.zeros(1, 3, device=device)).item()
        note   = ""
        if not (0.7 < wn.item() < 1.4):
            note = ("  [WARN] ‖w‖ far from 1: this activation/encoding is not cleanly "
                    "1-Lipschitz-radial; the closed-form sphere is unreliable — prefer "
                    "SGD fit_sphere_init")
        print(f"  cpl geometric_init  r={radius:.3f}  ‖w‖={wn.item():.3f} (~1=eikonal)  "
              f"f(0)={f0:.4f} (target={-radius:.4f}){note}")


class NeuSMLP(nn.Module):
    """NeuS-style MLP: 8-layer Softplus network with skip connection at layer 4.

    Architecture follows Wang et al. 2021 (NeuS):
      - PE input encoding
      - Softplus(β=100) activations
      - Skip: encoded input concatenated back at layer skip_layer
      - Single scalar SDF output (no colour head — colour is handled separately)
    """

    def __init__(self, hidden: int = 256, depth: int = 8, skip_layer: int = 4,
                 input_encoding: str = "pe", multires: int = 6,
                 lipschitz_mode: str = "none",
                 beta: float = 100.0) -> None:
        super().__init__()
        assert lipschitz_mode in ("none", "uniform", "per_band"), \
            f"unknown lipschitz_mode {lipschitz_mode!r}"
        self.hidden         = hidden
        self.depth          = depth
        self.skip_layer     = skip_layer
        self.input_encoding = input_encoding
        self.multires       = multires
        self.lipschitz_mode = lipschitz_mode
        self.group_size     = 2
        self.activation     = "softplus"
        self.architecture   = "neus"
        self.encoder = PositionalEncoding(
            multires=multires, input_dims=3,
            lipschitz_mode=None if lipschitz_mode == "none" else lipschitz_mode,
        ) if input_encoding == "pe" else None
        in_dim = self.encoder.out_dim if self.encoder is not None else 3
        self.in_dim = in_dim

        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden))
            elif i == skip_layer:
                self.layers.append(nn.Linear(hidden + in_dim, hidden))
            else:
                self.layers.append(nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, 1)
        self.act = nn.Softplus(beta=beta)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x) if self.encoder is not None else x
        feat = h
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                feat = torch.cat([feat, h], dim=-1)
            feat = self.act(layer(feat))
        return self.out(feat).squeeze(-1)

    def sdf(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def geometric_init(self, radius: float = 0.5) -> None:
        """IGR/SAL/IDR geometric initialization: start f as the signed distance to
        a sphere of the given radius (negative inside, positive outside). Breaks the
        inside/outside sign ambiguity of an unoriented point cloud so DiffCD/eikonal
        fitting converges to a signed field instead of the unsigned distance.

        The first (and skip) layer zero out the positional-encoding channels so the
        field starts as a pure function of raw xyz; PE detail grows during training.
        """
        import numpy as np
        d_in = 3  # raw xyz channels sit first in the PE (identity block)
        for i, lin in enumerate(self.layers):
            out_dim = lin.weight.shape[0]
            if self.encoder is not None and i == 0:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.constant_(lin.weight[:, d_in:], 0.0)              # zero PE
                nn.init.normal_(lin.weight[:, :d_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif self.encoder is not None and i == self.skip_layer:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                nn.init.constant_(lin.weight[:, -(self.in_dim - d_in):], 0.0)  # zero skip-PE
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
        # output layer: f(x) ~= ||x|| - radius
        nn.init.normal_(self.out.weight, mean=np.sqrt(np.pi) / np.sqrt(self.hidden),
                        std=1e-4)
        nn.init.constant_(self.out.bias, -radius)


class RadianceNet(nn.Module):
    """Small IDR-style view-dependent colour MLP: (x, n, v) → RGB in [0,1].

    Deliberately *not* Lipschitz-constrained — appearance has no 1-Lipschitz
    prior. Trained with an L1 rendering loss against the observed per-ray pixel
    colour; gradient flows RGB → x_θ → SDF, so it also lightly sharpens geometry
    (à la Yariv et al. 2020, IDR).

    view_dep=False drops the view direction (pure diffuse albedo).

    input_encoding="pe" applies Fourier positional encoding to the position x
    (NeRF/IDR convention: encode position only, keep n and v raw). Unlike the SDF
    net this head is *not* 1-Lipschitz, so raw (lipschitz_mode=None) PE is used —
    no gradient-budget rescaling needed. This lets the colour MLP fit
    high-frequency appearance; whether that helps geometry is an empirical
    tradeoff (sharper depth cue vs. more appearance capacity to absorb error).
    """

    def __init__(self, hidden: int = 256, depth: int = 3,
                 view_dep: bool = True, input_encoding: str = "identity",
                 multires: int = 6) -> None:
        super().__init__()
        assert input_encoding in ("identity", "pe"), \
            f"unknown input_encoding {input_encoding!r}"
        self.view_dep       = view_dep
        self.input_encoding = input_encoding
        self.multires       = multires
        self.encoder = (PositionalEncoding(multires=multires, input_dims=3,
                                            lipschitz_mode=None)
                        if input_encoding == "pe" else None)
        x_dim = self.encoder.out_dim if self.encoder is not None else 3
        in_dim = x_dim + 3 + (3 if view_dep else 0)   # γ(x) + n(3) [+ v(3)]
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers.append(nn.Linear(d, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, n: Tensor, v: Tensor) -> Tensor:
        if self.encoder is not None:
            x = self.encoder(x)
        feats = [x, n, v] if self.view_dep else [x, n]
        return torch.sigmoid(self.net(torch.cat(feats, dim=-1)))


def make_model(hidden: int, depth: int, group_size: int = 2,
               activation: str = "groupsort", input_encoding: str = "identity",
               multires: int = 6, architecture: str = "cpl",
               lipschitz_mode: str = "none") -> "FTheta | NeuSMLP":
    if architecture == "neus":
        return NeuSMLP(hidden=hidden, depth=depth,
                       input_encoding=input_encoding, multires=multires,
                       lipschitz_mode=lipschitz_mode)
    return FTheta(hidden=hidden, depth=depth, group_size=group_size,
                  activation=activation, input_encoding=input_encoding,
                  multires=multires, lipschitz_mode=lipschitz_mode)
