"""Value functions v(x_t, t) ~ E[reward(x_0) | x_t], used to rank SVDD candidates."""

from __future__ import annotations

import torch
from torch import nn


N_TIME_FREQ = 4


def time_features(t_frac: torch.Tensor, n_freq: int = N_TIME_FREQ) -> torch.Tensor:
    """Timestep embedding ``[t, sin(2*pi*k*t), cos(2*pi*k*t)]``, k = 1..n_freq, from ``t/T`` of shape ``(B,)``."""
    k = torch.arange(1, n_freq + 1, dtype=t_frac.dtype, device=t_frac.device)
    ang = 2 * torch.pi * t_frac.unsqueeze(-1) * k
    return torch.cat([t_frac.unsqueeze(-1), ang.sin(), ang.cos()], dim=-1)


class MLPValueFunction(nn.Module):
    """MLP on state features plus an embedding of the normalised timestep ``t/T``.

    The timestep is an input because the same partial graph means different things at
    different noise levels (early: reward almost undetermined; late: nearly decided).
    """

    feature_kind = "histogram"

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_hidden: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim + 1 + 2 * N_TIME_FREQ
        for _ in range(num_hidden):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_dim)`` features and ``(B,)`` (or scalar) ``t/T`` to ``(B,)`` predicted rewards."""
        t = torch.as_tensor(t_frac, dtype=x.dtype, device=x.device).expand(x.shape[0])
        return self.net(torch.cat([x, time_features(t)], dim=-1)).squeeze(-1)


class GNNValueFunction(nn.Module):
    """3-layer message-passing value function on the dense graph state (proposal arm of ablation A2).

    Input is the flat vector of ``graph_features`` (atom one-hots ``X``, bond one-hots ``E``, node mask), so noisy
    states are used as they are: soft information is limited to what the discrete state carries. One layer is
    ``h_i <- ReLU(W_self h_i + sum_c A_c (h W_c))_i`` with ``A_c`` the 0/1 adjacency of bond class ``c`` (class 0 = "no
    bond" is a message channel too, so the network can see non-bonds), with a residual connection. The readout concatenates masked mean- and sum-pooling of ``h`` with the timestep embedding and
    feeds a 2-layer MLP. Go/no-go Nov 20: dropped if it does not beat the MLP on validation MSE (proposal risk (ii)).
    """

    feature_kind = "graph"

    def __init__(self, n_nodes: int, dx: int, de: int, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.n, self.dx, self.de = n_nodes, dx, de
        self.embed = nn.Linear(dx, hidden_dim)
        self.self_w = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.edge_w = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim * de, bias=False) for _ in range(num_layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim + 1 + 2 * N_TIME_FREQ, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))
        self.hidden = hidden_dim

    def unpack(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the flat state into ``X (B,n,dx)``, ``E (B,n,n,de)`` and ``mask (B,n)``."""
        b, n, dx, de = flat.shape[0], self.n, self.dx, self.de
        X = flat[:, :n * dx].reshape(b, n, dx)
        E = flat[:, n * dx:n * dx + n * n * de].reshape(b, n, n, de)
        return X, E, flat[:, n * dx + n * n * de:]

    def forward(self, x: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        """Map ``(B, n*dx + n*n*de + n)`` graph features and ``(B,)`` (or scalar) ``t/T`` to ``(B,)`` predicted rewards."""
        X, E, mask = self.unpack(x)
        h = torch.relu(self.embed(X)) * mask.unsqueeze(-1)
        for self_w, edge_w in zip(self.self_w, self.edge_w):
            hw = edge_w(h).reshape(h.shape[0], self.n, self.de, self.hidden)               # per-class transform of each node
            msg = torch.einsum("bijc,bjch->bih", E, hw)                                    # sum over neighbours per bond class
            h = (h + torch.relu(self_w(h) + msg)) * mask.unsqueeze(-1)
        n_real = mask.sum(1, keepdim=True).clamp(min=1)
        pooled = torch.cat([h.sum(1) / n_real, h.sum(1) / self.n], dim=-1)
        t = torch.as_tensor(t_frac, dtype=x.dtype, device=x.device).expand(x.shape[0])
        return self.head(torch.cat([pooled, time_features(t)], dim=-1)).squeeze(-1)


def build_value_function(arch: str, in_dim: int, hidden_dim: int = 128, num_hidden: int = 2,
                         graph_shape: tuple[int, int, int] | None = None) -> nn.Module:
    """Construct a value function from the ``svdd.value_fn_arch`` config key.

    ``graph_shape = (max_nodes, atom_classes, bond_classes)`` of the denoiser's state is needed by the GNN only.
    For the GNN, ``hidden_dim`` is the message width and ``num_hidden`` the number of message-passing layers.
    """
    if arch == "mlp":
        return MLPValueFunction(in_dim, hidden_dim, num_hidden)
    if arch == "gnn":
        if graph_shape is None:
            raise ValueError("the GNN value function needs a denoiser with graph states (DiGress), not the toy")
        return GNNValueFunction(*graph_shape, hidden_dim=hidden_dim, num_layers=num_hidden)
    raise KeyError(f"Unknown value_fn_arch {arch!r}")
