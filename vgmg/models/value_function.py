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
    """Message-passing value function on DiGress graph states (proposal arm of ablation A2).

    TODO: Nov 2-8, needs the DiGress state format; dropped if it does not beat the MLP on validation MSE by Nov 6.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("GNNValueFunction not implemented yet (planned Nov 2-8)")


def build_value_function(arch: str, in_dim: int, hidden_dim: int = 128, num_hidden: int = 2) -> nn.Module:
    """Construct a value function from the ``svdd.value_fn_arch`` config key."""
    if arch == "mlp":
        return MLPValueFunction(in_dim, hidden_dim, num_hidden)
    if arch == "gnn":
        return GNNValueFunction()
    raise KeyError(f"Unknown value_fn_arch {arch!r}")
