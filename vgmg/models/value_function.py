"""Value functions v(x_t, t) ~ E[reward(x_0) | x_t], used to rank SVDD candidates."""

from __future__ import annotations

import torch
from torch import nn


class MLPValueFunction(nn.Module):
    """MLP on state features plus the normalised timestep ``t/T``.

    The timestep is an input because the same partial graph means different things at
    different noise levels (early: reward almost undetermined; late: nearly decided).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_hidden: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim + 1
        for _ in range(num_hidden):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_dim)`` features and ``(B,)`` (or scalar) ``t/T`` to ``(B,)`` predicted rewards."""
        t = torch.as_tensor(t_frac, dtype=x.dtype, device=x.device).expand(x.shape[0]).unsqueeze(-1)
        return self.net(torch.cat([x, t], dim=-1)).squeeze(-1)


class GNNValueFunction(nn.Module):
    """Message-passing value function on DiGress graph states (stretch goal T3).

    TODO: only after tier T2 is complete (see docs/PLAN.md). Needs the DiGress state format.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("GNNValueFunction is a tier-3 item; not implemented")


def build_value_function(arch: str, in_dim: int, hidden_dim: int = 128, num_hidden: int = 2) -> nn.Module:
    """Construct a value function from the ``svdd.value_fn_arch`` config key."""
    if arch == "mlp":
        return MLPValueFunction(in_dim, hidden_dim, num_hidden)
    if arch == "gnn":
        return GNNValueFunction()
    raise KeyError(f"Unknown value_fn_arch {arch!r}")
