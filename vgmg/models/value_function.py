"""Value functions v(x_t) ~ E[reward(x_0) | x_t], used to rank SVDD candidates."""

import torch
from torch import nn


class MLPValueFunction(nn.Module):
    """MLP on a fixed-size feature vector (e.g. a Morgan fingerprint)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_hidden: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_hidden):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_dim)`` features to ``(B,)`` predicted rewards."""
        return self.net(x).squeeze(-1)


class GNNValueFunction(nn.Module):
    """Graph-based value function operating on DiGress graph states.

    TODO(week 3): implement once the DiGress state format (X, E, y, t) is fixed.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("GNNValueFunction is not implemented yet")


def build_value_function(arch: str, in_dim: int) -> nn.Module:
    """Construct a value function from the ``svdd.value_fn_arch`` config key."""
    if arch == "mlp":
        return MLPValueFunction(in_dim)
    if arch == "gnn":
        return GNNValueFunction()
    raise KeyError(f"Unknown value_fn_arch {arch!r}")
