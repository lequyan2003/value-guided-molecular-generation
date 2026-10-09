"""Core SVDD (Li et al., 2024): at each reverse step draw M candidates, rank by value, select.

With M=1 this reduces to unguided sampling, which is the baseline.
"""

from typing import Any, Callable, Literal

import torch
from torch import nn

from vgmg.models.digress import Denoiser

Featurizer = Callable[[Any], torch.Tensor]  # state -> 1-D feature tensor


@torch.no_grad()
def select_candidates(values: torch.Tensor, mode: Literal["argmax", "softmax"],
                      alpha: float, generator: torch.Generator) -> torch.Tensor:
    """Pick one candidate index per row of ``values`` (shape ``(N, M)``).

    ``argmax`` is greedy (SVDD-PM style); ``softmax`` samples from softmax(values / alpha).
    """
    if mode == "argmax":
        return values.argmax(dim=1)
    if mode == "softmax":
        return torch.multinomial(torch.softmax(values / alpha, dim=1), 1, generator=generator).squeeze(1)
    raise ValueError(f"Unknown selection mode {mode!r}")


@torch.no_grad()
def svdd_sample(denoiser: Denoiser, value_fn: nn.Module | None, featurize: Featurizer,
                n: int, m: int, generator: torch.Generator,
                mode: Literal["argmax", "softmax"] = "argmax", alpha: float = 0.1,
                device: str = "cpu") -> list[Any]:
    """Run SVDD and return ``n`` final states x_0.

    ``value_fn=None`` selects a random candidate (the random-selection baseline).
    """
    states = denoiser.init(n, generator)
    for t in range(denoiser.num_steps, 0, -1):
        cands = denoiser.propose(states, t, m, generator)  # n lists of m states
        if value_fn is None or m == 1:
            choice = torch.randint(m, (n,), generator=generator) if m > 1 else torch.zeros(n, dtype=torch.long)
        else:
            feats = torch.stack([featurize(c) for row in cands for c in row]).to(device)
            values = value_fn(feats).view(n, m).cpu()
            choice = select_candidates(values, mode, alpha, generator)
        states = [cands[i][int(choice[i])] for i in range(n)]
    return states
