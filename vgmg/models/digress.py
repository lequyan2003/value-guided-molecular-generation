"""Denoiser interface for SVDD, plus the DiGress wrapper and a smoke-test stand-in.

SVDD only needs a generator that can (a) start from noise, (b) propose M samples of
x_{t-1} given x_t, and (c) decode a final state to SMILES. ``Denoiser`` captures that.
"""

from typing import Any, Protocol, Sequence

import torch

from vgmg.constants import SMOKE_SMILES


class Denoiser(Protocol):
    """Interface a generator must satisfy to be guided by SVDD."""

    num_steps: int

    def init(self, n: int, generator: torch.Generator) -> list[Any]:
        """Return ``n`` initial (fully noised) states x_T."""

    def propose(self, states: Sequence[Any], t: int, m: int,
                generator: torch.Generator) -> list[list[Any]]:
        """For each state, draw ``m`` i.i.d. samples of x_{t-1} ~ p(. | x_t)."""

    def decode(self, state: Any) -> str:
        """Convert a (final) state to a SMILES string."""


class DiGressDenoiser:
    """Wrapper around the official pretrained DiGress checkpoint (inference only).

    TODO(week 1): load cvignac/DiGress checkpoint (log commit hash in CONTRIBUTIONS.md),
    expose one reverse step as ``propose``, and decode graphs to SMILES via RDKit.
    """

    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        raise NotImplementedError("DiGress wrapper is not implemented yet")


class ToyDenoiser:
    """Smoke-test stand-in for DiGress. NOT a real generative model.

    States are SMILES strings from a fixed pool. Each reverse step keeps the current
    molecule or replaces it with a random pool member (50/50), so the chain has a
    well-defined, seedable p(x_{t-1} | x_t) that SVDD can be exercised against.
    """

    def __init__(self, num_steps: int = 4, pool: Sequence[str] = SMOKE_SMILES) -> None:
        self.num_steps = num_steps
        self.pool = list(pool)

    def _draw(self, k: int, generator: torch.Generator) -> list[str]:
        idx = torch.randint(len(self.pool), (k,), generator=generator)
        return [self.pool[i] for i in idx.tolist()]

    def init(self, n: int, generator: torch.Generator) -> list[str]:
        """Random pool members as x_T."""
        return self._draw(n, generator)

    def propose(self, states: Sequence[str], t: int, m: int,
                generator: torch.Generator) -> list[list[str]]:
        """``m`` candidates per state: keep with prob 0.5, else a random pool member."""
        fresh = self._draw(len(states) * m, generator)
        keep = torch.rand(len(states) * m, generator=generator) < 0.5
        out: list[list[str]] = []
        for i, s in enumerate(states):
            out.append([s if keep[i * m + j] else fresh[i * m + j] for j in range(m)])
        return out

    def decode(self, state: str) -> str:
        """States are already SMILES."""
        return state
