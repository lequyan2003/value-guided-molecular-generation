"""Batched denoiser interface used by every sampler, plus a smoke-test stand-in.

All samplers (unguided, best-of-N, SVDD) talk to a generator only through ``Denoiser``.
States are opaque *batches* (for DiGress: dense X/E tensors; for the toy: a list of
SMILES). Children are grouped by parent: child ``j`` of parent ``i`` sits at row ``i*m+j``.

Cost accounting (this is what makes the compute comparison honest): ``nfe`` counts
network forward passes, one per graph per call that needs the network. ``propose`` costs
one forward pass per *parent* (the network output is shared by all ``m`` children);
``predict_x0`` costs one forward pass per graph it is given.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
import torch

from vgmg.constants import FINGERPRINT_BITS, SMOKE_SMILES
from vgmg.utils.rdkit_utils import morgan_fingerprint

State = Any  # opaque batch of graph states


class Denoiser(Protocol):
    """Interface a generator must satisfy to be guided by SVDD."""

    num_steps: int      # T; reverse steps run t = T, T-1, ..., 1
    feature_dim: int    # width of ``features`` output
    nfe: int            # network forward passes since construction / last reset

    def init(self, n: int, generator: torch.Generator) -> State:
        """``n`` fully noised states x_T."""

    def propose(self, state: State, t: int, m: int, generator: torch.Generator) -> State:
        """For each of the ``n`` parents x_t draw ``m`` i.i.d. x_{t-1} ~ p(. | x_t); returns ``n*m`` states."""

    def take(self, state: State, idx: torch.Tensor) -> State:
        """Sub-batch of ``state`` at row indices ``idx`` (1-D LongTensor)."""

    def features(self, state: State) -> torch.Tensor:
        """Value-function input, ``(B, feature_dim)``. Does NOT include the timestep."""

    def predict_x0(self, state: State, t: int) -> list[str]:
        """Decode the denoiser's own clean-graph prediction for states at time ``t`` (costs a forward pass)."""

    def decode(self, state: State) -> list[str]:
        """Decode final states x_0 to SMILES (invalid -> a string that fails validity checks)."""

    def encode(self, smiles: Sequence[str]) -> State:
        """Inverse of ``decode`` for real molecules (clean graphs at t = 0); used for the t=0 sanity check."""


class ToyDenoiser:
    """Smoke-test stand-in for DiGress. NOT a real generative model.

    States are SMILES strings from a fixed pool. Each reverse step keeps the current
    molecule or replaces it with a random pool member (50/50), so the chain has a
    well-defined, seedable p(x_{t-1} | x_t) that guidance can be exercised against.
    """

    feature_dim = FINGERPRINT_BITS

    def __init__(self, num_steps: int = 4, pool: Sequence[str] = SMOKE_SMILES) -> None:
        self.num_steps = num_steps
        self.pool = list(pool)
        self.nfe = 0

    def _draw(self, k: int, generator: torch.Generator) -> list[str]:
        idx = torch.randint(len(self.pool), (k,), generator=generator)
        return [self.pool[i] for i in idx.tolist()]

    def init(self, n: int, generator: torch.Generator) -> list[str]:
        """Random pool members as x_T."""
        return self._draw(n, generator)

    def propose(self, state: list[str], t: int, m: int, generator: torch.Generator) -> list[str]:
        """``m`` candidates per parent: keep with prob 0.5, else a random pool member."""
        self.nfe += len(state)
        fresh = self._draw(len(state) * m, generator)
        keep = torch.rand(len(state) * m, generator=generator) < 0.5
        return [state[k // m] if keep[k] else fresh[k] for k in range(len(state) * m)]

    def take(self, state: list[str], idx: torch.Tensor) -> list[str]:
        """Sub-batch by index."""
        return [state[i] for i in idx.tolist()]

    def features(self, state: list[str]) -> torch.Tensor:
        """Morgan fingerprints (the toy states are real molecules)."""
        return torch.from_numpy(
            np.stack([morgan_fingerprint(s, FINGERPRINT_BITS) for s in state])
        )

    def predict_x0(self, state: list[str], t: int) -> list[str]:
        """The toy's best guess of x_0 is the current molecule."""
        self.nfe += len(state)
        return list(state)

    def decode(self, state: list[str]) -> list[str]:
        """States are already SMILES."""
        return list(state)

    def encode(self, smiles: Sequence[str]) -> list[str]:
        """States are already SMILES."""
        return list(smiles)
