"""Baselines. B1 unguided; B2 best-of-N (compute-matched in candidates); random selection (control)."""

from __future__ import annotations

import torch

from vgmg.models.denoiser import Denoiser
from vgmg.rl.rewards import RewardFn
from vgmg.rl.svdd import svdd_sample


def sample_unguided(denoiser: Denoiser, n: int, generator: torch.Generator) -> list[str]:
    """B1: plain DiGress samples (SVDD with m=1)."""
    return denoiser.decode(svdd_sample(denoiser, None, n, 1, generator))


def sample_best_of_n(denoiser: Denoiser, reward_fn: RewardFn, n: int, m: int,
                     generator: torch.Generator) -> list[str]:
    """B2: for each of ``n`` slots draw ``m`` independent unguided molecules, keep the highest reward.

    Uses the *true* reward on finished molecules (an oracle selector), so it is a strong
    baseline. Ties go to the first draw. Costs ``n*m`` full chains, i.e. ``n*m*T`` forward passes.
    """
    mols = denoiser.decode(svdd_sample(denoiser, None, n * m, 1, generator))
    rewards = torch.tensor([reward_fn(s) for s in mols]).view(n, m)
    best = rewards.argmax(dim=1)
    return [mols[i * m + int(best[i])] for i in range(n)]


def sample_random_selection(denoiser: Denoiser, n: int, m: int, generator: torch.Generator) -> list[str]:
    """Control: draw ``m`` children per step but keep a random one (guidance signal removed)."""
    return denoiser.decode(svdd_sample(denoiser, None, n, m, generator))
