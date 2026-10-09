"""Baselines. B1 unguided; B2 best-of-N (compute-matched in candidates); random selection (control)."""

from __future__ import annotations

import torch

from vgmg.models.denoiser import Denoiser
from vgmg.rl.rewards import RewardFn
from vgmg.rl.svdd import svdd_sample


def sample_unguided(denoiser: Denoiser, n: int, generator: torch.Generator) -> list[str]:
    """B1: plain DiGress samples (SVDD with m=1)."""
    return denoiser.decode(svdd_sample(denoiser, None, n, 1, generator))


def select_best_of(pool: list[str], n: int, pool_m: int, m: int, reward_fn: RewardFn) -> list[str]:
    """Best-of-``m`` from a pool of ``n*pool_m`` independent unguided molecules laid out slot-major.

    Slot ``i`` owns ``pool[i*pool_m:(i+1)*pool_m]``; best-of-``m`` uses its first ``m`` draws, so
    best-of-5, -10, -20 can all be read off one pool of 20 draws (nested, hence correlated across m but each
    marginally an exact best-of-m). Ties go to the first draw; invalid molecules score 0.
    """
    assert m <= pool_m and len(pool) == n * pool_m
    out = []
    for i in range(n):
        draws = pool[i * pool_m: i * pool_m + m]
        scores = [reward_fn(x) for x in draws]
        out.append(draws[max(range(m), key=lambda j: (scores[j], -j))])
    return out


def sample_best_of_n(denoiser: Denoiser, reward_fn: RewardFn, n: int, m: int,
                     generator: torch.Generator) -> list[str]:
    """B2: for each of ``n`` slots draw ``m`` independent unguided molecules, keep the highest reward.

    Uses the *true* reward on finished molecules (an oracle selector), so it is a strong
    baseline. Costs ``n*m`` full chains, i.e. ``n*m*T`` forward passes.
    """
    pool = denoiser.decode(svdd_sample(denoiser, None, n * m, 1, generator))
    return select_best_of(pool, n, m, m, reward_fn)


def sample_random_selection(denoiser: Denoiser, n: int, m: int, generator: torch.Generator) -> list[str]:
    """Control: draw ``m`` children per step but keep a random one (guidance signal removed)."""
    return denoiser.decode(svdd_sample(denoiser, None, n, m, generator))
