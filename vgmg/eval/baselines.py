"""Baselines: unconditional sampling (M=1) and random selection among M candidates."""

import torch

from vgmg.models.digress import Denoiser
from vgmg.rl.svdd import svdd_sample


def sample_unconditional(denoiser: Denoiser, n: int, generator: torch.Generator) -> list[str]:
    """Unguided generator samples (SVDD with M=1)."""
    states = svdd_sample(denoiser, None, lambda s: torch.zeros(1), n, 1, generator)
    return [denoiser.decode(s) for s in states]


def sample_random_selection(denoiser: Denoiser, n: int, m: int,
                            generator: torch.Generator) -> list[str]:
    """Control for SVDD: draw M candidates per step but pick uniformly at random."""
    states = svdd_sample(denoiser, None, lambda s: torch.zeros(1), n, m, generator)
    return [denoiser.decode(s) for s in states]
