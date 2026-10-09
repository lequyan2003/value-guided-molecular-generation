"""Deterministic preprocessing: canonicalization and seeded split-index generation."""

from __future__ import annotations

import numpy as np

from vgmg.utils.rdkit_utils import canonicalize


def canonicalize_all(smiles: list[str]) -> list[str]:
    """Canonicalize and drop invalid entries (order-preserving, idempotent)."""
    out = (canonicalize(s) for s in smiles)
    return [s for s in out if s is not None]


def make_split_indices(n_total: int, sizes: tuple[int, int, int], seed: int) -> tuple[np.ndarray, ...]:
    """Disjoint (train, val, test) index arrays of the given sizes, from a seeded permutation."""
    if sum(sizes) > n_total:
        raise ValueError("Requested split sizes exceed dataset size")
    perm = np.random.default_rng(seed).permutation(n_total)
    bounds = np.cumsum((0, *sizes))
    return tuple(np.sort(perm[bounds[i]:bounds[i + 1]]) for i in range(3))
