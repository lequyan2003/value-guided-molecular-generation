"""Evaluation metrics. One function evaluates every method, so no method gets different preprocessing."""

from __future__ import annotations

from typing import Callable, Collection

import numpy as np

from vgmg.utils.rdkit_utils import canonicalize, is_valid, tanimoto_diversity


def per_molecule_rewards(smiles: list[str], reward_fn: Callable[[str], float]) -> np.ndarray:
    """Reward of every generated molecule; invalid (unparsable or disconnected) molecules get 0."""
    return np.array([reward_fn(s) if is_valid(s) else 0.0 for s in smiles], dtype=np.float64)


def compute_metrics(smiles: list[str], reward_fn: Callable[[str], float],
                    reference_smiles: Collection[str] = ()) -> dict[str, float]:
    """Validity, uniqueness, novelty, reward statistics, diversity.

    * ``validity``: parsable AND connected, over all N generated.
    * ``reward_mean`` (PRIMARY): mean over all N with invalid = 0, so a method cannot raise its
      reward by producing fewer valid molecules. ``reward_top10``: mean of the best 10% of the N.
      ``reward_mean_valid`` is reported as secondary context only.
    * ``uniqueness``: unique canonical SMILES / valid. ``novelty``: unique valid molecules absent
      from ``reference_smiles`` (the full QM9 set, canonicalised here; stricter than DiGress's own
      train-split novelty). ``diversity``: 1 - mean pairwise ECFP4 Tanimoto over valid molecules.
    """
    valid = [s for s in smiles if is_valid(s)]
    canon = {canonicalize(s) for s in valid}
    ref = {canonicalize(s) for s in reference_smiles} - {None}
    r_all = per_molecule_rewards(smiles, reward_fn)
    r_valid = r_all[[is_valid(s) for s in smiles]] if valid else np.array([0.0])
    k = max(1, len(r_all) // 10)
    return {
        "n": float(len(smiles)),
        "validity": len(valid) / max(len(smiles), 1),
        "uniqueness": len(canon) / max(len(valid), 1),
        "novelty": len(canon - ref) / max(len(canon), 1),
        "reward_mean": float(r_all.mean()) if len(r_all) else 0.0,
        "reward_std": float(r_all.std()) if len(r_all) else 0.0,
        "reward_top10": float(np.sort(r_all)[-k:].mean()) if len(r_all) else 0.0,
        "reward_mean_valid": float(r_valid.mean()),
        "diversity": tanimoto_diversity(valid),
    }
