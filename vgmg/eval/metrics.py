"""Evaluation metrics for a set of generated molecules."""

from typing import Callable, Collection

import numpy as np

from vgmg.utils.rdkit_utils import canonicalize, is_valid, tanimoto_diversity


def compute_metrics(smiles: list[str], reward_fn: Callable[[str], float],
                    train_smiles: Collection[str] = ()) -> dict[str, float]:
    """Validity, uniqueness, novelty, reward mean/std (over valid molecules), diversity.

    Novelty is the fraction of unique valid molecules whose canonical SMILES is absent
    from ``train_smiles`` (canonicalized here). Invalid molecules are excluded from
    reward statistics so reward is not confounded with validity.
    """
    valid = [s for s in smiles if is_valid(s)]
    canon = {canonicalize(s) for s in valid}
    train = {canonicalize(s) for s in train_smiles} - {None}
    rewards = np.array([reward_fn(s) for s in valid]) if valid else np.array([0.0])
    return {
        "n": float(len(smiles)),
        "validity": len(valid) / max(len(smiles), 1),
        "uniqueness": len(canon) / max(len(valid), 1),
        "novelty": len(canon - train) / max(len(canon), 1),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "diversity": tanimoto_diversity(valid),
    }
