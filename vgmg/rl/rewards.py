"""Reward functions mapping a SMILES string to a scalar in [0, 1] (higher = better)."""

from __future__ import annotations

from typing import Callable

from vgmg.utils.rdkit_utils import is_valid, qed_score, sa_score

RewardFn = Callable[[str], float]


def sa_reward(smiles: str) -> float:
    """Synthetic accessibility, rescaled so 1 = easiest: ``(10 - SA) / 9``. Invalid -> 0."""
    return (10.0 - sa_score(smiles)) / 9.0 if is_valid(smiles) else 0.0


def qed_reward(smiles: str) -> float:
    """Drug-likeness (QED). Invalid -> 0."""
    return qed_score(smiles) if is_valid(smiles) else 0.0


def docking_proxy_reward(smiles: str) -> float:
    """Docking-score proxy. TODO(week 5): choose a proxy (e.g. a learned surrogate) and document it."""
    raise NotImplementedError("docking_proxy reward is not implemented yet")


REWARDS: dict[str, RewardFn] = {
    "sa": sa_reward,
    "qed": qed_reward,
    "docking_proxy": docking_proxy_reward,
}


def get_reward(name: str) -> RewardFn:
    """Look up a reward function by config name."""
    if name not in REWARDS:
        raise KeyError(f"Unknown reward {name!r}; choose from {sorted(REWARDS)}")
    return REWARDS[name]
