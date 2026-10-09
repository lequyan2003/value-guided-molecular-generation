"""Reward functions return sensible values for simple molecules."""

from __future__ import annotations

import pytest

from vgmg.constants import SMOKE_SMILES
from vgmg.rl.rewards import get_reward, qed_reward, sa_reward


def test_smoke_pool_is_valid_and_rewards_in_range() -> None:
    for s in SMOKE_SMILES:
        assert 0.0 <= sa_reward(s) <= 1.0
        assert 0.0 <= qed_reward(s) <= 1.0
        assert sa_reward(s) > 0.0  # all pool molecules must parse


def test_invalid_smiles_gets_zero() -> None:
    assert sa_reward("not_a_smiles") == 0.0
    assert qed_reward("not_a_smiles") == 0.0


def test_sa_prefers_simple_over_complex() -> None:
    assert sa_reward("CCO") > sa_reward("CC12CC3CC(C1)CC(C3)(C2)C(=O)N")


def test_unknown_reward_raises() -> None:
    with pytest.raises(KeyError):
        get_reward("nope")
