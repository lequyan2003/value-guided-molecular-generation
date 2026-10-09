"""Evaluation-correctness tests: the metric/baseline code does what the report will say it does."""

from __future__ import annotations

import numpy as np
import torch

from vgmg.eval.baselines import sample_best_of_n, sample_random_selection, sample_unguided
from vgmg.eval.metrics import compute_metrics
from vgmg.models.denoiser import ToyDenoiser
from vgmg.rl.rewards import qed_reward, sa_reward
from vgmg.rl.svdd import pm_scorer, svdd_sample, value_scorer
from vgmg.utils.rdkit_utils import is_valid
from vgmg.utils.reproducibility import seed_streams


def test_disconnected_and_garbage_are_invalid_with_zero_reward() -> None:
    for bad in ("CC.O", "not_a_smiles", ""):
        assert not is_valid(bad) and sa_reward(bad) == 0.0 and qed_reward(bad) == 0.0


def test_primary_reward_counts_invalid_as_zero() -> None:
    m = compute_metrics(["CCO", "garbage"], sa_reward)
    assert m["validity"] == 0.5
    assert abs(m["reward_mean"] - sa_reward("CCO") / 2) < 1e-9          # invalid counted as 0
    assert abs(m["reward_mean_valid"] - sa_reward("CCO")) < 1e-9        # secondary view


def test_novelty_uses_canonical_smiles() -> None:
    assert compute_metrics(["OCC"], sa_reward, reference_smiles=["CCO"])["novelty"] == 0.0
    assert compute_metrics(["CCCC"], sa_reward, reference_smiles=["CCO"])["novelty"] == 1.0


def test_metrics_identical_for_identical_molecules() -> None:
    """Same function, same inputs -> same numbers; no method-specific preprocessing exists."""
    a = compute_metrics(["CCO", "CCN"], sa_reward, ["C"])
    assert a == compute_metrics(["CCO", "CCN"], sa_reward, ["C"])


def test_compute_accounting() -> None:
    n, m, T = 8, 5, 4
    def nfe(f):
        d = ToyDenoiser(T); f(d); return d.nfe
    g = lambda: torch.Generator().manual_seed(0)  # noqa: E731
    assert nfe(lambda d: sample_unguided(d, n, g())) == n * T
    assert nfe(lambda d: sample_best_of_n(d, sa_reward, n, m, g())) == n * m * T
    assert nfe(lambda d: sample_random_selection(d, n, m, g())) == n * T          # candidates are free
    vf = lambda x, t: torch.zeros(len(x))  # noqa: E731
    assert nfe(lambda d: svdd_sample(d, value_scorer(vf), n, m, g())) == n * T     # learned value: 1 pass/step
    assert nfe(lambda d: svdd_sample(d, pm_scorer(sa_reward), n, m, g())) == n * T + n * m * (T)  # PM: +1 pass/child


def test_best_of_n_never_worse_than_its_first_draw_and_m1_equals_unguided() -> None:
    d = ToyDenoiser()
    one = sample_best_of_n(d, sa_reward, 50, 1, torch.Generator().manual_seed(3))
    assert one == sample_unguided(ToyDenoiser(), 50, torch.Generator().manual_seed(3))
    many = sample_best_of_n(ToyDenoiser(), sa_reward, 200, 10, torch.Generator().manual_seed(3))
    base = sample_unguided(ToyDenoiser(), 200, torch.Generator().manual_seed(3))
    assert np.mean([sa_reward(s) for s in many]) > np.mean([sa_reward(s) for s in base])


def test_seed_streams_independent_and_reproducible() -> None:
    a, b = seed_streams(0), seed_streams(0)
    draws = {k: torch.randint(10**9, (4,), generator=v).tolist() for k, v in a.items()}
    assert draws == {k: torch.randint(10**9, (4,), generator=v).tolist() for k, v in b.items()}
    assert len({tuple(v) for v in draws.values()}) == len(draws)       # all streams differ
    assert torch.randint(10**9, (4,), generator=seed_streams(1)["eval"]).tolist() != draws["eval"]
