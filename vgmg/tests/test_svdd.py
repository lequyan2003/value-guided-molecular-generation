"""SVDD selection and sampling behave as specified."""

from __future__ import annotations

import torch

from vgmg.models.denoiser import ToyDenoiser
from vgmg.rl.svdd import select_candidates, svdd_sample


def length_scorer(den, children, s):  # oracle: longer SMILES = higher value
    return torch.tensor([float(len(c)) for c in children])


def test_argmax_selects_highest_value() -> None:
    values = torch.tensor([[0.1, 0.9, 0.3], [0.5, 0.2, 0.7]])
    assert select_candidates(values, "argmax", 0.1, torch.Generator().manual_seed(0)).tolist() == [1, 2]


def test_low_alpha_softmax_is_near_greedy() -> None:
    values = torch.tensor([[0.0, 1.0, 0.0]] * 20)
    assert (select_candidates(values, "softmax", 1e-3, torch.Generator().manual_seed(0)) == 1).all()


def test_returns_n_states_and_is_deterministic() -> None:
    run = lambda: svdd_sample(ToyDenoiser(), length_scorer, 10, 5, torch.Generator().manual_seed(7))  # noqa: E731
    assert len(run()) == 10 and run() == run()


def test_guidance_beats_unguided_on_oracle_value() -> None:
    mean_len = lambda m: sum(  # noqa: E731
        len(s) for s in svdd_sample(ToyDenoiser(), length_scorer, 200, m, torch.Generator().manual_seed(0))) / 200
    assert mean_len(10) > mean_len(1)


def test_selected_child_comes_from_its_own_parent() -> None:
    """Row bookkeeping: child j of parent i is row i*m+j; selection must not mix parents."""
    class Tagged(ToyDenoiser):
        def init(self, n, g): return [f"p{i}" for i in range(n)]
        def propose(self, state, t, m, g): return [f"{state[k // m].split('.')[0]}.{k % m}" for k in range(len(state) * m)]
    out = svdd_sample(Tagged(num_steps=3), lambda d, c, s: torch.rand(len(c)), 6, 4, torch.Generator().manual_seed(0))
    assert [s.split(".")[0] for s in out] == [f"p{i}" for i in range(6)]


def test_value_fn_sees_child_time_t_minus_1() -> None:
    seen = []
    svdd_sample(ToyDenoiser(num_steps=3), lambda d, c, s: seen.append(s) or torch.zeros(len(c)), 4, 2,
                torch.Generator().manual_seed(0))
    assert seen == [2, 1, 0]
