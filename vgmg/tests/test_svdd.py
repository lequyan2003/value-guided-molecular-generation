"""SVDD selection and sampling behave as specified."""

import torch

from vgmg.models.digress import ToyDenoiser
from vgmg.rl.svdd import select_candidates, svdd_sample


class _OracleValue(torch.nn.Module):
    """Value function that returns feature[0] (we encode the reward there)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]


def test_argmax_selects_highest_value() -> None:
    values = torch.tensor([[0.1, 0.9, 0.3], [0.5, 0.2, 0.7]])
    gen = torch.Generator().manual_seed(0)
    assert select_candidates(values, "argmax", 0.1, gen).tolist() == [1, 2]


def test_low_alpha_softmax_is_near_greedy() -> None:
    values = torch.tensor([[0.0, 1.0, 0.0]] * 20)
    gen = torch.Generator().manual_seed(0)
    assert (select_candidates(values, "softmax", 1e-3, gen) == 1).all()


def test_svdd_returns_n_states_and_is_deterministic() -> None:
    den = ToyDenoiser()
    feat = lambda s: torch.tensor([float(len(s))])  # noqa: E731  longer SMILES = higher value
    run = lambda: svdd_sample(den, _OracleValue(), feat, 10, 5, torch.Generator().manual_seed(7))  # noqa: E731
    out = run()
    assert len(out) == 10 and out == run()


def test_guidance_beats_unguided_on_oracle_value() -> None:
    den = ToyDenoiser()
    feat = lambda s: torch.tensor([float(len(s))])  # noqa: E731
    mean_len = lambda m: sum(  # noqa: E731
        len(s) for s in svdd_sample(den, _OracleValue(), feat, 200, m, torch.Generator().manual_seed(0))
    ) / 200
    assert mean_len(10) > mean_len(1)
