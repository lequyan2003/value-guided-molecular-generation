"""Value-function training, chain-level splits and diagnostics."""

from __future__ import annotations

import torch

from vgmg.models.denoiser import ToyDenoiser
from vgmg.models.value_function import MLPValueFunction
from vgmg.rl.rewards import sa_reward
from vgmg.rl.value_trainer import collect_rollouts, evaluate_value_function, sibling_ranking, spearman, train_value_function


def test_training_reduces_val_loss_and_uses_time_input() -> None:
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    from vgmg.rl.value_trainer import RolloutData
    def make(n, off):
        x = torch.randint(0, 2, (n, 16), generator=g).float()
        t = torch.rand(n, generator=g)
        return RolloutData(x, t, x[:, :4].mean(1) * t, torch.arange(n) + off)  # target depends on t too
    tr, va = make(512, 0), make(128, 512)
    model = MLPValueFunction(16, 32, 1)
    before = evaluate_value_function(model, va)["mse"]
    hist = train_value_function(model, tr, va, steps=400, eval_every=50, generator=g)
    assert hist["best_val_mse"] < 0.3 * before


def test_rollout_chains_are_disjoint_between_train_and_val_and_labels_constant_per_chain() -> None:
    den = ToyDenoiser()
    tr = collect_rollouts(den, sa_reward, 30, torch.Generator().manual_seed(1), ts_per_chain=3)
    va = collect_rollouts(den, sa_reward, 10, torch.Generator().manual_seed(2), ts_per_chain=3, chain_offset=30)
    assert set(tr.chain.tolist()).isdisjoint(va.chain.tolist())
    for c in tr.chain.unique():  # same chain -> same target
        assert tr.targets[tr.chain == c].unique().numel() == 1
    assert len(tr) == 30 * 3 and tr.t_frac.min() > 0


def test_spearman() -> None:
    import numpy as np
    assert spearman(np.array([1, 2, 3.0]), np.array([10, 20, 30.0])) == 1.0
    assert spearman(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) == -1.0


def test_sibling_ranking_runs() -> None:
    out = sibling_ranking(ToyDenoiser(), MLPValueFunction(128, 16, 1), sa_reward, 2, 20, 4, 3,
                          torch.Generator().manual_seed(0))
    assert out["n_parents_used"] >= 0 and out["m"] == 4
