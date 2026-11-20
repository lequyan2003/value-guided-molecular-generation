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


def test_every_chain_draws_its_own_distinct_timesteps() -> None:
    den = ToyDenoiser(num_steps=50)
    d = collect_rollouts(den, sa_reward, 200, torch.Generator().manual_seed(3), ts_per_chain=5)
    assert len(d) == 200 * 5 and (d.chain.bincount() == 5).all()        # exactly 5 rows per chain
    per_chain = [tuple(sorted(d.t_frac[d.chain == c].mul(50).round().int().tolist())) for c in range(200)]
    assert all(len(set(ts)) == 5 for ts in per_chain)                   # distinct within a chain
    assert len(set(per_chain)) > 100                                    # but not shared between chains
    assert set(torch.cat([d.t_frac[d.chain == c] for c in range(200)]).mul(50).round().int().tolist()) == set(range(1, 51))
    assert d.features.shape == (len(d), den.feature_dim)


def test_spearman() -> None:
    import numpy as np
    assert spearman(np.array([1, 2, 3.0]), np.array([10, 20, 30.0])) == 1.0
    assert spearman(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) == -1.0


def test_sibling_ranking_runs() -> None:
    out = sibling_ranking(ToyDenoiser(), MLPValueFunction(128, 16, 1), sa_reward, 2, 20, 4, 3,
                          torch.Generator().manual_seed(0))
    assert out["n_parents_used"] >= 0 and out["m"] == 4


def test_guided_rollouts_with_one_candidate_equal_unguided_rollouts() -> None:
    """With m = 1 the guided collector must consume the RNG exactly like the unguided one (no hidden selection)."""
    from vgmg.rl.value_trainer import collect_guided_rollouts
    den = ToyDenoiser(num_steps=6)
    a = collect_rollouts(den, sa_reward, 40, torch.Generator().manual_seed(5), ts_per_chain=3)
    b = collect_guided_rollouts(den, sa_reward, lambda d, c, s: torch.zeros(len(c)), 1, 40,
                                torch.Generator().manual_seed(5), ts_per_chain=3)
    assert torch.equal(a.features, b.features) and torch.equal(a.targets, b.targets) and torch.equal(a.chain, b.chain)


def test_guided_rollouts_follow_the_scorer_and_aggregation_keeps_chains_disjoint() -> None:
    from vgmg.rl.value_trainer import collect_guided_rollouts, concat_rollouts
    import pytest
    den = ToyDenoiser(num_steps=8)
    prefer_long = lambda d, c, s: torch.tensor([float(len(x)) for x in d.decode(c)])  # noqa: E731
    prefer_short = lambda d, c, s: -prefer_long(d, c, s)  # noqa: E731
    long_ = collect_guided_rollouts(den, sa_reward, prefer_long, 5, 200, torch.Generator().manual_seed(1), ts_per_chain=2)
    short = collect_guided_rollouts(den, sa_reward, prefer_short, 5, 200, torch.Generator().manual_seed(1), ts_per_chain=2,
                                    chain_offset=1000)
    assert long_.targets.numel() == 400 and torch.all(long_.t_frac > 0)
    assert not torch.equal(long_.targets, short.targets)      # the selection rule changes where chains end
    both = concat_rollouts(long_, short)
    assert len(both) == len(long_) + len(short)
    with pytest.raises(AssertionError):
        concat_rollouts(long_, long_)                         # overlapping chain ids are refused


def test_refit_rounds_change_run_id_and_cache_tag_but_not_default_ones(tmp_path) -> None:
    """H4 is opt-in: with refit_rounds absent every run id and cache key is the old one; with it, both carry h4r<k>."""
    from vgmg.config import load_config
    from vgmg.experiments import get_value_function, make_run_id
    base = [f"output_dir={tmp_path}", "svdd.num_candidates_m=3", "value_fn.train_chains=30", "value_fn.val_chains=10",
            "value_fn.ts_per_chain=3", "value_fn.train_steps=20", "value_fn.eval_every=10", "value_fn.diag_t0_molecules=0",
            "value_fn.diag_sibling_parents=0", "seed=1"]
    plain = load_config("configs/smoke_test.yaml", base)
    h4 = load_config("configs/smoke_test.yaml", base + ["value_fn.refit_rounds=2"])
    assert make_run_id(plain) == "svdd_value_mlp_sa_M3_s1" and make_run_id(h4) == "svdd_value_mlp_sa_M3_h4r2_s1"
    from vgmg.experiments import build_denoiser
    model, diag = get_value_function(h4, build_denoiser(h4))
    assert [r["round"] for r in diag["refit"]] == [1, 2]
    assert diag["n_train_rows"] == 30 * 3 * (1 + 2) and "refit" not in get_value_function(plain, build_denoiser(plain))[1]
    assert any("h4r2" in p.name for p in (tmp_path / "value_fns").glob("*.pt"))
