"""Statistics, verdict rule, sweep expansion, and end-to-end determinism / resumability."""

from __future__ import annotations

import numpy as np

from vgmg.config import load_config
from vgmg.eval.stats import mean_std, paired_bootstrap_ci, preregistered_verdict
from vgmg.experiments import make_run_id, run_experiment


def test_mean_std_uses_sample_std() -> None:
    assert mean_std([1.0, 3.0]) == (2.0, np.sqrt(2.0))
    assert mean_std([5.0]) == (5.0, 0.0)


def test_bootstrap_ci_covers_known_difference() -> None:
    rng = np.random.default_rng(0)
    a = {s: rng.normal(0.6, 0.1, 500) for s in range(3)}
    b = {s: rng.normal(0.5, 0.1, 500) for s in range(3)}
    lo, hi = paired_bootstrap_ci(a, b, n_boot=300, rng=np.random.default_rng(1))
    assert lo < 0.1 < hi and lo > 0.0


def test_verdict_rule() -> None:
    win = preregistered_verdict([0.70, 0.71, 0.69], [0.60, 0.61, 0.59], [0.98] * 3, [0.99] * 3)
    assert win["wins"]
    noisy = preregistered_verdict([0.62, 0.70, 0.55], [0.60, 0.61, 0.59], [0.98] * 3, [0.99] * 3)
    assert not noisy["wins"]
    invalid = preregistered_verdict([0.70, 0.71, 0.69], [0.60, 0.61, 0.59], [0.80] * 3, [0.99] * 3)
    assert not invalid["wins"]  # validity dropped by 19 points


def _cfg(tmp_path, **ov):
    over = [f"output_dir={tmp_path}", "model.name=toy", "data.reference=smoke", "device=cpu"] + [f"{k}={v}" for k, v in ov.items()]
    return load_config("configs/smoke_test.yaml", over)


def test_run_is_deterministic_and_resumable(tmp_path) -> None:
    a = run_experiment(_cfg(tmp_path / "a", method="svdd_value"))
    b = run_experiment(_cfg(tmp_path / "b", method="svdd_value"))
    assert a["metrics"] == b["metrics"]                                    # same seed -> same numbers
    assert run_experiment(_cfg(tmp_path / "a", method="svdd_value")) == a  # finished run is reused
    c = run_experiment(_cfg(tmp_path / "c", method="svdd_value", seed=1))
    assert c["metrics"] != a["metrics"]                                    # seed matters


def test_run_ids_distinguish_configs() -> None:
    ids = {make_run_id(load_config("configs/default.yaml", o)) for o in (
        ["method=unguided"], ["method=best_of_n"], ["method=best_of_n", "svdd.num_candidates_m=5"],
        ["method=svdd_value", "seed=1"], ["method=svdd_value", "svdd.selection=softmax"], ["method=svdd_pm"])}
    assert len(ids) == 6


def test_sweep_expansion_covers_grid() -> None:
    import sys
    sys.path.insert(0, "scripts")
    from run_ablations import expand_sweep
    cfgs = expand_sweep("configs/sweeps/a1_m_sweep.yaml")
    assert len(cfgs) == 2 * 4 * 3 and len({make_run_id(c) for c in cfgs}) == len(cfgs)
    assert all(c["method"] == "svdd_value" for c in cfgs)


def test_every_sweep_file_expands_and_matches_the_proposal_grid() -> None:
    import glob, sys
    sys.path.insert(0, "scripts")
    from run_ablations import expand_sweep
    for path in glob.glob("configs/sweeps/*.yaml"):
        cfgs = expand_sweep(path)
        assert cfgs and len({make_run_id(c) for c in cfgs}) == len(cfgs), path
    seeds = lambda f: sorted({c["seed"] for c in expand_sweep(f"configs/sweeps/{f}.yaml")})  # noqa: E731
    assert seeds("a1_m_sweep") == [0, 1, 2] and seeds("b2_best_of_n") == [0, 1, 2]       # proposal: 3 seeds
    assert seeds("extra_seeds_svdd") == [3, 4] and seeds("b1_unguided") == [0, 1, 2, 3, 4]
    assert {c["svdd"]["num_candidates_m"] for c in expand_sweep("configs/sweeps/a1_m_sweep.yaml")} == {1, 5, 10, 20}


def test_interrupted_run_resumes_to_identical_molecules(tmp_path) -> None:
    from vgmg.experiments import build_denoiser, sample_molecules
    from vgmg.utils.reproducibility import seed_streams
    cfg = _cfg(tmp_path, method="svdd_pm")
    def go(n, d):
        c = {**cfg, "eval": {**cfg["eval"], "num_molecules": n, "batch_size": 50}}
        den = build_denoiser(c)
        return sample_molecules(c, den, None, seed_streams(0)["eval"], d)[0], den.nfe
    full, nfe_full = go(100, None)
    (tmp_path / "r").mkdir()
    go(50, tmp_path / "r")                 # "crash" after the first chunk
    resumed, nfe_resumed = go(100, tmp_path / "r")
    assert resumed == full and nfe_resumed == nfe_full


def test_best_of_n_pool_gives_all_m_and_matches_direct_selection(tmp_path) -> None:
    from vgmg.eval.baselines import select_best_of
    from vgmg.rl.rewards import sa_reward
    pool = ["CCCC", "CCO", "C1CC1", "CCN", "CC=O", "CC#N"]       # 2 slots x 3 draws
    assert select_best_of(pool, 2, 3, 1, sa_reward) == ["CCCC", "CCN"]  # m=1 -> first draw
    best3 = select_best_of(pool, 2, 3, 3, sa_reward)
    assert [sa_reward(b) for b in best3] == [max(sa_reward(x) for x in pool[:3]), max(sa_reward(x) for x in pool[3:])]
    cfg = _cfg(tmp_path, method="best_of_n", **{"best_of_n.m_values": "[2,4]"})
    rec = run_experiment(cfg)
    assert rec["method"] == "best_of_n" and rec["nfe_per_molecule"] == rec["M"] * 4 and rec["gpu"] == "cpu"
    logs = list((tmp_path / "logs").glob("best_of_n_*_M*_s42"))
    assert len(logs) == 2 * 2                                      # 2 rewards x 2 M values


def test_time_features_and_t0_check() -> None:
    import torch
    from vgmg.models.denoiser import ToyDenoiser
    from vgmg.models.value_function import MLPValueFunction, time_features
    from vgmg.rl.rewards import sa_reward
    from vgmg.rl.value_trainer import value_fn_t0_check
    assert time_features(torch.tensor([0.0, 0.5])).shape == (2, 9)
    out = value_fn_t0_check(ToyDenoiser(), MLPValueFunction(128, 8, 1), ["CCO", "CCN", "C1CC1", "CC=O"], sa_reward)
    assert out["n"] == 4 and "spearman" in out


def test_aggregation_and_verdict_end_to_end_on_toy(tmp_path) -> None:
    from vgmg.eval.comparison import headline_verdict, load_runs, runs_to_rows, summarize
    for seed in (0, 1, 2):
        run_experiment(_cfg(tmp_path, method="svdd_value", seed=seed, **{"svdd.num_candidates_m": 10}))
        run_experiment(_cfg(tmp_path, method="best_of_n", seed=seed, **{"best_of_n.m_values": "[10]"}))
        run_experiment(_cfg(tmp_path, method="unguided", seed=seed))
    rows = runs_to_rows(load_runs(tmp_path / "logs"))
    table = summarize(rows)
    assert (table.n_seeds == 3).all() and set(table.gpus) == {"cpu"}
    v = headline_verdict(rows, tmp_path / "logs", "sa", seeds=[0, 1, 2])
    assert v["seeds"] == [0, 1, 2] and "wins" in v and len(v["bootstrap_ci95"]) == 2
    assert headline_verdict(rows, tmp_path / "logs", "sa", seeds=[0, 1])["status"].startswith("insufficient")
