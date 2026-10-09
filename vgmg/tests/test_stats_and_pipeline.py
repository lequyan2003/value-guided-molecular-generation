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
    assert len(cfgs) == 2 * 4 * 5 and len({make_run_id(c) for c in cfgs}) == len(cfgs)
    assert all(c["method"] == "svdd_value" for c in cfgs)
