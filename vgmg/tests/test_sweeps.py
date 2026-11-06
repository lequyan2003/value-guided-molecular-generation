"""Every committed sweep file expands to valid, uniquely identified runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_ablations import expand_sweep
from vgmg.constants import CONFIG_DIR
from vgmg.experiments import METHODS, make_run_id

SWEEPS = sorted((CONFIG_DIR / "sweeps").glob("*.yaml"))


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.stem)
def test_sweep_expands_to_unique_known_runs(path: Path) -> None:
    cfgs = expand_sweep(str(path))
    assert cfgs and all(c["method"] in METHODS for c in cfgs)
    ids = [make_run_id(c) for c in cfgs]
    assert len(ids) == len(set(ids)), "two grid points would overwrite each other's record"


def test_slot_a_sweeps_cover_the_main_comparison_at_m10_seeds_0_to_2() -> None:
    by = {p.stem: expand_sweep(str(p)) for p in SWEEPS if p.stem.startswith("slot_a")}
    runs = [(c["method"], c["svdd"]["num_candidates_m"], c["seed"]) for cfgs in by.values() for c in cfgs]
    for method in ("unguided", "best_of_n", "svdd_value", "svdd_pm", "random_select"):
        assert {s for m, _, s in runs if m == method} == {0, 1, 2}
    assert all(m == 10 for meth, m, _ in runs if meth != "unguided")
    assert by["slot_a_b2"][0]["best_of_n"]["m_values"] == [10]


def test_softmax_arm_is_separate_from_the_preregistered_argmax_primary() -> None:
    """The primary SVDD-MLP runs use argmax; the exploratory softmax arm shares their value function (same cache key)."""
    primary = expand_sweep("configs/sweeps/slot_a_svdd.yaml")
    arm = expand_sweep("configs/sweeps/slot_a_svdd_softmax.yaml")
    assert {c["svdd"]["selection"] for c in primary} == {"argmax"}
    assert {c["svdd"]["selection"] for c in arm} == {"softmax"} and {c["svdd"]["alpha"] for c in arm} == {0.1, 0.03, 0.01}
    assert primary[0]["value_fn"] == arm[0]["value_fn"] and primary[0]["svdd"]["value_fn_arch"] == arm[0]["svdd"]["value_fn_arch"]
