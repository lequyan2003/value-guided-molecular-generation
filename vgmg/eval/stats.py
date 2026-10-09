"""Statistics for the pre-registered comparison (docs/PREREGISTRATION.md). Pure numpy, unit-tested."""

from __future__ import annotations

import numpy as np


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std (ddof=1; 0.0 for a single value) across seeds."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def paired_bootstrap_ci(a: dict[int, np.ndarray], b: dict[int, np.ndarray], n_boot: int = 2000,
                        level: float = 0.95, rng: np.random.Generator | None = None) -> tuple[float, float]:
    """CI for mean(a) - mean(b) from resampling molecules within each seed (seeds held fixed).

    ``a`` and ``b`` map seed -> per-molecule rewards. This captures molecule-sampling noise only;
    seed-to-seed variation (value-function retraining etc.) is captured by ``mean_std`` over
    per-seed means. Both must be reported; neither alone justifies a claim.
    """
    rng = rng or np.random.default_rng(0)
    seeds = sorted(set(a) & set(b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        per_seed = [rng.choice(a[s], len(a[s])).mean() - rng.choice(b[s], len(b[s])).mean() for s in seeds]
        diffs[i] = np.mean(per_seed)
    lo, hi = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def preregistered_verdict(svdd_means: list[float], base_means: list[float],
                          svdd_validity: list[float], base_validity: list[float]) -> dict:
    """The decision rule fixed in the proposal, evaluated mechanically.

    SVDD "wins" iff  mean gain > 2 * noise  AND  validity drop <= 0.05 (absolute),
    with noise = max(std over seeds of SVDD means, std over seeds of baseline means).
    """
    gain = float(np.mean(svdd_means) - np.mean(base_means))
    noise = max(mean_std(svdd_means)[1], mean_std(base_means)[1])
    val_drop = float(np.mean(base_validity) - np.mean(svdd_validity))
    return {"gain": gain, "noise": noise, "gain_over_noise": gain / noise if noise > 0 else float("inf"),
            "validity_drop": val_drop, "wins": bool(gain > 2 * noise and val_drop <= 0.05)}
