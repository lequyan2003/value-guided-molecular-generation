"""Smoke test: every method end to end on the toy stand-in generator (CPU, a few seconds).

Runs unguided, best-of-N, SVDD-PM, SVDD-value and random-selection through the same
``run_experiment`` code path as the real experiments, then prints a comparison. The toy is
NOT a molecular generator: these numbers only show that the pipeline runs and is reproducible.
Replace ``model.name=toy`` with ``digress`` for real runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import parse_config  # noqa: E402
from vgmg.experiments import run_experiment  # noqa: E402


def main() -> None:
    """Run all methods once and print reward / validity / network passes per molecule."""
    cfg = parse_config("SVDD smoke test", default_config="configs/smoke_test.yaml")
    print(f"{'method':<14}{'reward':>8}{'valid':>8}{'nfe/mol':>9}")
    for method in ("unguided", "best_of_n", "svdd_pm", "svdd_value", "random_select"):
        s = run_experiment({**cfg, "method": method})
        head = s["reward"] or cfg["svdd"]["reward_type"]
        m = s["metrics"][head]
        print(f"{method:<14}{m['reward_mean']:>8.3f}{m['validity']:>8.2f}{s['nfe_per_molecule']:>9.1f}")


if __name__ == "__main__":
    main()
