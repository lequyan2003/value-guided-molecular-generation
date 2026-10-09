"""Run ONE experiment (method x reward x M x seed): ``python scripts/run_inference.py method=svdd_pm seed=1``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import parse_config  # noqa: E402
from vgmg.experiments import run_experiment  # noqa: E402


def main() -> None:
    """Parse config + overrides, run the experiment, print where the record went."""
    cfg = parse_config("Run one SVDD / baseline experiment")
    summary = run_experiment(cfg)
    print(f"{summary['run_id']}: done")


if __name__ == "__main__":
    main()
