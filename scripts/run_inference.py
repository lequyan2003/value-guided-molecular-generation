"""Sample molecules with DiGress + SVDD and write metrics. TODO(week 4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import parse_config  # noqa: E402
from vgmg.utils.logging import setup_logging  # noqa: E402
from vgmg.utils.reproducibility import set_seed  # noqa: E402


def main() -> None:
    """Load DiGress + value fn checkpoint, run svdd_sample, write metrics.csv/summary.json."""
    cfg = parse_config("SVDD inference")
    set_seed(cfg["seed"])
    with setup_logging("svdd_inference", cfg):
        raise NotImplementedError("Needs DiGressDenoiser (week 1) and a trained value function")


if __name__ == "__main__":
    main()
