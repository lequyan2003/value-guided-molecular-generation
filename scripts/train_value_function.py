"""Train the value function on DiGress rollouts. TODO(week 3): wire up DiGressDenoiser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import parse_config  # noqa: E402
from vgmg.utils.logging import setup_logging  # noqa: E402
from vgmg.utils.reproducibility import set_seed  # noqa: E402


def main() -> None:
    """Load DiGress, collect rollouts, train value fn, save checkpoint (log seed/config)."""
    cfg = parse_config("Train SVDD value function")
    set_seed(cfg["seed"])
    with setup_logging("train_value_fn", cfg):
        raise NotImplementedError("Needs DiGressDenoiser (week 1) and collect_rollouts on it (week 3)")


if __name__ == "__main__":
    main()
