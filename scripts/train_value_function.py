"""Train (and cache) the value function for one (model, reward, arch, seed); prints its diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import parse_config  # noqa: E402
from vgmg.experiments import build_denoiser, get_value_function  # noqa: E402
from vgmg.utils.logging import setup_logging  # noqa: E402
from vgmg.utils.reproducibility import set_seed  # noqa: E402


def main() -> None:
    """Collect unguided rollouts, train with validation-based model selection, save checkpoint + diagnostics."""
    cfg = parse_config("Train SVDD value function")
    set_seed(cfg["seed"])
    with setup_logging("train_value_fn", cfg) as log:
        _, diag = get_value_function(cfg, build_denoiser(cfg), log)
        log.save_json("value_fn_diagnostics.json", diag)
        print(json.dumps(diag["val"], indent=2))


if __name__ == "__main__":
    main()
