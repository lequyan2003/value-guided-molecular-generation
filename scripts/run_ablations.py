"""Run every config in a sweep file (cartesian product over `sweep:` keys). TODO(week 6)."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import apply_overrides, load_config, load_yaml  # noqa: E402


def expand_sweep(sweep_path: str) -> list[dict]:
    """Expand ``sweep:`` in a YAML file into one full config per grid point."""
    sweep = load_yaml(sweep_path)["sweep"]
    keys = list(sweep)
    return [
        apply_overrides(load_config("configs/default.yaml"), [f"{k}={v}" for k, v in zip(keys, vals)])
        for vals in itertools.product(*(sweep[k] for k in keys))
    ]


def main() -> None:
    """Expand the sweep and call run_inference for each point, one logged run per config."""
    sweep_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ablation_m.yaml"
    configs = expand_sweep(sweep_path)
    print(f"{len(configs)} runs in {sweep_path}")
    raise NotImplementedError("Dispatch each config to run_inference once DiGress is wired up")


if __name__ == "__main__":
    main()
