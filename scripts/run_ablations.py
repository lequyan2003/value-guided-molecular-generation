"""Run every point of a sweep file, in process, skipping finished runs (safe to resubmit after preemption).

Sweep file format:  base: {dotted.key: value}   # fixed overrides
                    sweep: {dotted.key: [v1, v2]}  # cartesian grid
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import apply_overrides, load_config, load_yaml  # noqa: E402


def expand_sweep(sweep_path: str, extra_overrides: list[str] = ()) -> list[dict]:
    """One full config per grid point of ``sweep_path`` (default config + base + grid + extra overrides)."""
    spec = load_yaml(sweep_path)
    base = [f"{k}={v}" for k, v in spec.get("base", {}).items()]
    grid = spec["sweep"]
    keys = list(grid)
    return [
        apply_overrides(load_config("configs/default.yaml"),
                        base + [f"{k}={v}" for k, v in zip(keys, vals)] + list(extra_overrides))
        for vals in itertools.product(*(grid[k] for k in keys))
    ]


def main() -> None:
    """``python scripts/run_ablations.py configs/sweeps/a1_m_sweep.yaml [key=value ...]``"""
    from vgmg.experiments import make_run_id, run_experiment
    sweep_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sweeps/a1_m_sweep.yaml"
    configs = expand_sweep(sweep_path, sys.argv[2:])
    print(f"{len(configs)} grid points in {sweep_path} (finished runs are skipped)")
    for cfg in configs:
        existed = (Path(cfg["output_dir"]) / "logs" / make_run_id(cfg) / "summary.json").exists()
        run_experiment(cfg)
        print(f"  {make_run_id(cfg)}: {'reused' if existed else 'ran'}")


if __name__ == "__main__":
    main()
