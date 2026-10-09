"""Centralized config: YAML file + dotted command-line overrides.

Example:
    python scripts/smoke_test.py --config configs/smoke_test.yaml svdd.num_candidates=10 seed=1
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Sequence

import yaml

from vgmg.constants import CONFIG_DIR

Config = dict[str, Any]


def load_yaml(path: str | Path) -> Config:
    """Load a YAML file into a nested dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Config, new: Config) -> Config:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML) to a copy of ``cfg``."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load ``configs/default.yaml``, merge ``path`` on top, then apply overrides."""
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    if Path(path).resolve() != (CONFIG_DIR / "default.yaml").resolve():
        cfg = _deep_merge(cfg, load_yaml(path))
    return apply_overrides(cfg, overrides)


def parse_config(description: str, argv: Sequence[str] | None = None,
                 default_config: str = "configs/default.yaml") -> Config:
    """Standard CLI for all scripts: ``--config PATH [key=value ...]``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="YAML config path")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. seed=1")
    args = parser.parse_args(argv)
    return load_config(args.config, args.overrides)
