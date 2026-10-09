"""Per-run logging: creates ``results/logs/<name>_<timestamp>/`` and records provenance."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from vgmg.utils.reproducibility import log_run_metadata


class RunLogger:
    """Logger bundled with its run directory."""

    def __init__(self, logger: logging.Logger, run_dir: Path) -> None:
        self.logger = logger
        self.run_dir = run_dir

    def info(self, msg: str) -> None:
        """Log an info message to console and ``run.log``."""
        self.logger.info(msg)

    def save_json(self, name: str, obj: Any) -> None:
        """Write ``obj`` as JSON into the run directory."""
        (self.run_dir / name).write_text(json.dumps(obj, indent=2, default=float))


@contextmanager
def setup_logging(name: str, config: dict[str, Any], run_id: str | None = None) -> Iterator[RunLogger]:
    """Create a run dir, save config/seed/commit/metadata, and yield a ``RunLogger``.

    With ``run_id`` the directory is deterministic (``results/logs/<run_id>``) so sweeps can
    detect finished runs and resume after preemption; otherwise it is ``<name>_<timestamp>``.
    Wall-clock time is appended to ``metadata.json`` on exit.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get("output_dir", "./results")) / "logs" / (run_id or f"{name}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"vgmg.{name}.{stamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    seed = int(config["seed"])
    meta = log_run_metadata(seed)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "seed.txt").write_text(f"{seed}\n")
    (run_dir / "git_commit.txt").write_text(f"{meta['git_commit']}\n")

    run = RunLogger(logger, run_dir)
    run.info(f"Run dir: {run_dir}")
    run.info(f"Seed: {seed}, Commit: {meta['git_commit']}")
    start = time.time()
    try:
        yield run
    finally:
        meta["wall_clock_s"] = round(time.time() - start, 2)
        run.save_json("metadata.json", meta)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
