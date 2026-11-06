"""Seeding and run-metadata capture."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from vgmg.constants import REPO_ROOT


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU+CUDA) and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    """Return the current git commit hash (with ``-dirty`` suffix), or ``unknown``."""
    try:
        def run(*args: str) -> str:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()

        commit = run("rev-parse", "HEAD")
        return commit + ("-dirty" if run("status", "--porcelain") else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_run_metadata(seed: int) -> dict[str, Any]:
    """Collect seed, git commit, and software/hardware versions for provenance."""
    cuda = torch.cuda.is_available()
    return {
        "seed": seed,
        "git_commit": get_git_commit(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "platform": platform.platform(),
    }


STREAM_NAMES = ("rollout_train", "rollout_val", "value_init", "eval", "diagnostics")  # append only: keeps older streams


def seed_streams(seed: int) -> dict[str, torch.Generator]:
    """Independent named generators derived from one run seed.

    Value-function rollouts (train / val), network init and final evaluation sampling each
    get their own stream, so evaluation molecules can never coincide with value-function
    training data by construction, and a change in one stage does not shift another.
    """
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return {
        name: torch.Generator().manual_seed(int(c.generate_state(1)[0]))
        for name, c in zip(STREAM_NAMES, children)
    }
