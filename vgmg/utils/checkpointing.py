"""Save / restore value-function checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    """Save weights plus metadata (seed, config, commit) to ``path``."""
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Load weights into ``model`` in place; return the stored metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt["meta"]
