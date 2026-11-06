"""The demo runs in toy mode, writes its cache and images, and replays identically from the cache."""

from __future__ import annotations

import json
import subprocess
import sys

from vgmg.constants import REPO_ROOT


def _run(*args: str) -> str:
    return subprocess.run([sys.executable, "scripts/demo.py", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout


def test_toy_demo_then_replay_gives_the_same_report(tmp_path) -> None:
    cache, out = tmp_path / "c.json", tmp_path / "img"
    live = _run("--toy", "--config", "configs/smoke_test.yaml", "--cache", str(cache), "--out", str(out),
                f"output_dir={tmp_path}/res")
    assert "TOY: pipeline only" in live and cache.exists()
    assert len(list(out.glob("*.png"))) == 1 + 3 + 3                       # movie, 3 SVDD steps, 3 method grids
    replay = _run("--replay", "--cache", str(cache), "--out", str(tmp_path / "img2"))
    body = lambda s: [ln for ln in s.splitlines() if not ln.startswith("images in")]  # noqa: E731
    assert body(live) == body(replay)
    assert json.loads(cache.read_text())["model"] == "toy"
