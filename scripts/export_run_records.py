"""Copy the small, final files of every finished run into ``release_results/logs`` (tracked in git).

``results/`` is git-ignored (it also holds checkpoints and resume files); ``release_results/`` is the evidence the
report cites: for each run ``summary.json`` (metrics, value-function diagnostics, commit, GPU, checkpoint SHA-256),
``config.yaml``, ``seed.txt``, ``git_commit.txt``, ``metadata.json`` and ``molecules.csv`` (needed by the paired
bootstrap of the verdict). ``make reproduce`` regenerates the results table, the verdict and the figures from it with no
GPU. Usage: ``python scripts/export_run_records.py [src=results/logs] [dst=release_results/logs]``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

KEEP = ("summary.json", "config.yaml", "seed.txt", "git_commit.txt", "metadata.json", "molecules.csv")


def main() -> None:
    """Copy finished runs (those with a summary.json that is a run record) and report what was copied."""
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/logs")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("release_results/logs")
    copied = skipped = 0
    for run in sorted(p for p in src.iterdir() if p.is_dir()):
        summary = run / "summary.json"
        if not summary.exists() or '"metrics"' not in summary.read_text():
            skipped += 1                      # smoke runs, partial pools: not evidence
            continue
        (dst / run.name).mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            if (run / name).exists():
                shutil.copy2(run / name, dst / run.name / name)
        copied += 1
    print(f"copied {copied} runs to {dst} (skipped {skipped} folders without a final run record)")


if __name__ == "__main__":
    main()
