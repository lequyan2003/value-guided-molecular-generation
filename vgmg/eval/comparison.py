"""Aggregate per-run ``summary.json`` files into an ablation table (CSV / LaTeX)."""

import json
from pathlib import Path

import pandas as pd


def collect_summaries(logs_dir: str | Path) -> pd.DataFrame:
    """Read every ``<run>/summary.json`` under ``logs_dir`` into one row per run."""
    rows = []
    for path in sorted(Path(logs_dir).glob("*/summary.json")):
        row = json.loads(path.read_text())
        row["run"] = path.parent.name
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as a booktabs LaTeX table (3 decimals)."""
    return df[columns].to_latex(index=False, float_format="%.3f")
