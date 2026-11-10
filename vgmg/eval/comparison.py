"""Aggregate per-run ``summary.json`` files into tables and the pre-registered verdict."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from vgmg.eval.stats import mean_std, paired_bootstrap_ci, preregistered_verdict

METRIC_COLS = ["reward_mean", "reward_top10", "validity", "validity_digress", "uniqueness", "novelty", "diversity"]


def load_runs(logs_dir: str | Path) -> list[dict]:
    """Every finished run's summary (a run is finished iff ``summary.json`` exists).

    A ``summary.json`` without ``run_id`` and ``metrics`` is not a run record (e.g. the smoke-test script writes its own
    format into the same folder): it is skipped with a warning instead of crashing the whole table.
    """
    runs = []
    for p in sorted(Path(logs_dir).glob("*/summary.json")):
        rec = json.loads(p.read_text())
        if "run_id" in rec and "metrics" in rec:
            runs.append(rec)
        else:
            warnings.warn(f"skipping {p.parent.name}: summary.json is not a run record", stacklevel=2)
    return runs


def method_label(run: dict) -> str:
    """Table label of a run. Softmax runs carry their temperature (read from the run id, which encodes it), because
    averaging over different alphas would hide the validity collapse of the small ones."""
    if run["selection"] != "softmax":
        return run["method"]
    alpha = re.search(r"softmax([0-9.]+?)_s\d+$", run["run_id"])
    return f"{run['method']}_softmax" + (f"_a{alpha.group(1)}" if alpha else "")


def runs_to_rows(runs: list[dict]) -> pd.DataFrame:
    """One row per (run, scored reward). Unguided runs are scored under every reported reward."""
    rows = []
    for r in runs:
        for reward, m in r["metrics"].items():
            if r["reward"] not in (None, reward):
                continue  # a guided run is only evaluated under the reward it optimised
            rows.append({"method": method_label(r),
                         "reward": reward, "M": r["M"], "seed": r["seed"], "run_id": r["run_id"], "gpu": r.get("gpu", "unknown"),
                         "nfe_per_molecule": r["nfe_per_molecule"],
                         "wall_s_per_1k": 1000 * r["sample_wall_clock_s"] / r["num_molecules"],
                         **{c: m.get(c, float("nan")) for c in METRIC_COLS}})  # records written before a metric existed give nan
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std across seeds for each (method, reward, M); ``n_seeds`` shows the evidence."""
    out = []
    for (method, reward, M), g in rows.groupby(["method", "reward", "M"]):
        rec = {"method": method, "reward": reward, "M": M, "n_seeds": g["seed"].nunique(),
               "gpus": "+".join(sorted(g["gpu"].unique()))}
        for c in METRIC_COLS + ["nfe_per_molecule", "wall_s_per_1k"]:
            mu, sd = mean_std(g[c].tolist())
            rec[c], rec[c + "_std"] = mu, sd
        out.append(rec)
    return pd.DataFrame(out)


def to_latex(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as a booktabs LaTeX table (3 decimals)."""
    return df[columns].to_latex(index=False, float_format="%.3f")


def _per_molecule(logs_dir: Path, run_ids: dict[int, str], reward: str) -> dict[int, np.ndarray]:
    return {s: pd.read_csv(logs_dir / rid / "molecules.csv")[f"reward_{reward}"].to_numpy()
            for s, rid in run_ids.items()}


def headline_verdict(rows: pd.DataFrame, logs_dir: str | Path, reward: str, M: int = 10,
                     method: str = "svdd_value", baseline: str = "best_of_n",
                     seeds: list[int] | None = None) -> dict:
    """Apply the pre-registered rule: ``method`` vs ``baseline`` at ``M`` for ``reward``, paired by seed.

    ``seeds=None`` uses every seed present for both methods (0-4 once the extra seeds are run).
    """
    logs_dir = Path(logs_dir)
    sel = rows[(rows.reward == reward) & (rows.M == M)]
    a, b = sel[sel.method == method].set_index("seed"), sel[sel.method == baseline].set_index("seed")
    seeds = sorted(set(a.index) & set(b.index) & (set(seeds) if seeds else set(a.index)))
    if len(seeds) < 3:
        return {"reward": reward, "M": M, "status": f"insufficient paired seeds ({len(seeds)} < 3)"}
    # Rule 2 of docs/PREREGISTRATION.md compares validity with B1 (unguided), not with the reward baseline.
    u = rows[(rows.reward == reward) & (rows.method == "unguided")].set_index("seed")
    if not set(seeds) <= set(u.index):
        return {"reward": reward, "M": M, "status": "missing unguided (B1) seeds for the validity clause"}
    v = preregistered_verdict(a.loc[seeds, "reward_mean"].tolist(), b.loc[seeds, "reward_mean"].tolist(),
                              a.loc[seeds, "validity"].tolist(), u.loc[seeds, "validity"].tolist())
    lo, hi = paired_bootstrap_ci(_per_molecule(logs_dir, a.loc[seeds, "run_id"].to_dict(), reward),
                                 _per_molecule(logs_dir, b.loc[seeds, "run_id"].to_dict(), reward))
    return {"reward": reward, "M": M, "method": method, "baseline": baseline, "seeds": [int(x) for x in seeds],
            "bootstrap_ci95": [lo, hi], **v}
