"""Aggregate results/logs/*/summary.json -> results/tables/{summary.csv,summary.tex,verdict.json}.

Every number in the report must come from these files (no hand-typed values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.constants import RESULTS_DIR  # noqa: E402
from vgmg.eval.comparison import headline_verdict, load_runs, runs_to_rows, summarize, to_latex  # noqa: E402


def main() -> None:
    """Write the summary table and the mechanical verdict for the headline comparison (M=10)."""
    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "logs"
    runs = load_runs(logs)
    if not runs:
        raise SystemExit(f"No summary.json files found under {logs}")
    rows = runs_to_rows(runs)
    commits = sorted({r.get("git_commit", "unknown") for r in runs})
    if len(commits) > 1 or any(c.endswith("-dirty") or c == "unknown" for c in commits):
        print(f"WARNING: runs come from {len(commits)} commits (dirty/unknown present: "
              f"{any(c.endswith('-dirty') or c == 'unknown' for c in commits)}). Headline runs must come from one clean commit:", commits)
    table = summarize(rows).sort_values(["reward", "method", "M"])
    out = logs.parent / "tables"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "summary.csv", index=False)
    cols = ["method", "reward", "M", "n_seeds", "reward_mean", "reward_mean_std", "validity", "validity_std",
            "uniqueness", "diversity", "nfe_per_molecule", "gpus"]
    (out / "summary.tex").write_text(to_latex(table, cols))
    verdicts = [headline_verdict(rows, logs, r, seeds=sd) | {"seed_set": name}
                for r in sorted(rows.reward.unique()) for name, sd in (("all", None), ("0-2 (proposal)", [0, 1, 2]))]
    # wall-clock is only comparable within one GPU type: one row per (method, reward, M, GPU)
    wall = (rows.groupby(["method", "reward", "M", "gpu"])["wall_s_per_1k"].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "wall_s_per_1k_mean", "count": "n_runs"}))
    wall.to_csv(out / "wall_clock_by_gpu.csv", index=False)
    mixed = table[table["gpus"].str.contains(r"\+")]
    if len(mixed):
        print("WARNING: rows mix GPU types, wall-clock is not comparable there:", mixed[["method", "reward", "M", "gpus"]].to_dict("records"))
    (out / "verdict.json").write_text(json.dumps(verdicts, indent=2))
    # Exploratory arms (not pre-registered as the primary test) scored with the SAME rule against best-of-N, kept in a
    # separate file so the pre-registered verdict above is never mixed with them.
    explore = [m for m in sorted(rows.method.unique()) if m.startswith("svdd_value_") and m != "svdd_value"]
    expl = [headline_verdict(rows, logs, r, method=m) | {"seed_set": "all paired seeds", "exploratory": True}
            for m in explore for r in sorted(rows[rows.method == m].reward.unique())]
    (out / "verdict_exploratory.json").write_text(json.dumps(expl, indent=2))
    print(table[cols].to_string(index=False))
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
