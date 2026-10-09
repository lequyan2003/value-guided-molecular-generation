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
    table = summarize(rows).sort_values(["reward", "method", "M"])
    out = logs.parent / "tables"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "summary.csv", index=False)
    cols = ["method", "reward", "M", "n_seeds", "reward_mean", "reward_mean_std", "validity", "validity_std",
            "diversity", "nfe_per_molecule"]
    (out / "summary.tex").write_text(to_latex(table, cols))
    verdicts = [headline_verdict(rows, logs, r) for r in sorted(rows.reward.unique())]
    (out / "verdict.json").write_text(json.dumps(verdicts, indent=2))
    print(table[cols].to_string(index=False))
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
