"""Aggregate results/logs/*/summary.json into results/tables/summary.tex."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.constants import RESULTS_DIR  # noqa: E402
from vgmg.eval.comparison import collect_summaries, to_latex  # noqa: E402


def main() -> None:
    """Write a LaTeX table of key metrics, one row per logged run."""
    df = collect_summaries(RESULTS_DIR / "logs")
    if df.empty:
        raise SystemExit("No summary.json files found under results/logs/")
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)
    out = RESULTS_DIR / "tables" / "summary.tex"
    out.write_text(to_latex(df, [c for c in ("run", "seed", "M", "reward") if c in df.columns]))
    print(out)


if __name__ == "__main__":
    main()
