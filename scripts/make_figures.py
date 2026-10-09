"""Generate every report/slide figure from results/tables/summary.csv and run summaries (no hand-made plots).

usage: python scripts/make_figures.py [logs_dir=results/logs] [out_dir=results/figures]
Colors: fixed categorical slots per method (never cycled) plus a distinct marker per method, so identity
never rests on color alone. Error bars are the std across seeds (the n is in summary.csv, n_seeds).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vgmg.eval.comparison import load_runs, runs_to_rows, summarize  # noqa: E402

STYLE = {  # method -> (color, marker, label); slots from the validated reference palette, light mode
    "unguided": ("#52514e", "o", "B1 unguided"),
    "best_of_n": ("#2a78d6", "s", "B2 best-of-N"),
    "svdd_value": ("#eb6834", "^", "SVDD-MLP"),
    "svdd_pm": ("#1baf7a", "D", "SVDD-PM"),
    "random_select": ("#e87ba4", "v", "random select"),
}
INK, MUTED = "#0b0b0b", "#52514e"


def _axes(n: int):
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), squeeze=False)
    for ax in axes[0]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4df", lw=0.8)
        ax.tick_params(colors=MUTED)
    return fig, axes[0]


def _curves(table, x: str, y: str, ystd: str | None, xlabel: str, ylabel: str, out: Path, rewards: list[str]) -> None:
    fig, axes = _axes(len(rewards))
    for ax, reward in zip(axes, rewards):
        for method, (color, marker, label) in STYLE.items():
            g = table[(table.reward == reward) & (table.method == method)].sort_values(x)
            if g.empty:
                continue
            ax.errorbar(g[x], g[y], yerr=g[ystd] if ystd else None, color=color, marker=marker, ms=7, lw=2,
                        capsize=3, label=label, mec="#fcfcfb", mew=1.5)
        ax.set_title(f"{reward.upper()} reward", color=INK, loc="left", fontsize=11)
        ax.set_xlabel(xlabel, color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    axes[-1].legend(seen.values(), seen.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    """Write reward-vs-M, reward-vs-network-passes, validity-vs-M and value-fn diagnostics figures."""
    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/logs")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else logs.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(logs)
    if not runs:
        raise SystemExit(f"No runs under {logs}")
    rows = runs_to_rows(runs)
    table = summarize(rows)
    # B1 is the M=1 point of every curve; show it on each reward panel
    b1 = table[table.method == "unguided"]
    rewards = sorted(table.reward.unique())
    _curves(table, "M", "reward_mean", "reward_mean_std", "candidates per step / per slot (M = N)",
            "mean reward (invalid = 0)", out / "reward_vs_M.pdf", rewards)
    _curves(table, "nfe_per_molecule", "reward_mean", "reward_mean_std", "network forward passes per molecule",
            "mean reward (invalid = 0)", out / "reward_vs_nfe.pdf", rewards)
    _curves(table, "M", "validity", "validity_std", "candidates per step / per slot (M = N)", "validity",
            out / "validity_vs_M.pdf", rewards)
    # H1: value-function ranking quality by timestep bin, from the cached diagnostics inside run summaries
    bins = {}
    for r in runs:
        vf = r.get("value_fn")
        if vf:
            for b in vf["val"]["by_time"]:
                bins.setdefault((r["reward"], round(b["t_hi"], 2)), []).append(b["spearman"])
    if bins:
        fig, axes = _axes(1)
        ax = axes[0]
        for reward, color in zip(sorted({k[0] for k in bins}), ("#2a78d6", "#eb6834")):
            pts = sorted((t, sum(v) / len(v)) for (rw, t), v in bins.items() if rw == reward)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o", lw=2, label=reward.upper())
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("normalised time t/T (1 = pure noise)", color=MUTED)
        ax.set_ylabel("Spearman(pred, realised reward)", color=MUTED)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "value_fn_spearman_by_t.pdf", dpi=200, facecolor="#fcfcfb")
        plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
