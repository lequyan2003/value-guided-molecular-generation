"""Slot A headline figure: SA reward, validity and uniqueness per method (3 seeds, error bar = seed std).

Reads results/tables/summary.csv (made by scripts/generate_results_table.py). No GPU needed.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / "results/tables/summary.csv")
t = t[t.reward == "sa"].set_index("method")
order = [
    ("unguided", "Unguided\n(500 passes)"),
    ("random_select", "Random\nselect"),
    ("best_of_n", "Best-of-10\n(5000)"),
    ("svdd_value", "SVDD-MLP\nargmax"),
    ("svdd_value_softmax_a0.1", "SVDD-MLP\nsoftmax a=0.1"),
    ("svdd_value_softmax_a0.03", "SVDD-MLP\nsoftmax a=0.03"),
    ("svdd_value_softmax_a0.01", "SVDD-MLP\nsoftmax a=0.01"),
    ("svdd_pm", "SVDD-PM\n(5500)"),
]
order = [(k, l) for k, l in order if k in t.index]
panels = [("reward_mean", "SA reward (higher = better)"), ("validity", "Strict validity"), ("uniqueness", "Uniqueness")]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (col, title) in zip(axs, panels):
    xs = range(len(order))
    ys = [t.loc[k, col] for k, _ in order]
    es = [t.loc[k, col + "_std"] for k, _ in order]
    colors = ["#888888" if k in ("unguided", "random_select") else "#2a6fbb" if k == "best_of_n" else "#d95f02" for k, _ in order]
    ax.bar(xs, ys, yerr=es, color=colors, capsize=2)
    ax.set_xticks(list(xs)); ax.set_xticklabels([l for _, l in order], rotation=60, ha="right", fontsize=8)
    ax.set_title(title); ax.set_ylim(0, 1.05)
    if col == "reward_mean":
        ax.axhline(t.loc["unguided", col], color="k", ls="--", lw=0.8)
fig.tight_layout()
out = ROOT / "results/figures"
fig.savefig(out / "slot_a_headline.pdf"); fig.savefig(out / "slot_a_headline.png", dpi=200)
print("wrote", out / "slot_a_headline.png")
