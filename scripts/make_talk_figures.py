"""Talk figures (Slot A): reward vs passes, validity, alpha sweep, value-function Spearman by timestep.

Reads results/tables/summary.csv and results/logs (no GPU). Writes docs/slides/fig/*.png.
Error bars are the std over 3 sampling seeds (one DiGress training seed).
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vgmg.eval.comparison import load_runs

INK, MUTED, GRID = "#1F2933", "#6B7785", "#E4E8EC"
BASE, BON, SVDD, PM = "#8A949E", "#2B5C8A", "#D9622B", "#2E8B6E"
plt.rcParams.update({"font.family": ["Carlito", "DejaVu Sans"], "font.size": 15, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = ROOT / "docs/slides/fig"
OUT.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(ROOT / "results/tables/summary.csv")
t = t[t.reward == "sa"].set_index("method")

# (key, label, colour, marker, label offset (dx, dy) in points, ha)
M = [("unguided", "Unguided", BASE, "o", (0, -22), "center"),
     ("random_select", "Random select", BASE, "o", (0, 12), "center"),
     ("svdd_value_softmax_a0.1", "SVDD-MLP softmax a=0.1", SVDD, "o", (-10, -4), "right"),
     ("svdd_value_softmax_a0.03", "a=0.03", SVDD, "o", (-10, 10), "right"),
     ("svdd_value_softmax_a0.01", "a=0.01", SVDD, "o", (12, 0), "left"),
     ("svdd_value", "SVDD-MLP argmax", SVDD, "X", (12, 0), "left"),
     ("best_of_n", "Best-of-10", BON, "s", (0, 12), "center"),
     ("svdd_pm", "SVDD-PM", PM, "D", (0, 12), "center")]

# 1. one figure, shared rows: passes | SA reward | strict validity (3 seeds, bar end = mean, whisker = seed std)
rows = [("unguided", "Unguided", "500"), ("random_select", "Random select", "500"),
        ("svdd_value_softmax_a0.1", "SVDD-MLP softmax, a = 0.1", "500"), ("svdd_value_softmax_a0.03", "SVDD-MLP softmax, a = 0.03", "500"),
        ("svdd_value_softmax_a0.01", "SVDD-MLP softmax, a = 0.01", "500"), ("svdd_value", "SVDD-MLP argmax (pre-registered)", "500"),
        ("best_of_n", "Best-of-10", "5000"), ("svdd_pm", "SVDD-PM", "5500")]
col = {"unguided": BASE, "random_select": BASE, "best_of_n": BON, "svdd_pm": PM}
fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.3), sharey=True, gridspec_kw={"wspace": 0.06})
n = len(rows)
for ax, (cn, title) in zip(axs, [("reward_mean", "SA reward (invalid = 0)"), ("validity", "Strict validity")]):
    for i, (k, lab, ps) in enumerate(rows):
        y = n - 1 - i
        v, e = t.loc[k, cn], t.loc[k, cn + "_std"]
        ax.barh(y, v, color=col.get(k, SVDD), height=0.62, zorder=2)
        ax.errorbar(v, y, xerr=e, color=INK, lw=1.6, capsize=3, zorder=3)
        ax.text(min(v + e, 1.0) + 0.025, y, f"{v:.2f}", va="center", fontsize=14, color=INK, fontweight="bold")
    ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.5, 1]); ax.set_title(title, loc="left", fontsize=16, color=INK, pad=10)
    ax.grid(axis="x", color=GRID, zorder=0); ax.set_axisbelow(True)
axs[0].axvline(t.loc["unguided", "reward_mean"], color=MUTED, ls="--", lw=1.2, zorder=1)
axs[0].set_yticks(range(n)); axs[0].set_yticklabels([f"{l}" for _, l, _ in rows[::-1]], fontsize=14)
for i, (k, lab, ps) in enumerate(rows):
    axs[1].text(1.30, n - 1 - i, ps, va="center", ha="center", fontsize=14, color=MUTED, clip_on=False)
axs[1].text(1.30, n - 0.35, "passes", ha="center", fontsize=13, color=MUTED, clip_on=False)
fig.subplots_adjust(left=0.30, right=0.93, top=0.90, bottom=0.08)
fig.savefig(OUT / "main_result.png", dpi=220, facecolor="white"); plt.close(fig)

# 3. alpha sweep (argmax = alpha -> 0)
fig, ax = plt.subplots(figsize=(6.6, 4.6))
al = [0.1, 0.03, 0.01]
keys = [f"svdd_value_softmax_a{a}" for a in al]
for col_, lab, c, mk in [("reward_mean", "SA reward", SVDD, "o"), ("validity", "Strict validity", INK, "s")]:
    ax.errorbar(range(4), [t.loc[k, col_] for k in keys] + [t.loc["svdd_value", col_]], yerr=[t.loc[k, col_ + "_std"] for k in keys] + [t.loc["svdd_value", col_ + "_std"]],
                color=c, marker=mk, ms=9, lw=2.4, capsize=3, label=lab)
ax.axhline(t.loc["unguided", "reward_mean"], color=MUTED, ls="--", lw=1.4)
ax.text(3.05, t.loc["unguided", "reward_mean"] + 0.03, "unguided SA 0.635", ha="right", fontsize=12, color=MUTED)
ax.set_xticks(range(4)); ax.set_xticklabels(["0.1", "0.03", "0.01", "argmax\n(a to 0)"])
ax.set_xlabel("softmax temperature a (greedier to the right)"); ax.set_ylim(-0.05, 1.08)
ax.legend(frameon=False, loc="lower left", fontsize=13); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "alpha_sweep.png", dpi=220, facecolor="white"); plt.close(fig)

# 4. value-function Spearman by timestep (12 runs: band = min..max)
runs = load_runs(ROOT / "results/logs")
bins = {}
for r in runs:
    vf = r.get("value_fn")
    if vf and r["reward"] == "sa":
        for b in vf["val"]["by_time"]:
            bins.setdefault(round(b["t_hi"], 2), []).append(b["spearman"])
ts = sorted(bins)
mean = [sum(bins[x]) / len(bins[x]) for x in ts]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.fill_between(ts, [min(bins[x]) for x in ts], [max(bins[x]) for x in ts], color=SVDD, alpha=0.18, lw=0)
ax.plot(ts, mean, color=SVDD, marker="o", ms=9, lw=2.6)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("noise level t/T  (1 = pure noise, 0 = clean)"); ax.set_ylabel("Spearman(value, final reward)")
ax.set_ylim(-0.02, 0.72); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.annotate("0.63", (ts[0], mean[0]), xytext=(8, 10), textcoords="offset points", fontsize=13)
ax.annotate("0.08", (ts[-1], mean[-1]), xytext=(-6, 14), textcoords="offset points", fontsize=13, ha="right")
fig.tight_layout(); fig.savefig(OUT / "spearman.png", dpi=220, facecolor="white"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
