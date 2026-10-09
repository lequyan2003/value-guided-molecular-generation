"""Reward statistics of REAL QM9 molecules (held-out 1k test split): reward-code sanity check and headroom reference.

Never used for training or tuning. A guided method that pushes generated rewards far above the
real-data distribution should be inspected for reward hacking (e.g. collapse to tiny molecules).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.constants import RESULTS_DIR  # noqa: E402
from vgmg.data.loaders import load_qm9_smiles, load_split  # noqa: E402
from vgmg.eval.metrics import compute_metrics  # noqa: E402
from vgmg.rl.rewards import get_reward  # noqa: E402


def main() -> None:
    """Write results/tables/qm9_reference.json."""
    smiles = load_qm9_smiles()
    test = [smiles[i] for i in load_split("test")]
    out = {r: compute_metrics(test, get_reward(r), []) for r in ("sa", "qed")}
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables" / "qm9_reference.json").write_text(json.dumps(out, indent=2))
    for r, m in out.items():
        print(f"{r}: validity={m['validity']:.3f} mean={m['reward_mean']:.3f} std={m['reward_std']:.3f} "
              f"top10={m['reward_top10']:.3f} diversity={m['diversity']:.3f}")


if __name__ == "__main__":
    main()
