"""Smoke test: end-to-end SVDD on the toy stand-in generator (no download, no checkpoint).

Pipeline: collect unguided rollouts -> train MLP value fn -> SVDD (M>1) vs. M=1 baseline
-> metrics. Same seed => identical numbers. Replace ToyDenoiser with DiGressDenoiser
once the wrapper exists (week 1).
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import parse_config  # noqa: E402
from vgmg.constants import FINGERPRINT_BITS, SMOKE_SMILES  # noqa: E402
from vgmg.eval.baselines import sample_unconditional  # noqa: E402
from vgmg.eval.metrics import compute_metrics  # noqa: E402
from vgmg.models.digress import ToyDenoiser  # noqa: E402
from vgmg.models.value_function import build_value_function  # noqa: E402
from vgmg.rl.rewards import get_reward  # noqa: E402
from vgmg.rl.svdd import svdd_sample  # noqa: E402
from vgmg.rl.value_trainer import collect_rollouts, train_value_function  # noqa: E402
from vgmg.utils.checkpointing import save_checkpoint  # noqa: E402
from vgmg.utils.logging import setup_logging  # noqa: E402
from vgmg.utils.rdkit_utils import morgan_fingerprint  # noqa: E402
from vgmg.utils.reproducibility import set_seed  # noqa: E402


def featurize(smiles: str) -> torch.Tensor:
    """Morgan fingerprint as a tensor."""
    return torch.from_numpy(morgan_fingerprint(smiles, FINGERPRINT_BITS))


def main() -> None:
    """Run the smoke test and print/save a summary."""
    cfg = parse_config("SVDD smoke test", default_config="configs/smoke_test.yaml")
    set_seed(cfg["seed"])
    n, m = cfg["smoke"]["num_molecules"], cfg["svdd"]["num_candidates_m"]
    reward_fn = get_reward(cfg["svdd"]["reward_type"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    denoiser = ToyDenoiser()

    with setup_logging("smoke_test", cfg) as run:
        feats, targets = collect_rollouts(denoiser, featurize, reward_fn, cfg["smoke"]["num_rollouts"], gen)
        vf = build_value_function(cfg["svdd"]["value_fn_arch"], FINGERPRINT_BITS)
        losses = train_value_function(vf, feats, targets, cfg["svdd"]["value_train_steps"], generator=gen)
        run.info(f"Value-fn loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        base = sample_unconditional(denoiser, n, torch.Generator().manual_seed(cfg["seed"] + 1))
        states = svdd_sample(denoiser, vf, featurize, n, m, torch.Generator().manual_seed(cfg["seed"] + 1))
        guided = [denoiser.decode(s) for s in states]

        summary = {
            "seed": cfg["seed"], "M": m, "reward": cfg["svdd"]["reward_type"],
            "baseline": compute_metrics(base, reward_fn, SMOKE_SMILES),
            "svdd": compute_metrics(guided, reward_fn, SMOKE_SMILES),
        }
        run.save_json("summary.json", summary)
        save_checkpoint(vf, run.run_dir / "value_fn_checkpoint.pt", {"seed": cfg["seed"]})
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
