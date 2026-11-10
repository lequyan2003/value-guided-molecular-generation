"""Live demo of SVDD vs the baselines, through the same code path as the experiments (``svdd_sample`` & friends).

    python scripts/demo.py --live   [--config configs/default.yaml key=value ...]   generate on the GPU, cache, render
    python scripts/demo.py --replay [--cache results/demo/cache.json]                  render a cached run (instant)
    python scripts/demo.py --toy                                                       CPU, ToyDenoiser: pipeline only

Three views (PNG + text in ``--out``):
  1. denoising movie  : one molecule from noise to final, the network's current guess at 8 timesteps;
  2. one SVDD step    : the M children of one parent at three timesteps with their value-function scores, chosen one marked;
  3. head-to-head     : 16 molecules each from B1 (unguided), B2 (best-of-N, N = M) and SVDD-MLP with rewards and means.
``--toy`` states are fixed SMILES from a pool, NOT generated molecules: its output is labelled and shows mechanics only.
Never present a ``--toy`` or ``--live`` number as a result; results come from the ``talk-snapshot`` run records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vgmg.config import load_config  # noqa: E402
from vgmg.eval.baselines import sample_best_of_n, sample_unguided  # noqa: E402
from vgmg.experiments import build_denoiser, get_value_function  # noqa: E402
from vgmg.rl.rewards import get_reward  # noqa: E402
from vgmg.rl.svdd import select_candidates, svdd_sample, value_scorer  # noqa: E402
from vgmg.utils.reproducibility import set_seed  # noqa: E402

N_SNAPSHOTS = 8
STEP_FRACS = (0.75, 0.4, 0.1)   # view 2: noise levels t/T at which one SVDD step is shown
N_GRID = 16                     # view 3: molecules per method


def denoising_movie(den: Any, gen: torch.Generator) -> list[dict]:
    """One unguided chain; at ``N_SNAPSHOTS`` times record the network's current guess of the clean molecule."""
    T = den.num_steps
    times = sorted({max(1, round(T * i / (N_SNAPSHOTS - 1))) for i in range(N_SNAPSHOTS)}, reverse=True)
    state, frames = den.init(1, gen), []
    for t in range(T, 0, -1):
        if t in times:
            frames.append({"t": t, "smiles": den.predict_x0(state, t)[0]})
        state = den.propose(state, t, 1, gen)
    frames.append({"t": 0, "smiles": den.decode(state)[0]})
    return frames


def svdd_steps(den: Any, scorer: Any, m: int, gen: torch.Generator, mode: str = "argmax", alpha: float = 0.1) -> list[dict]:
    """One SVDD chain (n = 1, selection ``mode`` at temperature ``alpha``); at three timesteps record all M children, their scores and the choice."""
    T = den.num_steps
    shown = {max(1, round(f * T)) for f in STEP_FRACS}
    state, out = den.init(1, gen), []
    for t in range(T, 0, -1):
        children = den.propose(state, t, m, gen)
        values = scorer(den, children, t - 1).view(1, m)
        choice = int(select_candidates(values, mode, alpha, gen)[0])
        if t in shown:
            out.append({"t": t, "children": den.predict_x0(children, t - 1), "values": values[0].tolist(), "chosen": choice})
        state = den.take(children, torch.tensor([choice]))
    return out


def head_to_head(den: Any, scorer: Any, reward_fn: Any, m: int, seed: int, mode: str = "argmax", alpha: float = 0.1) -> dict[str, list[dict]]:
    """16 molecules per method with their true rewards (invalid = 0), each method on its own seeded stream."""
    gens = {k: torch.Generator().manual_seed(seed + i) for i, k in enumerate(("b1", "b2", "svdd"))}
    mols = {"B1 unguided": sample_unguided(den, N_GRID, gens["b1"]),
            f"B2 best-of-{m}": sample_best_of_n(den, reward_fn, N_GRID, m, gens["b2"]),
            f"SVDD-MLP M={m} {mode}" + (f" a={alpha}" if mode == "softmax" else ""): den.decode(svdd_sample(den, scorer, N_GRID, m, gens["svdd"], mode, alpha))}
    from vgmg.utils.rdkit_utils import is_valid
    return {k: [{"smiles": s, "reward": reward_fn(s) if is_valid(s) else 0.0} for s in v] for k, v in mols.items()}


def build(cfg: dict, den: Any, value_fn: Any, seed: int) -> dict:
    """Run all three views and return plain, JSON-serialisable data (this is what ``--replay`` re-renders)."""
    m, reward_fn = cfg["svdd"]["num_candidates_m"], get_reward(cfg["svdd"]["reward_type"])
    scorer = value_scorer(value_fn)
    mode, alpha = cfg["svdd"].get("selection", "argmax"), float(cfg["svdd"].get("alpha", 0.1))
    return {"model": cfg["model"]["name"], "reward": cfg["svdd"]["reward_type"], "M": m, "seed": seed, "T": den.num_steps,
            "movie": denoising_movie(den, torch.Generator().manual_seed(seed)),
            "steps": svdd_steps(den, scorer, m, torch.Generator().manual_seed(seed + 100), mode, alpha),
            "grid": head_to_head(den, scorer, reward_fn, m, seed + 200, mode, alpha)}


def _grid_png(smiles: list[str], legends: list[str], path: Path, per_row: int) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw
    mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
    blank = Chem.MolFromSmiles("")
    img = Draw.MolsToGridImage([m or blank for m in mols], molsPerRow=per_row, subImgSize=(230, 190), legends=legends)
    img.save(str(path)) if hasattr(img, "save") else path.write_bytes(img)


def render(data: dict, out: Path) -> None:
    """Write the three PNGs and print the text summary."""
    from vgmg.utils.rdkit_utils import is_valid
    out.mkdir(parents=True, exist_ok=True)
    tag = "TOY: pipeline only, not molecules" if data["model"] == "toy" else f"DiGress, seed {data['seed']}"
    print(f"== {tag} | reward {data['reward']} | M = {data['M']} | T = {data['T']} ==")
    mv = data["movie"]
    _grid_png([f["smiles"] for f in mv], [f"t={f['t']}" for f in mv], out / "1_denoising_movie.png", 5)
    print("1. denoising movie:", " -> ".join(f"t{f['t']}:{f['smiles'] or 'invalid'}" for f in mv))
    for st in data["steps"]:
        legends = [f"{'CHOSEN ' if i == st['chosen'] else ''}v={v:.3f}" for i, v in enumerate(st["values"])]
        _grid_png(st["children"], legends, out / f"2_svdd_step_t{st['t']}.png", 5)
        spread = max(st["values"]) - min(st["values"])
        print(f"2. step t={st['t']}: value spread among the {len(legends)} children = {spread:.4f}"
              f" ({'nearly constant: guidance has little to choose from' if spread < 1e-3 else 'children are distinguished'})")
    for name, rows in data["grid"].items():
        _grid_png([r["smiles"] for r in rows], [f"r={r['reward']:.2f}" for r in rows],
                  out / f"3_{name.split()[0].lower().replace('-', '_')}_{name.split()[-1].replace('=', '')}.png", 4)
        mean = sum(r["reward"] for r in rows) / len(rows)
        print(f"3. {name:<18} mean reward over {len(rows)} molecules: {mean:.3f}"
              f"  (valid {sum(r['reward'] > 0 or is_valid(r['smiles']) for r in rows)}/{len(rows)})")
    print(f"images in {out}")


def main() -> None:
    """Parse the mode, build or load the demo data, render it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    for k in ("live", "replay", "toy"):
        mode.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", type=Path, default=Path("results/demo/cache.json"))
    ap.add_argument("--out", type=Path, default=Path("results/demo"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    if args.replay:
        render(json.loads(args.cache.read_text()), args.out)
        return
    over = (["model.name=toy", "data.reference=smoke", "device=cpu", "output_dir=./results/demo_toy"] if args.toy else []) + args.overrides
    cfg = load_config(args.config, over + [f"seed={args.seed}"])
    set_seed(args.seed)
    den = build_denoiser(cfg)
    value_fn, _ = get_value_function(cfg, den)      # cached after the first call; training only if no cache exists
    data = build(cfg, den, value_fn, args.seed)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(data, indent=2))
    render(data, args.out)


if __name__ == "__main__":
    main()
