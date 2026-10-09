"""One function that runs one experiment (method x reward x M x seed) and writes its record.

Used by ``scripts/run_inference.py`` (one run), ``scripts/run_ablations.py`` (sweeps) and
``scripts/smoke_test.py``, so every method goes through identical sampling, evaluation and
logging code. A run is identified by a deterministic ``run_id`` and is skipped if finished.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from vgmg.constants import SMOKE_SMILES
from vgmg.eval.baselines import sample_best_of_n, sample_random_selection, sample_unguided
from vgmg.eval.metrics import compute_metrics, per_molecule_rewards
from vgmg.models.denoiser import Denoiser, ToyDenoiser
from vgmg.models.value_function import build_value_function
from vgmg.rl.rewards import get_reward
from vgmg.rl.svdd import pm_scorer, svdd_sample, value_scorer
from vgmg.rl.value_trainer import collect_rollouts, evaluate_value_function, train_value_function
from vgmg.utils.checkpointing import load_checkpoint, save_checkpoint
from vgmg.utils.logging import RunLogger, setup_logging
from vgmg.utils.reproducibility import seed_streams, set_seed

METHODS = ("unguided", "best_of_n", "svdd_pm", "svdd_value", "random_select")


def build_denoiser(cfg: dict[str, Any]) -> Denoiser:
    """Instantiate the generator named in ``cfg['model']['name']``."""
    name = cfg["model"]["name"]
    if name == "toy":
        return ToyDenoiser(num_steps=cfg["model"].get("toy_num_steps", 4))
    if name == "digress":
        from vgmg.models.digress import DiGressDenoiser
        return DiGressDenoiser(cfg["model"]["digress_checkpoint"], cfg.get("device", "cuda"))
    raise KeyError(f"Unknown model {name!r}")


def reference_smiles(cfg: dict[str, Any]) -> list[str]:
    """Molecules novelty is measured against: full QM9, or the smoke pool for the toy."""
    if cfg["data"]["reference"] == "smoke":
        return list(SMOKE_SMILES)
    from vgmg.data.loaders import load_qm9_smiles
    return load_qm9_smiles()


def make_run_id(cfg: dict[str, Any]) -> str:
    """Deterministic, human-readable id: one directory per (method, reward, M, ...) and seed."""
    method, s, seed = cfg["method"], cfg["svdd"], cfg["seed"]
    if method == "unguided":
        return f"unguided_s{seed}"
    parts = [method, s["reward_type"], f"M{s['num_candidates_m']}"]
    if method == "svdd_value":
        parts.insert(1, s["value_fn_arch"])
    if method in ("svdd_value", "svdd_pm") and s["selection"] == "softmax":
        parts.append(f"softmax{s['alpha']}")
    return "_".join(parts) + f"_s{seed}"


def value_fn_dir(cfg: dict[str, Any]) -> Path:
    """Where trained value functions are cached."""
    return Path(cfg["output_dir"]) / "value_fns"


def get_value_function(cfg: dict[str, Any], denoiser: Denoiser, log: RunLogger | None = None) -> tuple[torch.nn.Module, dict]:
    """Load the cached value function for (model, reward, arch, seed) or train and cache it.

    Training and validation rollouts come from different RNG streams than evaluation
    (``seed_streams``), and validation chains are disjoint from training chains.
    """
    s, vf = cfg["svdd"], cfg["value_fn"]
    tag = f"{cfg['model']['name']}_{s['reward_type']}_{s['value_fn_arch']}_s{cfg['seed']}"
    ckpt, diag_path = value_fn_dir(cfg) / f"{tag}.pt", value_fn_dir(cfg) / f"{tag}.json"
    model = build_value_function(s["value_fn_arch"], denoiser.feature_dim, vf["hidden_dim"], vf["num_hidden"])
    if ckpt.exists():
        load_checkpoint(model, ckpt)
        model.eval()
        return model, json.loads(diag_path.read_text())
    streams = seed_streams(cfg["seed"])
    torch.manual_seed(int(torch.randint(2**31 - 1, (1,), generator=streams["value_init"])))
    reward_fn = get_reward(s["reward_type"])
    nfe0 = denoiser.nfe
    train = collect_rollouts(denoiser, reward_fn, vf["train_chains"], streams["rollout_train"], vf["ts_per_chain"])
    val = collect_rollouts(denoiser, reward_fn, vf["val_chains"], streams["rollout_val"], vf["ts_per_chain"],
                           chain_offset=vf["train_chains"])
    assert set(train.chain.tolist()).isdisjoint(val.chain.tolist()), "train/val chains must be disjoint"
    hist = train_value_function(model, train, val, vf["train_steps"], vf["lr"], vf["batch_size"],
                                vf["eval_every"], streams["value_init"])
    diag = {"best_val_mse": hist["best_val_mse"], "val_trace": hist["val_trace"],
            "val": evaluate_value_function(model, val), "train_nfe": denoiser.nfe - nfe0,
            "n_train_rows": len(train), "n_val_rows": len(val)}
    value_fn_dir(cfg).mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, ckpt, {"seed": cfg["seed"], "tag": tag})
    diag_path.write_text(json.dumps(diag, indent=2, default=float))
    if log:
        log.info(f"Value fn {tag}: val MSE {diag['val']['mse']:.4f} vs constant {diag['val']['mse_const']:.4f}, "
                 f"spearman {diag['val']['spearman']:.3f}")
    return model, diag


def sample_molecules(cfg: dict[str, Any], denoiser: Denoiser, value_fn: torch.nn.Module | None,
                     generator: torch.Generator) -> list[str]:
    """Generate ``eval.num_molecules`` SMILES with ``cfg['method']``, in chunks of ``eval.batch_size``."""
    method, s = cfg["method"], cfg["svdd"]
    m, reward_fn = s["num_candidates_m"], get_reward(s["reward_type"])
    n_total, bs = cfg["eval"]["num_molecules"], cfg["eval"]["batch_size"]
    out: list[str] = []
    for start in range(0, n_total, bs):
        n = min(bs, n_total - start)
        if method == "unguided":
            out += sample_unguided(denoiser, n, generator)
        elif method == "best_of_n":
            out += sample_best_of_n(denoiser, reward_fn, n, m, generator)
        elif method == "random_select":
            out += sample_random_selection(denoiser, n, m, generator)
        elif method in ("svdd_pm", "svdd_value"):
            scorer = pm_scorer(reward_fn) if method == "svdd_pm" else value_scorer(value_fn)
            state = svdd_sample(denoiser, scorer, n, m, generator, s["selection"], s["alpha"])
            out += denoiser.decode(state)
        else:
            raise KeyError(f"Unknown method {method!r}; choose from {METHODS}")
    return out


def run_experiment(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Run one experiment and write ``summary.json`` and ``molecules.csv`` into its run dir."""
    run_id = make_run_id(cfg)
    run_dir = Path(cfg["output_dir"]) / "logs" / run_id
    if (run_dir / "summary.json").exists() and not force:
        return json.loads((run_dir / "summary.json").read_text())
    set_seed(cfg["seed"])
    with setup_logging(run_id, cfg, run_id=run_id) as log:
        denoiser = build_denoiser(cfg)
        value_fn, diag = (get_value_function(cfg, denoiser, log) if cfg["method"] == "svdd_value" else (None, None))
        streams = seed_streams(cfg["seed"])
        denoiser.nfe = 0
        t0 = time.time()
        smiles = sample_molecules(cfg, denoiser, value_fn, streams["eval"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.time() - t0
        ref = reference_smiles(cfg)
        rewards = cfg["eval"]["report_rewards"]
        metrics = {r: compute_metrics(smiles, get_reward(r), ref) for r in rewards}
        n = len(smiles)
        summary = {
            "run_id": run_id, "method": cfg["method"], "seed": cfg["seed"], "model": cfg["model"]["name"],
            "M": 1 if cfg["method"] == "unguided" else cfg["svdd"]["num_candidates_m"],
            "reward": None if cfg["method"] == "unguided" else cfg["svdd"]["reward_type"],
            "selection": cfg["svdd"]["selection"], "num_molecules": n,
            "nfe_total": denoiser.nfe, "nfe_per_molecule": denoiser.nfe / n,
            "sample_wall_clock_s": round(wall, 3), "value_fn": diag, "metrics": metrics,
        }
        summary = json.loads(json.dumps(summary, default=float))  # same types as a reloaded record
        log.save_json("summary.json", summary)
        with open(run_dir / "molecules.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["smiles", *[f"reward_{r}" for r in rewards]])
            cols = [per_molecule_rewards(smiles, get_reward(r)) for r in rewards]
            for i, smi in enumerate(smiles):
                w.writerow([smi, *[f"{c[i]:.6f}" for c in cols]])
        head = cfg["svdd"]["reward_type"] if cfg["method"] != "unguided" else rewards[0]
        log.info(f"{run_id}: nfe/mol={summary['nfe_per_molecule']:.1f} wall={wall:.1f}s "
                 f"validity={metrics[head]['validity']:.3f} reward_mean[{head}]={metrics[head]['reward_mean']:.4f}")
    return summary
