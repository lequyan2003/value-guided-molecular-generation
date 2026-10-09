"""One function that runs one experiment (method x reward x M x seed) and writes its record.

Used by ``scripts/run_inference.py`` (one run), ``scripts/run_ablations.py`` (sweeps) and
``scripts/smoke_test.py``, so every method goes through identical sampling, evaluation and
logging code. A run is identified by a deterministic ``run_id`` and is skipped if finished.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from vgmg.constants import SMOKE_SMILES
from vgmg.eval.baselines import sample_random_selection, sample_unguided, select_best_of
from vgmg.eval.metrics import compute_metrics, per_molecule_rewards
from vgmg.models.denoiser import Denoiser, ToyDenoiser
from vgmg.models.value_function import build_value_function
from vgmg.rl.rewards import get_reward
from vgmg.rl.svdd import pm_scorer, svdd_sample, value_scorer
from vgmg.rl.value_trainer import collect_rollouts, evaluate_value_function, train_value_function
from vgmg.utils.checkpointing import load_checkpoint, save_checkpoint
from vgmg.utils.logging import RunLogger, setup_logging
from vgmg.utils.reproducibility import get_git_commit, seed_streams, set_seed

CHUNK_FILE = "partial.pt"

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
    # the hash makes a changed value_fn config or checkpoint miss the cache instead of silently reusing a stale model
    h = hashlib.sha1(json.dumps([vf, cfg["model"]], sort_keys=True, default=str).encode()).hexdigest()[:8]
    tag = f"{cfg['model']['name']}_{s['reward_type']}_{s['value_fn_arch']}_s{cfg['seed']}_{h}"
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


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def sample_molecules(cfg: dict[str, Any], denoiser: Denoiser, value_fn: torch.nn.Module | None,
                     generator: torch.Generator, run_dir: Path | None = None) -> tuple[list[str], float]:
    """Generate ``eval.num_molecules`` SMILES with ``cfg['method']`` in chunks of ``eval.batch_size``.

    After every chunk the partial result, RNG state, forward-pass count and elapsed time are saved to
    ``run_dir/partial.pt`` (the proposal's "checkpoint every batch"); a preempted run resumes from the last
    chunk and yields exactly the molecules an uninterrupted run would. Returns (smiles, sampling seconds).
    ``best_of_n`` is a pooled family and is handled by ``run_best_of_n``, not here.
    """
    method, s = cfg["method"], cfg["svdd"]
    m, reward_fn = s["num_candidates_m"], get_reward(s["reward_type"])
    n_total, bs = cfg["eval"]["num_molecules"], cfg["eval"]["batch_size"]
    out: list[str] = []
    elapsed = 0.0
    part = run_dir / CHUNK_FILE if run_dir else None
    if part is not None and part.exists():
        st = torch.load(part, weights_only=False)
        out, elapsed, denoiser.nfe = st["smiles"], st["elapsed"], st["nfe"]
        generator.set_state(st["gen_state"])
    while len(out) < n_total:
        n = min(bs, n_total - len(out))
        t0 = time.time()
        if method == "unguided":
            out += sample_unguided(denoiser, n, generator)
        elif method == "random_select":
            out += sample_random_selection(denoiser, n, m, generator)
        elif method in ("svdd_pm", "svdd_value"):
            scorer = pm_scorer(reward_fn) if method == "svdd_pm" else value_scorer(value_fn)
            state = svdd_sample(denoiser, scorer, n, m, generator, s["selection"], s["alpha"])
            out += denoiser.decode(state)
        else:
            raise KeyError(f"Unknown method {method!r}; choose from {METHODS}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed += time.time() - t0
        if part is not None:
            torch.save({"smiles": out, "elapsed": elapsed, "nfe": denoiser.nfe,
                        "gen_state": generator.get_state()}, part)
    return out, elapsed


def _write_record(cfg: dict[str, Any], run_id: str, method: str, reward: str | None, M: int, smiles: list[str],
                  nfe_total: float, wall: float, diag: dict | None, log: RunLogger | None, extra: dict | None = None) -> dict:
    """Score ``smiles`` under every reported reward and write summary.json + molecules.csv for ``run_id``."""
    run_dir = Path(cfg["output_dir"]) / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rewards = cfg["eval"]["report_rewards"]
    ref = reference_smiles(cfg)
    metrics = {r: compute_metrics(smiles, get_reward(r), ref) for r in rewards}
    n = len(smiles)
    summary = {
        "run_id": run_id, "method": method, "seed": cfg["seed"], "model": cfg["model"]["name"],
        "M": M, "reward": reward, "selection": cfg["svdd"]["selection"], "num_molecules": n,
        "nfe_total": nfe_total, "nfe_per_molecule": nfe_total / n, "sample_wall_clock_s": round(wall, 3),
        "gpu": _gpu_name(), "git_commit": get_git_commit(), "value_fn": diag, "metrics": metrics, **(extra or {}),
    }
    summary = json.loads(json.dumps(summary, default=float))  # same types as a reloaded record
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    cols = [per_molecule_rewards(smiles, get_reward(r)) for r in rewards]
    with open(run_dir / "molecules.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", *[f"reward_{r}" for r in rewards]])
        for i, smi in enumerate(smiles):
            w.writerow([smi, *[f"{c[i]:.6f}" for c in cols]])
    head = reward or rewards[0]
    msg = (f"{run_id}: nfe/mol={summary['nfe_per_molecule']:.1f} wall={wall:.1f}s "
           f"validity={metrics[head]['validity']:.3f} reward_mean[{head}]={metrics[head]['reward_mean']:.4f}")
    (log.info if log else print)(msg)
    return summary


def run_best_of_n(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """B2 family: one pool of ``max(m_values)`` unguided draws per slot yields best-of-m for every m and reward.

    Unguided chains do not depend on the reward, so generating them once per seed and selecting per
    (reward, m) is exactly equivalent to separate best-of-m runs but ~3x cheaper (proposal: "B2 is
    reward-independent"). Each (reward, m) gets its own record with cost ``m*T`` passes per molecule
    (the first m draws of each slot); pool wall-clock is split in proportion to m.
    Returns the record for (svdd.reward_type, svdd.num_candidates_m) if present, else the first one.
    """
    ms, rewards, seed = sorted(cfg["best_of_n"]["m_values"]), cfg["eval"]["report_rewards"], cfg["seed"]
    ids = {(r, m): f"best_of_n_{r}_M{m}_s{seed}" for r in rewards for m in ms}
    logs = Path(cfg["output_dir"]) / "logs"
    pick = (cfg["svdd"]["reward_type"], cfg["svdd"]["num_candidates_m"])
    key = pick if pick in ids else next(iter(ids))
    if all((logs / i / "summary.json").exists() for i in ids.values()) and not force:
        return json.loads((logs / ids[key] / "summary.json").read_text())
    set_seed(seed)
    pool_id = f"best_of_n_pool_s{seed}"
    pcfg = {**cfg, "method": "unguided"}
    pcfg["eval"] = {**cfg["eval"], "num_molecules": cfg["eval"]["num_molecules"] * ms[-1]}
    with setup_logging(pool_id, cfg, run_id=pool_id) as log:
        denoiser = build_denoiser(cfg)
        denoiser.nfe = 0
        pool, wall = sample_molecules(pcfg, denoiser, None, seed_streams(seed)["eval"], logs / pool_id)
        n = cfg["eval"]["num_molecules"]
        out = {}
        for (r, m), rid in ids.items():
            chosen = select_best_of(pool, n, ms[-1], m, get_reward(r))
            out[(r, m)] = _write_record(cfg, rid, "best_of_n", r, m, chosen, nfe_total=n * m * denoiser.num_steps,
                                        wall=wall * m / ms[-1], diag=None, log=log, extra={"pool_m": ms[-1]})
    return out[key]


def run_experiment(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Run one experiment and write its record(s) into ``results/logs/<run_id>/``; finished runs are reused."""
    if cfg["method"] == "best_of_n":
        return run_best_of_n(cfg, force)
    run_id = make_run_id(cfg)
    run_dir = Path(cfg["output_dir"]) / "logs" / run_id
    if (run_dir / "summary.json").exists() and not force:
        return json.loads((run_dir / "summary.json").read_text())
    set_seed(cfg["seed"])
    with setup_logging(run_id, cfg, run_id=run_id) as log:
        denoiser = build_denoiser(cfg)
        needs_vf = cfg["method"] == "svdd_value" and cfg["svdd"]["num_candidates_m"] > 1  # M=1 is unguided
        value_fn, diag = get_value_function(cfg, denoiser, log) if needs_vf else (None, None)
        denoiser.nfe = 0
        smiles, wall = sample_molecules(cfg, denoiser, value_fn, seed_streams(cfg["seed"])["eval"], run_dir)
        summary = _write_record(
            cfg, run_id, cfg["method"], None if cfg["method"] == "unguided" else cfg["svdd"]["reward_type"],
            1 if cfg["method"] == "unguided" else cfg["svdd"]["num_candidates_m"], smiles, denoiser.nfe, wall, diag, log)
        (run_dir / CHUNK_FILE).unlink(missing_ok=True)
    return summary
