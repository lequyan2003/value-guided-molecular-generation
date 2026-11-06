"""Value-function training and diagnostics.

Training target (SVDD-MC): the state x_t of an *unguided* chain is labelled with r(x_0) of the
molecule that chain ends in, and the network regresses it with MSE. MSE regression estimates
the conditional mean E[r(x_0) | x_t, t], not the soft value alpha·log E[exp(r/alpha) | x_t];
for argmax selection only the ranking of siblings matters, but this is a stated approximation
(see docs/PREREGISTRATION.md, "Approximations"). Splits are made per *chain*, never per row,
because the rows of one chain share a label and would leak across a row-level split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from vgmg.models.denoiser import Denoiser
from vgmg.rl.rewards import RewardFn
from vgmg.rl.svdd import finish_unguided, select_candidates  # noqa: F401  (re-export for callers)


@dataclass
class RolloutData:
    """Training rows: features of x_t, normalised time t/T, regression target r(x_0), chain id."""
    features: torch.Tensor   # (R, D)
    t_frac: torch.Tensor     # (R,)
    targets: torch.Tensor    # (R,)
    chain: torch.Tensor      # (R,) index of the chain the row came from

    def __len__(self) -> int:
        return len(self.targets)


@torch.no_grad()
def collect_rollouts(denoiser: Denoiser, reward_fn: RewardFn, n: int, generator: torch.Generator,
                     ts_per_chain: int = 20, chain_offset: int = 0) -> RolloutData:
    """Roll out ``n`` unguided chains and keep ``ts_per_chain`` random timesteps of each.

    Subsampling timesteps keeps memory at n·ts_per_chain rows instead of n·T. Every chain draws its OWN set of
    distinct timesteps from {1..T} (independent across chains), so the value function sees all noise levels and
    not a handful shared by every row; the features recorded at t are those of the state x_t, i.e. *before* the
    step from t.
    """
    T = denoiser.num_steps
    k = min(ts_per_chain, T)
    chosen = torch.rand(n, T, generator=generator).argsort(dim=1)[:, :k] + 1          # (n, k) distinct times per chain
    keep = torch.zeros(n, T + 1, dtype=torch.bool)
    keep.scatter_(1, chosen, True)
    state = denoiser.init(n, generator)
    feats: list[torch.Tensor] = []
    tfs: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    for t in range(T, 0, -1):
        sel = keep[:, t].nonzero().squeeze(1)
        if len(sel):
            feats.append(denoiser.features(state)[sel])
            tfs.append(torch.full((len(sel),), t / T))
            rows.append(sel)
        state = denoiser.propose(state, t, 1, generator)
    final = torch.tensor([reward_fn(s) for s in denoiser.decode(state)], dtype=torch.float32)
    chain = torch.cat(rows)
    return RolloutData(features=torch.cat(feats), t_frac=torch.cat(tfs), targets=final[chain],
                       chain=chain + chain_offset)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); nan if either side is constant."""
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def evaluate_value_function(model: nn.Module, data: RolloutData, n_bins: int = 5) -> dict:
    """MSE and Spearman on held-out rows, overall and per timestep bin, plus the mean-predictor MSE.

    A value function is only useful if it beats predicting the training mean (``mse_const``).
    """
    model.eval()
    pred = model(data.features, data.t_frac).numpy()
    y = data.targets.numpy()
    out = {
        "mse": float(np.mean((pred - y) ** 2)),
        "mse_const": float(np.var(y)),
        "spearman": spearman(pred, y),
        "by_time": [],
    }
    edges = np.linspace(0, 1, n_bins + 1)
    tf = data.t_frac.numpy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (tf > lo) & (tf <= hi)
        if sel.sum() > 1:
            out["by_time"].append({"t_lo": float(lo), "t_hi": float(hi), "n": int(sel.sum()),
                                   "mse": float(np.mean((pred[sel] - y[sel]) ** 2)),
                                   "spearman": spearman(pred[sel], y[sel])})
    return out


def train_value_function(model: nn.Module, train: RolloutData, val: RolloutData, steps: int,
                         lr: float = 1e-3, batch_size: int = 256, eval_every: int = 50,
                         generator: torch.Generator | None = None) -> dict:
    """Minimise MSE with Adam; keep the weights with the lowest *validation* MSE.

    ``val`` must come from different chains than ``train`` (use a separate rollout stream).
    Returns the loss history and the validation MSE trace.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss: list[float] = []
    val_trace: list[tuple[int, float]] = []
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(len(train), (min(batch_size, len(train)),), generator=generator)
        loss = nn.functional.mse_loss(model(train.features[idx], train.t_frac[idx]), train.targets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(model(val.features, val.t_frac), val.targets).item()
            val_trace.append((step, v))
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return {"train_loss": train_loss, "val_trace": val_trace, "best_val_mse": best}


@torch.no_grad()
def sibling_ranking(denoiser: Denoiser, model: nn.Module, reward_fn: RewardFn, t: int,
                    n_parents: int, m: int, completions: int, generator: torch.Generator) -> dict:
    """Does the value function rank *siblings* correctly? (diagnostic H1)

    SVDD only compares the m children of one parent, so global correlation can look good while
    sibling ranking is at chance (e.g. when the value mostly encodes the timestep or graph size).
    For random parents at time ``t``: draw m children, estimate each child's true value by the
    mean reward of ``completions`` unguided continuations, and report the mean within-parent
    Spearman between predicted value and that estimate. Costs ~ n_parents·m·completions·t NFE.
    """
    T = denoiser.num_steps
    state = denoiser.init(n_parents, generator)
    for s in range(T, t, -1):
        state = denoiser.propose(state, s, 1, generator)
    children = denoiser.propose(state, t, m, generator)
    pred = model(denoiser.features(children), torch.tensor((t - 1) / T)).view(n_parents, m)
    truth = torch.zeros(n_parents * m)
    for _ in range(completions):
        final = finish_unguided(denoiser, children, t - 1, generator)
        truth += torch.tensor([reward_fn(s) for s in denoiser.decode(final)])
    truth = (truth / completions).view(n_parents, m)
    rhos = [spearman(pred[i].numpy(), truth[i].numpy()) for i in range(n_parents)]
    rhos = [r for r in rhos if not np.isnan(r)]
    return {"t": t, "mean_sibling_spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "n_parents_used": len(rhos), "m": m, "completions": completions}


@torch.no_grad()
def value_fn_t0_check(denoiser: Denoiser, model: nn.Module, smiles: list[str], reward_fn: RewardFn) -> dict:
    """Sanity check at t = 0 (proposal Sec. 4): on REAL QM9 molecules (held-out split) the value function,
    evaluated at t = 0, should agree with the true reward. Failing this means the state featurizer or the
    value function cannot represent the reward even for clean graphs, so t > 0 results would be meaningless.
    Uses only held-out molecules; never for training or tuning.
    """
    pred = model(denoiser.features(denoiser.encode(smiles)), torch.tensor(0.0)).numpy()
    truth = np.array([reward_fn(s) for s in smiles])
    return {"n": len(smiles), "spearman": spearman(pred, truth), "mse": float(np.mean((pred - truth) ** 2)),
            "mse_const": float(np.var(truth))}
