"""Soft Value-based Decoding (Li et al., 2024), batched.

At each reverse step every parent x_t gets ``m`` children x_{t-1} ~ p(. | x_t); a scorer
assigns each child a value and one child per parent is kept. With ``m = 1`` this is plain
unguided sampling.

Theory in one place (what you must be able to derive):
  optimal tilted kernel   p*(x_{t-1} | x_t)  ∝  p(x_{t-1} | x_t) · exp(v_{t-1}(x_{t-1}) / alpha)
  soft value function     v_t(x_t) = alpha · log E[ exp(r(x_0) / alpha) | x_t ]
SVDD approximates sampling from p* by importance resampling: draw m children from the
pretrained kernel p, weight them by exp(v/alpha), pick one (softmax mode), or take the argmax
(greedy mode, the alpha -> 0 limit). It is exact only as m -> infinity and only as good as v.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import nn

from vgmg.models.denoiser import Denoiser, State
from vgmg.rl.rewards import RewardFn

# A scorer maps (denoiser, children, s) to one score per child, shape (n*m,), where
# ``s`` is the time of the children (s = t-1 after the step from t).
Scorer = Callable[[Denoiser, State, int], torch.Tensor]


@torch.no_grad()
def select_candidates(values: torch.Tensor, mode: Literal["argmax", "softmax"],
                      alpha: float, generator: torch.Generator) -> torch.Tensor:
    """Pick one candidate index per row of ``values`` (shape ``(N, M)``)."""
    if mode == "argmax":
        return values.argmax(dim=1)
    if mode == "softmax":
        return torch.multinomial(torch.softmax(values / alpha, dim=1), 1, generator=generator).squeeze(1)
    raise ValueError(f"Unknown selection mode {mode!r}")


def value_scorer(value_fn: nn.Module) -> Scorer:
    """Score children with a learned value function v_phi(x_s, s/T) (SVDD-MLP/GNN)."""
    @torch.no_grad()
    def score(denoiser: Denoiser, children: State, s: int) -> torch.Tensor:
        feats = denoiser.features(children)
        return value_fn(feats, torch.tensor(s / denoiser.num_steps)).cpu()
    return score


def pm_scorer(reward_fn: RewardFn) -> Scorer:
    """Score children by r(decoded denoiser x_0-prediction) (SVDD-PM, no learned value).

    Costs one extra network forward pass per child (``predict_x0``), so SVDD-PM spends
    about m·T forward passes per molecule, the same as best-of-m.
    """
    def score(denoiser: Denoiser, children: State, s: int) -> torch.Tensor:
        return torch.tensor([reward_fn(smi) for smi in denoiser.predict_x0(children, s)])
    return score


@torch.no_grad()
def svdd_sample(denoiser: Denoiser, scorer: Scorer | None, n: int, m: int,
                generator: torch.Generator, mode: Literal["argmax", "softmax"] = "argmax",
                alpha: float = 0.1) -> State:
    """Run SVDD and return ``n`` final states x_0.

    ``scorer=None`` keeps a uniformly random child (the control that isolates the effect of
    the value signal from the effect of drawing extra candidates). ``m=1`` is unguided.
    """
    state = denoiser.init(n, generator)
    for t in range(denoiser.num_steps, 0, -1):
        children = denoiser.propose(state, t, m, generator)
        if m == 1:
            state = children
            continue
        if scorer is None:
            choice = torch.randint(m, (n,), generator=generator)
        else:
            values = scorer(denoiser, children, t - 1).view(n, m)
            choice = select_candidates(values, mode, alpha, generator)
        state = denoiser.take(children, torch.arange(n) * m + choice)
    return state


@torch.no_grad()
def finish_unguided(denoiser: Denoiser, state: State, t_from: int, generator: torch.Generator) -> State:
    """Continue ``state`` (at time ``t_from``) to x_0 with the pretrained kernel only."""
    for t in range(t_from, 0, -1):
        state = denoiser.propose(state, t, 1, generator)
    return state
