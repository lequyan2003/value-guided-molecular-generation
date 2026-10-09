"""Value-function training: collect (x_t, reward(x_0)) pairs from rollouts, then regress."""

from typing import Callable

import torch
from torch import nn

from vgmg.models.digress import Denoiser
from vgmg.rl.svdd import Featurizer


@torch.no_grad()
def collect_rollouts(denoiser: Denoiser, featurize: Featurizer,
                     reward_fn: Callable[[str], float], n: int,
                     generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out ``n`` unguided chains; return features of every x_t and the final reward.

    Returns ``(features (n*T, D), targets (n*T,))`` where each x_t is labeled with the
    reward of the x_0 its own chain ends in (Monte-Carlo regression target).
    """
    states = denoiser.init(n, generator)
    traj = [list(states)]
    for t in range(denoiser.num_steps, 0, -1):
        states = [c[0] for c in denoiser.propose(states, t, 1, generator)]
        traj.append(list(states))
    final_reward = torch.tensor([reward_fn(denoiser.decode(s)) for s in states])
    feats = torch.stack([featurize(s) for step in traj[:-1] for s in step])
    targets = final_reward.repeat(len(traj) - 1)
    return feats, targets


def train_value_function(model: nn.Module, features: torch.Tensor, targets: torch.Tensor,
                         steps: int, lr: float = 1e-3, batch_size: int = 256,
                         generator: torch.Generator | None = None) -> list[float]:
    """Minimize MSE with Adam for ``steps`` minibatch steps; return the loss history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        idx = torch.randint(len(features), (min(batch_size, len(features)),), generator=generator)
        loss = nn.functional.mse_loss(model(features[idx]), targets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses
