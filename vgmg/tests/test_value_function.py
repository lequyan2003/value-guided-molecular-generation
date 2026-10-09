"""Value-function training reduces loss."""

import torch

from vgmg.models.value_function import MLPValueFunction
from vgmg.rl.value_trainer import train_value_function


def test_training_reduces_loss() -> None:
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    x = torch.randint(0, 2, (256, 16), generator=gen).float()
    y = x[:, :4].mean(dim=1)  # learnable linear target
    losses = train_value_function(MLPValueFunction(16), x, y, steps=200, generator=gen)
    assert losses[-1] < 0.5 * losses[0]
