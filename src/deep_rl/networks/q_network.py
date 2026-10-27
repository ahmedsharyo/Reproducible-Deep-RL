from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)

