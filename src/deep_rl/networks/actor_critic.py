from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        def mlp(output_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            width = observation_dim
            for size in hidden_sizes:
                layers.extend((nn.Linear(width, size), nn.Tanh()))
                width = size
            layers.append(nn.Linear(width, output_dim))
            return nn.Sequential(*layers)
        self.actor = mlp(action_dim)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def sample(self, observations: torch.Tensor, deterministic: bool = False):
        mean = self.actor(observations)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        latent = mean if deterministic else dist.rsample()
        action = torch.tanh(latent)
        log_prob = dist.log_prob(latent).sum(-1)
        # Change of variables for tanh; clamp avoids log(0) at saturation.
        log_prob -= torch.log(1 - action.square() + 1e-6).sum(-1)
        return action, log_prob, self.value(observations)

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        # Stored actions are bounded. atanh recovers the latent policy sample.
        bounded = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        latent = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        mean = self.actor(observations)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        log_prob = dist.log_prob(latent).sum(-1) - torch.log(1 - bounded.square() + 1e-6).sum(-1)
        # Base Gaussian entropy is a stable diagnostic; exact squashed entropy is costly.
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, self.value(observations)

