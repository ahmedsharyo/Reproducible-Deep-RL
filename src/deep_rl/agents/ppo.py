from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.deep_rl.networks.actor_critic import ActorCritic


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    num_envs: int = 1
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    hidden_sizes: tuple[int, ...] = (64, 64)


class PPOAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: PPOConfig, device: torch.device):
        self.config, self.device = config, device
        self.network = ActorCritic(observation_dim, action_dim, config.hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, eps=1e-5)

    @torch.no_grad()
    def act(self, observation, deterministic: bool = False):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        action, log_prob, value = self.network.sample(obs, deterministic)
        return action.cpu().numpy(), float(log_prob.item()), float(value.item())

    def optimize(self, observations, actions, old_log_probs, advantages, returns) -> dict[str, float]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_lp = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = len(obs)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        count = 0
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(n, device=self.device)
            epoch_kls = []
            for ids in indices.split(self.config.minibatch_size):
                new_lp, entropy, value = self.network.evaluate_actions(obs[ids], act[ids])
                log_ratio = new_lp - old_lp[ids]
                ratio = log_ratio.exp()
                pg1 = -adv[ids] * ratio
                pg2 = -adv[ids] * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                policy_loss = torch.maximum(pg1, pg2).mean()
                value_loss = 0.5 * (value - ret[ids]).square().mean()
                entropy_mean = entropy.mean()
                total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_coef).float().mean()
                vals = (policy_loss, value_loss, entropy_mean, approx_kl, clip_fraction)
                for key, val in zip(metrics, vals):
                    metrics[key] += float(val.item())
                count += 1
                epoch_kls.append(float(approx_kl.item()))
            if self.config.target_kl is not None and epoch_kls and np.mean(epoch_kls) > self.config.target_kl:
                break
        return {key: value / max(count, 1) for key, value in metrics.items()}

