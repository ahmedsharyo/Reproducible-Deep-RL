from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from src.deep_rl.buffers.replay import ReplayBuffer
from src.deep_rl.networks.q_network import QNetwork


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    capacity: int = 100_000
    learning_starts: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    target_update_frequency: int = 500
    hidden_sizes: tuple[int, ...] = (128, 128)
    double_dqn: bool = False
    target_tau: float = 1.0
    max_grad_norm: float = 10.0


class DQNAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: DQNConfig, device: torch.device):
        self.config, self.device, self.action_dim = config, device, action_dim
        self.online = QNetwork(observation_dim, action_dim, config.hidden_sizes).to(device)
        self.target = QNetwork(observation_dim, action_dim, config.hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.capacity)
        self.steps = 0

    def epsilon(self, step: int) -> float:
        fraction = min(step / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, observation, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(obs).argmax(dim=-1).item())

    def bellman_targets(self, rewards, next_observations, terminated):
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        terminal = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online(next_obs).argmax(dim=-1)
                next_q = self.target(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=-1).values
            return rewards + self.config.gamma * (1.0 - terminal) * next_q

    def update(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.learning_starts):
            return None
        batch = self.replay.sample(self.config.batch_size)
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self.bellman_targets(batch["rewards"], batch["next_observations"], batch["terminated"])
        loss = nn.functional.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.config.target_update_frequency == 0:
            with torch.no_grad():
                for target, online in zip(self.target.parameters(), self.online.parameters()):
                    target.lerp_(online, self.config.target_tau)
        return float(loss.item())

