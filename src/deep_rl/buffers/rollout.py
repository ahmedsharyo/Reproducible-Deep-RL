from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Rollout:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_values: np.ndarray

    def advantages_and_returns(self, gamma: float, gae_lambda: float):
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        carry = np.zeros(self.rewards.shape[1:], dtype=np.float32)
        for t in reversed(range(len(self.rewards))):
            # Termination prevents bootstrap. Truncation gets a value bootstrap but
            # ends the GAE trace because subsequent samples may be a reset episode.
            bootstrap = 1.0 - self.terminated[t]
            delta = self.rewards[t] + gamma * bootstrap * self.next_values[t] - self.values[t]
            continuation = 1.0 - np.logical_or(self.terminated[t], self.truncated[t])
            carry = delta + gamma * gae_lambda * continuation * carry
            advantages[t] = carry
        return advantages, advantages + self.values

