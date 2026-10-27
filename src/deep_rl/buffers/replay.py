from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool


class ReplayBuffer:
    """Stores true terminal flags; time-limit truncations continue bootstrapping."""
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.data: deque[Transition] = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, terminated: bool) -> None:
        self.data.append(Transition(np.array(observation, copy=True), int(action), float(reward),
                                    np.array(next_observation, copy=True), bool(terminated)))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.choice(len(self.data), size=batch_size, replace=False)
        rows = [self.data[int(i)] for i in indices]
        return {"observations": np.stack([x.observation for x in rows]),
                "actions": np.asarray([x.action for x in rows]), "rewards": np.asarray([x.reward for x in rows], dtype=np.float32),
                "next_observations": np.stack([x.next_observation for x in rows]),
                "terminated": np.asarray([x.terminated for x in rows], dtype=np.float32)}

    def __len__(self) -> int:
        return len(self.data)

