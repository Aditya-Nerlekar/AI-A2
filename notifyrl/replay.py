"""Uniform experience replay stored in pre-allocated numpy ring arrays."""
from __future__ import annotations

from typing import Dict

import numpy as np


class TransitionMemory:
    """Fixed-capacity circular store of (s, a, r, s', done) tuples.

    Arrays are allocated once, so inserting is O(1) with no Python object
    churn, and sampling returns contiguous float32 batches ready for torch.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def push(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        i = self._write
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._write = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self._size < batch_size:
            raise ValueError(f"only {self._size} transitions stored, need {batch_size}")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx], "actions": self.actions[idx], "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx], "dones": self.dones[idx],
        }
