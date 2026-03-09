"""Uniform replay buffer implementation for Task 2 D3QN."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReplayTransition:
    """Container for one replay-buffer transition.

    Attributes:
        state (Any): Current state observation.
        action (int): Action index applied to the current state.
        reward (float): Immediate reward from the transition.
        next_state (Any): Observation after the action.
        done (bool): Episode termination flag for the transition.
    """

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer with FIFO overwrite semantics."""

    def __init__(self, capacity: int) -> None:
        """Initialize an empty replay buffer.

        Args:
            capacity (int): Maximum number of transitions retained in memory.

        Returns:
            None
        """

        self.capacity = int(capacity)
        self._storage: deque[ReplayTransition] = deque(maxlen=self.capacity)

    def add(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Add one transition to the buffer.

        Args:
            state (Any): Current state observation.
            action (int): Action index taken from `state`.
            reward (float): Immediate reward after the action.
            next_state (Any): Next state observation.
            done (bool): Whether the transition ends the episode.

        Returns:
            None
        """

        self._storage.append(
            ReplayTransition(
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> tuple[list[Any], np.ndarray, np.ndarray, list[Any], np.ndarray]:
        """Sample a mini-batch of transitions uniformly at random.

        Args:
            batch_size (int): Number of transitions to draw.

        Returns:
            tuple[list[Any], np.ndarray, np.ndarray, list[Any], np.ndarray]:
                A tuple of `(states, actions, rewards, next_states, dones)` where:
                - `states` (list[Any]): Sampled state observations.
                - `actions` (np.ndarray): Action indices with shape `(batch_size,)`.
                - `rewards` (np.ndarray): Rewards with shape `(batch_size,)` and dtype `float32`.
                - `next_states` (list[Any]): Sampled next-state observations.
                - `dones` (np.ndarray): Done flags with shape `(batch_size,)` and dtype `float32`.

        Raises:
            ValueError: If sampling is requested with insufficient stored transitions.
        """

        if batch_size > len(self._storage):
            raise ValueError(
                f"Cannot sample batch size {batch_size} from buffer of size {len(self._storage)}."
            )

        indices = np.random.choice(len(self._storage), size=batch_size, replace=False)
        sampled = [self._storage[index] for index in indices]

        states = [transition.state for transition in sampled]
        actions = np.array([transition.action for transition in sampled], dtype=np.int64)
        rewards = np.array([transition.reward for transition in sampled], dtype=np.float32)
        next_states = [transition.next_state for transition in sampled]
        dones = np.array([float(transition.done) for transition in sampled], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the number of transitions currently stored.

        Args:
            None

        Returns:
            int: Number of stored transitions.
        """

        return len(self._storage)
