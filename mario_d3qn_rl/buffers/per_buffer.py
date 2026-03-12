"""Prioritized experience replay buffer implementation using a Sum Tree."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SumTree:
    """Binary tree storing prefix sums for proportional priority sampling."""

    def __init__(self, capacity: int) -> None:
        """Initialize an empty Sum Tree.

        Args:
            capacity (int): Maximum number of replay entries stored in the tree.

        Returns:
            None
        """

        self.capacity = int(capacity)
        self.tree_capacity = 1
        while self.tree_capacity < self.capacity:
            self.tree_capacity *= 2

        self.leaf_start = self.tree_capacity - 1
        self.depth = int(math.log2(self.tree_capacity))
        self.tree = np.zeros(2 * self.tree_capacity - 1, dtype=np.float32)
        self.data: list[int | None] = [None] * self.capacity
        self.write_index = 0
        self.size = 0
        self.current_max_priority = 0.0

    def add(self, priority: float, data_index: int) -> int:
        """Insert a replay-slot index with the provided priority.

        Args:
            priority (float): Priority assigned to the transition.
            data_index (int): Replay-buffer slot index associated with the priority.

        Returns:
            int: Leaf index used by the Sum Tree.
        """

        data_index = int(data_index)
        leaf_index = self.leaf_start + data_index

        self.data[data_index] = int(data_index)
        self.update(leaf_index, priority)

        self.write_index = (data_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return leaf_index

    def update(self, leaf_index: int, priority: float) -> None:
        """Update one stored priority and propagate the change up the tree.

        Args:
            leaf_index (int): Tree leaf index associated with a stored transition.
            priority (float): New priority value.

        Returns:
            None
        """

        priority = float(priority)
        change = priority - float(self.tree[leaf_index])
        self.tree[leaf_index] = priority
        self.current_max_priority = max(self.current_max_priority, priority)

        while leaf_index != 0:
            leaf_index = (leaf_index - 1) // 2
            self.tree[leaf_index] += change

    def sample(self, value: float) -> tuple[int, float, int]:
        """Sample one transition by descending the tree with a prefix-sum value.

        Args:
            value (float): Prefix-sum query in `[0, total_priority)`.

        Returns:
            tuple[int, float, int]:
                Leaf index, stored priority, and sampled replay-buffer index.
        """

        while True:
            parent_index = 0
            search_value = float(value)

            while True:
                left_child = 2 * parent_index + 1
                right_child = left_child + 1

                if left_child >= len(self.tree):
                    leaf_index = parent_index
                    break

                if search_value <= self.tree[left_child]:
                    parent_index = left_child
                else:
                    search_value -= float(self.tree[left_child])
                    parent_index = right_child

            data_index = leaf_index - self.leaf_start
            if 0 <= data_index < self.capacity:
                stored_index = self.data[data_index]
                priority = float(self.tree[leaf_index])
                if stored_index is not None and priority > 0.0:
                    return leaf_index, priority, int(stored_index)

            value = float(np.nextafter(value, 0.0))
            if value <= 0.0:
                raise ValueError("Attempted to sample an empty Sum Tree leaf.")

    @property
    def total_priority(self) -> float:
        """Return the total priority stored in the tree.

        Args:
            None

        Returns:
            float: Sum of all leaf priorities.
        """

        return float(self.tree[0])

    def max_priority(self) -> float:
        """Return the maximum leaf priority among stored entries.

        Args:
            None

        Returns:
            float: Maximum stored priority, or `0.0` if the tree is empty.
        """

        if self.size == 0:
            return 0.0

        return float(self.current_max_priority)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer with proportional sampling and IS weights."""

    def __init__(
        self,
        capacity: int,
        alpha: float,
        epsilon: float,
        state_shape: tuple[int, ...] | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the prioritized replay buffer.

        Args:
            capacity (int): Maximum number of stored transitions.
            alpha (float): Prioritization exponent.
            epsilon (float): Small constant added to TD errors.
            state_shape (tuple[int, ...] | None): Optional observation shape for array-backed storage.
            batch_size (int): Expected sample batch size, used to pre-allocate
                output buffers. Defaults to 32.

        Returns:
            None
        """

        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.sum_tree = SumTree(capacity=self.capacity)
        self._state_shape = state_shape
        self._batch_size_hint = int(batch_size)

        if state_shape is not None:
            self._states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
            self._next_states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
            self._actions = np.zeros(self.capacity, dtype=np.int64)
            self._rewards = np.zeros(self.capacity, dtype=np.float32)
            self._dones = np.zeros(self.capacity, dtype=np.float32)
            self._sample_states_out = np.empty(
                (self._batch_size_hint, *state_shape), dtype=np.float32
            )
            self._sample_next_states_out = np.empty(
                (self._batch_size_hint, *state_shape), dtype=np.float32
            )
        else:
            self._states = None
            self._next_states = None
            self._actions = None
            self._rewards = None
            self._dones = None
            self._sample_states_out = None
            self._sample_next_states_out = None

    def add(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Insert a transition with initial max priority.

        Args:
            state (Any): Current state observation.
            action (int): Action index taken from `state`.
            reward (float): Immediate reward after the action.
            next_state (Any): Next state observation.
            done (bool): Whether the transition ends the episode.

        Returns:
            None
        """

        max_priority = self.sum_tree.max_priority()
        initial_priority = max_priority if max_priority > 0.0 else 1.0
        leaf_index = self.sum_tree.add(initial_priority, self.sum_tree.write_index)
        data_index = leaf_index - self.sum_tree.leaf_start

        if self._states is not None:
            self._states[data_index] = self._prepare_state_for_storage(state)
            self._next_states[data_index] = self._prepare_state_for_storage(next_state)
            self._actions[data_index] = int(action)
            self._rewards[data_index] = float(reward)
            self._dones[data_index] = float(done)
        else:
            raise ValueError("PrioritizedReplayBuffer requires state_shape-backed storage.")

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[list[Any], np.ndarray, np.ndarray, list[Any], np.ndarray, np.ndarray, np.ndarray]:
        """Sample a mini-batch proportionally to transition priorities.

        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): Importance-sampling correction exponent.

        Returns:
            tuple[list[Any], np.ndarray, np.ndarray, list[Any], np.ndarray, np.ndarray, np.ndarray]:
                `(states, actions, rewards, next_states, dones, indices, weights)`.

        Raises:
            ValueError: If the buffer has fewer than `batch_size` transitions.
        """

        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample batch size {batch_size} from buffer of size {len(self)}."
            )

        total_priority = self.sum_tree.total_priority
        if total_priority <= 0.0:
            raise ValueError("Cannot sample from a PER buffer with zero total priority.")

        segment = total_priority / batch_size
        starts = np.arange(batch_size, dtype=np.float64) * segment
        ends = np.nextafter(starts + segment, starts)
        sample_values = np.random.uniform(starts, ends)

        nodes = np.zeros(batch_size, dtype=np.int64)
        tree = self.sum_tree.tree

        for _ in range(self.sum_tree.depth):
            left = 2 * nodes + 1
            go_right = sample_values > tree[left]
            sample_values -= tree[left] * go_right
            nodes = np.where(go_right, left + 1, left)

        leaf_indices = nodes
        sampled_indices = leaf_indices - self.sum_tree.leaf_start
        priorities = tree[leaf_indices].astype(np.float32)

        valid_mask = (
            (sampled_indices >= 0)
            & (sampled_indices < self.sum_tree.size)
            & (priorities > 0.0)
        )

        if not np.all(valid_mask):
            invalid_positions = np.where(~valid_mask)[0]
            for position in invalid_positions:
                sample_value = float(min(sample_values[position], np.nextafter(total_priority, 0.0)))
                leaf_index, priority, data_index = self.sum_tree.sample(sample_value)
                leaf_indices[position] = leaf_index
                sampled_indices[position] = data_index
                priorities[position] = priority

        sampled_indices = np.clip(sampled_indices, 0, self.capacity - 1)
        probabilities = np.clip(priorities / total_priority, a_min=self.epsilon, a_max=None)
        weights = np.power(len(self) * probabilities, -float(beta)).astype(np.float32)
        weights /= weights.max()

        if self._states is None:
            raise ValueError("PrioritizedReplayBuffer requires state_shape-backed storage.")

        if (
            self._sample_states_out is None
            or self._sample_next_states_out is None
            or self._sample_states_out.shape[0] != batch_size
        ):
            self._sample_states_out = np.empty(
                (batch_size, *self._state_shape), dtype=np.float32
            )
            self._sample_next_states_out = np.empty(
                (batch_size, *self._state_shape), dtype=np.float32
            )

        np.multiply(
            self._states[sampled_indices],
            np.float32(1.0 / 255.0),
            out=self._sample_states_out,
        )
        np.multiply(
            self._next_states[sampled_indices],
            np.float32(1.0 / 255.0),
            out=self._sample_next_states_out,
        )
        states = self._sample_states_out
        next_states = self._sample_next_states_out
        actions = self._actions[sampled_indices]
        rewards = self._rewards[sampled_indices]
        dones = self._dones[sampled_indices]

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            leaf_indices.astype(np.int64),
            weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for a sampled mini-batch.

        Args:
            indices (np.ndarray): Sum Tree leaf indices for sampled transitions.
            td_errors (np.ndarray): TD errors corresponding to the sampled transitions.

        Returns:
            None
        """

        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for leaf_index, priority in zip(indices, priorities):
            self.sum_tree.update(int(leaf_index), float(priority))

    def _compute_priority(self, td_error: float) -> float:
        """Convert a TD error into a proportional-priority value.

        Args:
            td_error (float): Temporal-difference error.

        Returns:
            float: Priority value stored in the Sum Tree.
        """

        return float((abs(td_error) + self.epsilon) ** self.alpha)

    def _prepare_state_for_storage(self, state: Any) -> np.ndarray:
        """Convert one state observation to uint8 storage format.

        Args:
            state (Any): Raw state observation.

        Returns:
            np.ndarray: Uint8 state array in channel-first layout.
        """

        state_array = np.asarray(state)

        if state_array.dtype == np.uint8:
            if state_array.ndim == 4 and state_array.shape[-1] == 1:
                state_array = np.squeeze(state_array, axis=-1)
            if (
                state_array.ndim == 3
                and state_array.shape[-1] in (1, 4)
                and state_array.shape[0] not in (1, 4)
            ):
                state_array = np.transpose(state_array, (2, 0, 1))
            return (
                state_array
                if state_array.flags["C_CONTIGUOUS"]
                else np.ascontiguousarray(state_array, dtype=np.uint8)
            )

        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 4 and state_array.shape[-1] == 1:
            state_array = np.squeeze(state_array, axis=-1)
        if (
            state_array.ndim == 3
            and state_array.shape[-1] in (1, 4)
            and state_array.shape[0] not in (1, 4)
        ):
            state_array = np.transpose(state_array, (2, 0, 1))
        return (state_array * 255.0).astype(np.uint8)

    def __len__(self) -> int:
        """Return the number of stored transitions.

        Args:
            None

        Returns:
            int: Number of buffered transitions.
        """

        return self.sum_tree.size
