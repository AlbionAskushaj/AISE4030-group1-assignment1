"""D3QN agent implementation for Task 2 with uniform experience replay."""

from __future__ import annotations

from typing import Any

import torch

from agents.d3qn_agent import D3QNAgent
from buffers.replay_buffer import ReplayBuffer


class D3QNERAgent(D3QNAgent):
    """Task 2 D3QN agent with a uniform replay buffer and mini-batch updates."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        gamma: float,
        learning_rate: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        target_update_freq: int,
        gradient_clip: float,
        replay_buffer_capacity: int,
        learning_starts: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Initialize replay-enabled D3QN agent.

        Args:
            observation_shape (tuple[int, ...]): Model input shape as `(channels, height, width)`.
            action_size (int): Number of discrete actions.
            gamma (float): Discount factor used in TD target computation.
            learning_rate (float): Adam optimizer learning rate.
            epsilon_start (float): Initial epsilon for epsilon-greedy exploration.
            epsilon_min (float): Minimum epsilon value.
            epsilon_decay (float): Multiplicative epsilon decay factor per step.
            target_update_freq (int): Number of gradient steps between target-network syncs.
            gradient_clip (float): Maximum gradient norm for clipping.
            replay_buffer_capacity (int): Maximum transitions stored in the replay buffer.
            learning_starts (int): Minimum buffer size required before learning begins.
            batch_size (int): Number of sampled transitions per gradient update.
            device (torch.device): Device used for tensor operations.

        Returns:
            None
        """

        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            gradient_clip=gradient_clip,
            device=device,
        )

        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)

    def learn(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float | None:
        """Store transition and perform one replay-based update when eligible.

        Args:
            state (Any): Current state observation.
            action (int): Action selected in the current state.
            reward (float): Reward received after executing the action.
            next_state (Any): Next state observation.
            done (bool): Whether the transition terminates the episode.

        Returns:
            float | None: Scalar loss when an update is performed, otherwise `None`.
        """

        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        if len(self.replay_buffer) < max(self.learning_starts, self.batch_size):
            self._update_epsilon()
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = self._stack_state_batch(states)
        next_states_tensor = self._stack_state_batch(next_states)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        return self._learn_from_batch(
            states=states_tensor,
            actions=actions_tensor,
            rewards=rewards_tensor,
            next_states=next_states_tensor,
            dones=dones_tensor,
        )

    def _stack_state_batch(self, states: list[Any]) -> torch.Tensor:
        """Convert a list of raw observations into one batched tensor.

        Args:
            states (list[Any]): Sequence of state observations.

        Returns:
            torch.Tensor: Batched state tensor with shape `(batch, channels, height, width)`.
        """

        state_tensors = [self._state_to_tensor(state) for state in states]
        return torch.cat(state_tensors, dim=0)
