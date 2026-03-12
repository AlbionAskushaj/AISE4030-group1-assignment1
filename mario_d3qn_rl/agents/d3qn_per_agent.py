"""D3QN agent implementation for Task 3 with prioritized experience replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agents.d3qn_agent import D3QNAgent
from buffers.per_buffer import PrioritizedReplayBuffer


class D3QNPERAgent(D3QNAgent):
    """Task 3 D3QN agent using prioritized experience replay and IS weighting."""

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
        per_alpha: float,
        per_beta_start: float,
        per_beta_end: float,
        per_epsilon: float,
        training_total_steps: int,
        device: torch.device,
    ) -> None:
        """Initialize replay-enabled D3QN agent with PER.

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
            per_alpha (float): Prioritization exponent.
            per_beta_start (float): Initial importance-sampling exponent.
            per_beta_end (float): Final importance-sampling exponent.
            per_epsilon (float): Small constant added to TD errors before prioritization.
            training_total_steps (int): Total scheduled environment steps for beta annealing.
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

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_buffer_capacity,
            alpha=per_alpha,
            epsilon=per_epsilon,
            state_shape=observation_shape,
            batch_size=batch_size,
        )
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.beta_start = float(per_beta_start)
        self.beta_end = float(per_beta_end)
        self.beta = float(per_beta_start)
        self.beta_anneal_steps = max(int(training_total_steps), 1)
        self.beta_step = (self.beta_end - self.beta_start) / self.beta_anneal_steps

    def learn(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float | None:
        """Store transition and perform one PER-based update when eligible.

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

        self._anneal_beta()

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.batch_size, beta=self.beta)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        current_q = self.policy_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_network(next_states_tensor).argmax(dim=1, keepdim=True)
            next_q_target = self.target_network(next_states_tensor).gather(1, next_actions).squeeze(1)
            target_q = rewards_tensor + self.gamma * next_q_target * (1.0 - dones_tensor)

        td_errors = target_q - current_q
        per_sample_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = torch.mean(weights_tensor * per_sample_loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()

        self.replay_buffer.update_priorities(
            indices=indices,
            td_errors=td_errors.detach().abs().cpu().numpy(),
        )

        self.training_steps += 1
        self._update_epsilon()

        if self.training_steps % self.target_update_freq == 0:
            self.sync_target_network()

        return float(loss.item())
    def _anneal_beta(self) -> None:
        """Linearly anneal beta toward its final value.

        Args:
            None

        Returns:
            None
        """

        self.beta = min(self.beta_end, self.beta + self.beta_step)

    def checkpoint_state(self) -> dict[str, Any]:
        """Build a serializable snapshot without replay buffer contents.

        Args:
            None

        Returns:
            dict[str, Any]: Checkpoint payload for later restoration.
        """

        checkpoint = super().checkpoint_state()
        checkpoint["beta"] = self.beta
        return checkpoint

    def load_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore agent and PER buffer state from a checkpoint payload.

        Args:
            checkpoint (dict[str, Any]): Serialized checkpoint payload.

        Returns:
            None
        """

        super().load_checkpoint_state(checkpoint)
        self.beta = float(checkpoint.get("beta", self.beta))

        # Replay-buffer contents are intentionally not restored from checkpoints
        # to keep checkpoint files small and stable.
        return
