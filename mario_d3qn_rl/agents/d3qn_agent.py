"""D3QN agent implementation for Task 1 without experience replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from networks.d3qn_network import DuelingDQN


@dataclass
class Transition:
    """Container for a single online transition.

    Attributes:
        state (Any): Current state observation.
        action (int): Action chosen for the current state.
        reward (float): Immediate reward received after the action.
        next_state (Any): Next state observation.
        done (bool): Whether the transition ends the episode.
    """

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class D3QNAgent:
    """Task 1 D3QN agent using online updates and Double DQN targets."""

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
        device: torch.device,
    ) -> None:
        """Initialize networks, optimizer, and exploration schedule.

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
            device (torch.device): Device used for tensor operations.

        Returns:
            None
        """

        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip
        self.device = device
        self.training_steps = 0

        self.policy_network = DuelingDQN(
            input_shape=observation_shape,
            action_size=action_size,
        ).to(self.device)
        self.target_network = DuelingDQN(
            input_shape=observation_shape,
            action_size=action_size,
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(self, state: Any) -> int:
        """Choose an action using epsilon-greedy exploration.

        Args:
            state (Any): Environment observation for the current state.

        Returns:
            int: Selected discrete action index.
        """

        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_size))

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def learn(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """Update the policy network from the current transition only.

        Args:
            state (Any): Current state observation.
            action (int): Action selected in the current state.
            reward (float): Reward received after executing the action.
            next_state (Any): Next state observation.
            done (bool): Whether the transition terminates the episode.

        Returns:
            float: Scalar Huber loss value for this online update.
        """

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        state_tensor = self._state_to_tensor(transition.state)
        next_state_tensor = self._state_to_tensor(transition.next_state)
        action_tensor = torch.tensor([transition.action], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([transition.reward], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([float(transition.done)], dtype=torch.float32, device=self.device)

        return self._learn_from_batch(
            states=state_tensor,
            actions=action_tensor,
            rewards=reward_tensor,
            next_states=next_state_tensor,
            dones=done_tensor,
        )

    def _learn_from_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Apply one gradient update from a mini-batch of transitions.

        Args:
            states (torch.Tensor): Batched state tensor `(batch, channels, height, width)`.
            actions (torch.Tensor): Batched action indices `(batch,)`.
            rewards (torch.Tensor): Batched rewards `(batch,)`.
            next_states (torch.Tensor): Batched next-state tensor `(batch, channels, height, width)`.
            dones (torch.Tensor): Batched terminal flags `(batch,)`, as floats in `{0.0, 1.0}`.

        Returns:
            float: Scalar Huber loss value.
        """

        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_network(next_states).argmax(dim=1, keepdim=True)
            next_q_target = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q_target * (1.0 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()

        self.training_steps += 1
        self._update_epsilon()

        if self.training_steps % self.target_update_freq == 0:
            self.sync_target_network()

        return float(loss.item())

    def sync_target_network(self) -> None:
        """Copy policy network weights into the target network.

        Args:
            None

        Returns:
            None
        """

        self.target_network.load_state_dict(self.policy_network.state_dict())

    def checkpoint_state(self) -> dict[str, Any]:
        """Build a serializable snapshot of the agent state.

        Args:
            None

        Returns:
            dict[str, Any]: Checkpoint payload for later restoration.
        """

        return {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }

    def save_checkpoint(self, output_path: str | Any) -> None:
        """Persist the current agent state to disk.

        Args:
            output_path (str | Any): Destination path accepted by ``torch.save``.

        Returns:
            None
        """

        torch.save(self.checkpoint_state(), output_path, pickle_protocol=4)

    def load_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore the agent state from a checkpoint payload.

        Args:
            checkpoint (dict[str, Any]): Serialized checkpoint payload.

        Returns:
            None
        """

        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint["epsilon"])
        self.training_steps = int(checkpoint["training_steps"])

    def _update_epsilon(self) -> None:
        """Decay epsilon after every environment step.

        Args:
            None

        Returns:
            None
        """

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _state_to_tensor(self, state: Any) -> torch.Tensor:
        """Convert an environment state to a normalized NCHW tensor.

        Args:
            state (Any): Raw state output from the wrapped environment.

        Returns:
            torch.Tensor: Float tensor on `self.device` with shape `(batch, channels, height, width)`.
        """

        state_array = np.asarray(state, dtype=np.float32)

        if state_array.ndim == 3:
            return torch.as_tensor(
                state_array[np.newaxis], dtype=torch.float32, device=self.device
            ) / 255.0

        if state_array.ndim == 4 and state_array.shape[-1] == 1:
            state_array = np.squeeze(state_array, axis=-1)
        if state_array.ndim == 4 and state_array.shape[-1] in (1, 4):
            state_array = np.transpose(state_array, (0, 3, 1, 2))
        if state_array.ndim == 3:
            state_array = np.expand_dims(state_array, axis=0)

        return torch.as_tensor(state_array, dtype=torch.float32, device=self.device) / 255.0

    def _state_batch_to_tensor(self, states: list[Any]) -> torch.Tensor:
        """Convert a list of environment states to one normalized NCHW tensor.

        Args:
            states (list[Any]): Raw state outputs from the wrapped environment.

        Returns:
            torch.Tensor: Float tensor on `self.device` with shape `(batch, channels, height, width)`.
        """

        state_arrays = [np.array(state, copy=False, dtype=np.float32) for state in states]
        batch_array = np.stack(state_arrays, axis=0)

        if batch_array.ndim == 5 and batch_array.shape[-1] == 1:
            batch_array = np.squeeze(batch_array, axis=-1)

        if batch_array.ndim == 4 and batch_array.shape[-1] in (1, 4):
            batch_array = np.transpose(batch_array, (0, 3, 1, 2))

        batch_tensor = torch.as_tensor(batch_array, dtype=torch.float32, device=self.device)
        return batch_tensor / 255.0
