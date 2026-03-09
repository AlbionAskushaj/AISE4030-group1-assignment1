"""Dueling DQN network used by the Task 1 Mario agent."""

from __future__ import annotations

import torch
from torch import nn


class DuelingDQN(nn.Module):
    """Convolutional dueling architecture for value-based Mario control."""

    def __init__(self, input_shape: tuple[int, ...], action_size: int) -> None:
        """Build convolutional, value, and advantage streams.

        Args:
            input_shape (tuple[int, ...]): Input shape as (channels, height, width).
            action_size (int): Number of discrete actions.

        Returns:
            None
        """

        super().__init__()
        channels, _, _ = input_shape

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        feature_size = self._get_feature_size(input_shape)

        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dueling Q-values for each action.

        Args:
            x (torch.Tensor): Input state tensor with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Predicted Q-values with shape (batch, action_size).
        """

        features = self.feature_extractor(x)
        flattened = torch.flatten(features, start_dim=1)

        value = self.value_stream(flattened)
        advantage = self.advantage_stream(flattened)

        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def _get_feature_size(self, input_shape: tuple[int, ...]) -> int:
        """Infer flattened feature size after convolutional layers.

        Args:
            input_shape (tuple[int, ...]): Input shape as (channels, height, width).

        Returns:
            int: Number of flattened features output by the conv backbone.
        """

        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            features = self.feature_extractor(sample)
        return int(features.numel())
