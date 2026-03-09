"""Episode-level training logger for Mario D3QN experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingLogger:
    """Collect and save episode-level reward and loss metrics.

    Attributes:
        episode_rewards (list[float]): Reward total for each episode.
        episode_losses (list[float]): Mean training loss for each episode.
    """

    episode_rewards: list[float] = field(default_factory=list)
    episode_losses: list[float] = field(default_factory=list)

    def log_episode(self, episode_reward: float, average_loss: float) -> None:
        """Append one episode summary to in-memory metric history.

        Args:
            episode_reward (float): Total reward accumulated in the episode.
            average_loss (float): Mean loss observed in the episode.

        Returns:
            None
        """

        self.episode_rewards.append(float(episode_reward))
        self.episode_losses.append(float(average_loss))

    def save(self, output_dir: Path) -> None:
        """Persist tracked metrics to CSV files.

        Args:
            output_dir (Path): Directory where metric CSVs are written.

        Returns:
            None
        """

        self._write_metric(output_dir / "episode_rewards.csv", "episode_reward", self.episode_rewards)
        self._write_metric(output_dir / "episode_losses.csv", "episode_loss", self.episode_losses)

    def _write_metric(self, file_path: Path, column_name: str, values: list[float]) -> None:
        """Write a single metric sequence to a CSV file.

        Args:
            file_path (Path): Output CSV path.
            column_name (str): Metric column name.
            values (list[float]): Ordered metric values by episode.

        Returns:
            None
        """

        with file_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["episode", column_name])
            for episode_index, value in enumerate(values, start=1):
                writer.writerow([episode_index, value])
