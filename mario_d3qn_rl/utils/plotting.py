"""Plotting helpers for Mario D3QN training metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(
    rewards: list[float],
    losses: list[float],
    output_dir: Path,
) -> None:
    """Generate and save reward and loss curves.

    Args:
        rewards (list[float]): Episode reward history.
        losses (list[float]): Episode mean loss history.
        output_dir (Path): Directory where plot images are written.

    Returns:
        None
    """

    _plot_metric(
        values=rewards,
        title="Episode Rewards",
        ylabel="Reward",
        output_path=output_dir / "reward_plot.png",
    )
    _plot_metric(
        values=losses,
        title="Episode Average Loss",
        ylabel="Loss",
        output_path=output_dir / "loss_plot.png",
    )


def _plot_metric(values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    """Render one metric line chart and save to disk.

    Args:
        values (list[float]): Ordered metric values to plot.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        output_path (Path): File path for saved image.

    Returns:
        None
    """

    plt.figure(figsize=(10, 5))
    plt.plot(values, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
