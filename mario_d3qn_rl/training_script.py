"""Training entry point for Task 1 D3QN on SuperMarioBros-1-1-v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from agents.d3qn_agent import D3QNAgent
from environment.environment import create_env
from utils.logger import TrainingLogger
from utils.plotting import plot_training_curves


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment configuration from a YAML file.

    Args:
        config_path (Path): Path to the YAML config file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.
    """

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def make_mario_env(
    env_name: str,
    render_mode: str | None = None,
    seed: int | None = None,
):
    """Create an env instance for assignment installation verification.

    Args:
        env_name (str): Gym environment id.
        render_mode (str | None): Optional render mode.
        seed (int | None): Optional random seed.

    Returns:
        gym.Env: Wrapped Mario environment instance.
    """

    env, _, _ = create_env(env_name=env_name, render_mode=render_mode, seed=seed)
    return env


def train() -> None:
    """Run Task 1 training with online D3QN updates.

    Args:
        None

    Returns:
        None
    """

    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "config.yaml")

    env, observation_shape, action_size = create_env(config["env_name"])
    training_config = config["training"]
    device = torch.device(config["device"])

    agent = D3QNAgent(
        observation_shape=observation_shape,
        action_size=action_size,
        gamma=training_config["gamma"],
        learning_rate=training_config["learning_rate"],
        epsilon_start=training_config["epsilon_start"],
        epsilon_min=training_config["epsilon_min"],
        epsilon_decay=training_config["epsilon_decay"],
        target_update_freq=training_config["target_update_freq"],
        gradient_clip=training_config["gradient_clip"],
        device=device,
    )

    logger = TrainingLogger()
    results_dir = project_root / "results" / "task1_d3qn"
    results_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(
        range(1, training_config["episodes"] + 1),
        desc="Training D3QN",
        unit="episode",
    )

    for _episode in progress_bar:
        state, _ = env.reset()
        episode_reward = 0.0
        episode_losses: list[float] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            loss = agent.learn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated or truncated,
            )

            state = next_state
            episode_reward += float(reward)
            episode_losses.append(loss)

        average_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        logger.log_episode(episode_reward=episode_reward, average_loss=average_loss)

        progress_bar.set_postfix(
            reward=f"{episode_reward:.2f}",
            loss=f"{average_loss:.4f}",
            epsilon=f"{agent.epsilon:.4f}",
        )

    logger.save(results_dir)
    plot_training_curves(logger.episode_rewards, logger.episode_losses, results_dir)
    env.close()


if __name__ == "__main__":
    train()
