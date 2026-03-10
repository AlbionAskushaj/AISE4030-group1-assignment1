"""Unified training entry point for Mario D3QN assignment tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from agents.d3qn_agent import D3QNAgent
from agents.d3qn_er_agent import D3QNERAgent
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


def resolve_device(device_name: str) -> torch.device:
    """Resolve a configured device name with safe fallback behavior.

    Args:
        device_name (str): Requested device name from config.

    Returns:
        torch.device: Selected PyTorch device for training.
    """

    normalized_name = device_name.strip().lower()

    if normalized_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(normalized_name)


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
    """Run training for the configured D3QN agent variant.

    Args:
        None

    Returns:
        None
    """

    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "config.yaml")

    env, observation_shape, action_size = create_env(config["env_name"])
    training_config = config["training"]
    replay_config = config.get("replay", {})
    device = resolve_device(config["device"])
    agent_type = str(config["agent_type"]).lower()

    if agent_type == "d3qn":
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
        results_subdir = "task1_d3qn"
        progress_desc = "Training D3QN"
    elif agent_type == "d3qn_er":
        agent = D3QNERAgent(
            observation_shape=observation_shape,
            action_size=action_size,
            gamma=training_config["gamma"],
            learning_rate=training_config["learning_rate"],
            epsilon_start=training_config["epsilon_start"],
            epsilon_min=training_config["epsilon_min"],
            epsilon_decay=training_config["epsilon_decay"],
            target_update_freq=training_config["target_update_freq"],
            gradient_clip=training_config["gradient_clip"],
            replay_buffer_capacity=replay_config["capacity"],
            learning_starts=replay_config["learning_starts"],
            batch_size=training_config["batch_size"],
            device=device,
        )
        results_subdir = "task2_d3qn_er"
        progress_desc = "Training D3QN+ER"
    else:
        raise ValueError(
            "Unsupported agent_type in config.yaml. Expected one of: 'd3qn', 'd3qn_er'."
        )

    logger = TrainingLogger()
    results_dir = project_root / "results" / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(
        range(1, training_config["episodes"] + 1),
        desc=progress_desc,
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

            if loss is not None:
                episode_losses.append(loss)

        average_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        logger.log_episode(episode_reward=episode_reward, average_loss=average_loss)

        progress_metrics: dict[str, str | int] = {
            "reward": f"{episode_reward:.2f}",
            "loss": f"{average_loss:.4f}",
            "epsilon": f"{agent.epsilon:.4f}",
        }

        if isinstance(agent, D3QNERAgent):
            progress_metrics["buffer"] = len(agent.replay_buffer)

        progress_bar.set_postfix(progress_metrics)

    logger.save(results_dir)
    plot_training_curves(logger.episode_rewards, logger.episode_losses, results_dir)
    env.close()


if __name__ == "__main__":
    train()
