"""Environment factory and wrappers for Super Mario Bros Task 1."""

from __future__ import annotations

from typing import Any

import gym
import gym_super_mario_bros
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace


RIGHT_ONLY_ACTIONS = [["right"], ["right", "A"]]


class SkipFrame(gym.Wrapper):
    """Repeat each action for a fixed number of frames and sum rewards."""

    def __init__(self, env: gym.Env, skip: int) -> None:
        """Store the wrapped environment and frame-repeat count.

        Args:
            env (gym.Env): Environment instance to wrap.
            skip (int): Number of repeated frames per chosen action.

        Returns:
            None
        """

        super().__init__(env)
        self.skip = skip

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Repeat one action across multiple frames.

        Args:
            action (int): Discrete action index.

        Returns:
            tuple[Any, float, bool, bool, dict[str, Any]]:
                Observation, accumulated reward, terminated flag, truncated flag, and info dict.
        """

        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        observation: Any = None

        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


class MarioAPICompatibility(gym.Wrapper):
    """Normalize reset/step outputs to Gym 0.26 API conventions."""

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Return observation and info in tuple format.

        Args:
            **kwargs (Any): Keyword arguments passed to the wrapped env reset.

        Returns:
            tuple[Any, dict[str, Any]]: Observation and info dictionary.
        """

        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result
        return reset_result, {}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Return step outputs in `(obs, reward, terminated, truncated, info)` format.

        Args:
            action (int): Discrete action index.

        Returns:
            tuple[Any, float, bool, bool, dict[str, Any]]:
                Observation, reward, terminated, truncated, and info.
        """

        step_result = self.env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            return observation, float(reward), bool(terminated), bool(truncated), info

        observation, reward, done, info = step_result
        return observation, float(reward), bool(done), False, info


def create_env(
    env_name: str,
    render_mode: str | None = None,
    seed: int | None = None,
) -> tuple[gym.Env, tuple[int, ...], int]:
    """Create Mario env with assignment-specific wrappers and action space.

    Args:
        env_name (str): Gym environment id (e.g., `SuperMarioBros-1-1-v3`).
        render_mode (str | None): Optional render mode passed to env factory.
        seed (int | None): Optional random seed applied to base env.

    Returns:
        tuple[gym.Env, tuple[int, ...], int]:
            Wrapped env instance, observation shape, and discrete action size.
    """

    try:
        env = gym_super_mario_bros.make(
            env_name,
            render_mode=render_mode,
            apply_api_compatibility=True,
        )
    except TypeError:
        try:
            env = gym_super_mario_bros.make(
                env_name,
                apply_api_compatibility=True,
            )
        except TypeError:
            env = gym_super_mario_bros.make(env_name)

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)

    env = MarioAPICompatibility(env)
    env = JoypadSpace(env, RIGHT_ONLY_ACTIONS)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    observation_shape = _extract_observation_shape(env)
    action_size = int(env.action_space.n)
    return env, observation_shape, action_size


def _extract_observation_shape(env: gym.Env) -> tuple[int, ...]:
    """Convert wrapped observation shape into channel-first format.

    Args:
        env (gym.Env): Wrapped Mario environment.

    Returns:
        tuple[int, ...]: Observation shape expected by the policy network.

    Raises:
        ValueError: If observation dimensions are missing or malformed.
    """

    shape = env.observation_space.shape
    if shape is None:
        raise ValueError("Observation space shape could not be determined.")

    if len(shape) != 3:
        raise ValueError(f"Expected a 3D observation space after FrameStack, got {shape}.")

    frame_count = int(shape[0])
    height = int(shape[1])
    width = int(shape[2])

    if frame_count != 4:
        sample_observation, _ = env.reset()
        sample_array = np.array(sample_observation, copy=False)
        if sample_array.ndim != 3:
            raise ValueError(f"Expected stacked observations with 3 dims, got {sample_array.shape}.")
        return tuple(int(dim) for dim in sample_array.shape)

    return frame_count, height, width
