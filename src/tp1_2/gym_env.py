from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from tp1_2.config import ConfigEnv

# generic: Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]


class DualEnvWrapper[T](gym.Wrapper[np.ndarray, T, np.ndarray, T]):
    """Peronnalized Gym Wrapper to handle dual environments (silent and human).

    No other solution found than wrapping the gym environment itself to create
    two instances (one silent, one human-rendered)
    """

    def __init__(self, config: ConfigEnv):
        self.silent_env: gym.Env[np.ndarray, T] = gym.make(config.env)
        self.human_env: gym.Env[np.ndarray, T] = gym.make(
            config.env,
            render_mode="human",
        )

        self.config = config
        if self.silent_env is not None:
            super().__init__(self.silent_env)
        elif self.human_env is not None:
            super().__init__(self.human_env)
        else:
            msg = "At least one environment must be created."
            raise ValueError(msg)

        self._episode_train = 0
        self._episode_test = 0
        self._freq_human_train = config.freq_animate_train
        self._freq_human_test = config.freq_animate_test
        self._seed = config.seed
        self._current_env = self.silent_env

        self.silent_env.action_space.seed(self._seed)
        self.human_env.action_space.seed(self._seed)
        self.silent_env.observation_space.seed(self._seed)
        self.human_env.observation_space.seed(self._seed)

    def choose_env(self, kind: Literal["train", "test"]) -> gym.Env[np.ndarray, T]:
        """Get the current active environment (silent or human).

        Returns:
            gym.Env: The currently active environment instance.
        """
        episode = self._episode_train if kind == "train" else self._episode_test
        freq_human = (
            self._freq_human_train if kind == "train" else self._freq_human_test
        )
        use_human = (episode >= 0) and (freq_human > 0) and (episode % freq_human == 0)
        self._current_env = self.human_env if use_human else self.silent_env
        return self.human_env if use_human else self.silent_env

    def reset_human(self, *args, **kwargs):
        """Reset the human-rendered environment.

        Args:
            *args: positional arguments to pass to env.reset()
            **kwargs: keyword arguments to pass to env.reset()

        Returns:
            observation, info: the initial observation (numPy array) and info dict
            (see Gym documentation for details).
        """
        self._current_env = self.human_env
        return self.human_env.reset(*args, **kwargs)

    def reset(self, *args, **kwargs):
        """Reset the environment.

        Args:
            *args: positional arguments to pass to env.reset()
            **kwargs: keyword arguments to pass to env.reset()

        Returns:
            observation, info: the initial observation (numPy array) and info dict
            (see Gym documentation for details).
        """
        return self._current_env.reset(*args, **kwargs)

    def reset_train(self, *args, **kwargs):
        """Reset the human-rendered environment.

        Args:
            *args: positional arguments to pass to env.reset()
            **kwargs: keyword arguments to pass to env.reset()

        Returns:
            observation, info: the initial observation (numPy array) and info dict
            (see Gym documentation for details).
        """
        self._seed += 1
        self._episode_train += 1
        return self.choose_env("train").reset(*args, seed=self._seed, **kwargs)

    def reset_test(self, *args, **kwargs):
        """Reset the silent environment.

        Args:
            *args: positional arguments to pass to env.reset()
            **kwargs: keyword arguments to pass to env.reset()

        Returns:
            observation, info: the initial observation (numPy array) and info dict
            (see Gym documentation for details).
        """
        self._seed += 1
        self._episode_test += 1
        return self.choose_env("test").reset(*args, seed=self._seed, **kwargs)

    def step(self, action: T):
        return self._current_env.step(action)

    # def setPlan(self, map_, rewards):
    #     self.silent_env.unwrapped.setPlan(map_, rewards)  # type: ignore  # noqa: PGH003
    #     self.human_env.unwrapped.setPlan(map_, rewards)  # type: ignore  # noqa: PGH003
