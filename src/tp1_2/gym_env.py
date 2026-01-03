from __future__ import annotations

from typing import TYPE_CHECKING

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
        super().__init__(self.silent_env)
        self._episode = 0
        self._freq_verbose = config.freq_verbose
        self._seed = config.seed

    def reset(self, *args, **kwargs):
        """Reset the environment, choosing between silent and human-rendered.

        Args:
            *args: positional arguments to pass to env.reset()
            **kwargs: keyword arguments to pass to env.reset()

        Returns:
            observation, info: the initial observation (numPy array) and info dict
            (see Gym documentation for details).
        """
        use_human = self._episode >= 0 and self._episode % self._freq_verbose == 0
        self.env = self.human_env if use_human else self.silent_env
        self._seed += 1
        self._episode += 1
        return self.env.reset(*args, seed=self._seed, **kwargs)

    def set_plan(self, map_, rewards):
        self.silent_env.unwrapped.setPlan(map_, rewards)  # type: ignore  # noqa: PGH003
        self.human_env.unwrapped.setPlan(map_, rewards)  # type: ignore  # noqa: PGH003
