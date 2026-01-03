from typing import Protocol, override, runtime_checkable

import numpy as np
import numpy.typing as npt

from tp1_2.gym_env import DualEnvWrapper


@runtime_checkable
class FeatureExtractor(Protocol):
    def __init__(self, env: DualEnvWrapper): ...

    def out_size(self) -> int:
        """Number of features extracted."""
        ...

    def get_features(self, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Extract features from observation.

        Modify the shape if necessary to return a 2D array of shape (1, nb_features).
        """
        ...


class NothingToDo(FeatureExtractor):
    def __init__(self, env: DualEnvWrapper):
        observation, _ = env.reset()
        observation = observation.ravel()
        self._out_size = len(observation)

    def out_size(self) -> int:
        return self._out_size

    @override
    def get_features(self, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return obs.reshape(1, -1)
