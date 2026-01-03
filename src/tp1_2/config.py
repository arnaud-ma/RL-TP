from __future__ import annotations

import pkgutil
import tomllib
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
)

from tp1_2.feature_extractor import FeatureExtractor

ZeroToOne = Annotated[float, Field(ge=0.0, le=1.0)]


def import_from_string(dotted_path: str) -> object:
    return pkgutil.resolve_name(dotted_path)


type PythonObject[T] = Annotated[T, BeforeValidator(import_from_string)]


class ConfigEnv(BaseModel):
    """The necessary configuration to create the gym environment."""

    env: str
    seed: PositiveInt
    freq_verbose: PositiveInt


class ConfigModel(BaseModel):
    env: str

    seed: PositiveInt
    device: Literal["cpu"] | Annotated[int, Field(ge=-1)]

    deterministic_config: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def device_to_cpu_if_negative(cls, v):
        if isinstance(v, int) and v < 0:
            return "cpu"
        return v

    max_length_train: PositiveInt
    max_length_test: PositiveInt

    epsilon_start: ZeroToOne
    epsilon_min: ZeroToOne
    epsilon_decay: ZeroToOne

    gamma: ZeroToOne
    learning_rate: PositiveFloat
    batch_size: PositiveInt
    optimizer: PythonObject[type[torch.optim.Optimizer]]
    value_loss: PythonObject[type[torch.nn.Module]]
    freq_optim: PositiveInt

    feature_extractor: PythonObject[type[FeatureExtractor]]
    hidden_layers: list[int]
    hidden_layers_activation: PythonObject[type[torch.nn.Module]]
    final_layer_activation: PythonObject[type[torch.nn.Module]]
    dropout: ZeroToOne

    replay_memory_capacity: PositiveInt
    replay_memory_min_size: PositiveInt
    prioritized_replay: bool

    nb_episodes: PositiveInt
    freq_test: PositiveInt
    nb_tests: PositiveInt
    freq_save: PositiveInt
    freq_verbose: PositiveInt

    @property
    def config_env(self) -> ConfigEnv:
        return ConfigEnv(
            env=self.env,
            seed=self.seed,
            freq_verbose=self.freq_verbose,
        )


def load_config(config_path: Path) -> ConfigModel:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return ConfigModel(**data)
