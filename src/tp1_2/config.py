from __future__ import annotations

import pkgutil
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from tp1_2.feature_extractor import FeatureExtractor

ZeroToOne = Annotated[float, Field(ge=0.0, le=1.0)]


def import_from_string(dotted_path: str) -> object:
    return pkgutil.resolve_name(dotted_path)


def device_validator(v: Any, handler) -> torch.device:
    """Convert string to torch.device, pass through if already a device"""
    if isinstance(v, str):
        return torch.device(v)
    if isinstance(v, torch.device):
        return v
    # Let pydantic handle other cases
    return handler(v)


type PythonObject[T] = Annotated[T, BeforeValidator(import_from_string)]


class ConfigEnv(BaseModel):
    """The necessary configuration to create the gym environment."""

    env: str
    seed: PositiveInt
    freq_verbose_train: NonNegativeInt = 0
    freq_verbose_test: NonNegativeInt = 0
    freq_animate_train: NonNegativeInt = 0
    freq_animate_test: NonNegativeInt = 0
    render_mode: Literal["human", "terminal", "both"] = "both"


class ConfigDQN(BaseModel):
    env: str

    seed: PositiveInt
    device: str | int
    render_mode: Literal["human", "terminal", "both"] = "both"
    deterministic_config: bool = False

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
    train_freq: PositiveInt
    use_target_network: bool
    target_update_freq: NonNegativeInt
    target_soft_update: ZeroToOne
    clip_grad_norm: NonNegativeInt = 0
    stop_when_test_higher_than: float = float("inf")

    feature_extractor: PythonObject[type[FeatureExtractor]]
    hidden_layers: list[int]
    hidden_layers_activation: PythonObject[type[torch.nn.Module]]
    final_layer_activation: PythonObject[type[torch.nn.Module]]
    dropout: ZeroToOne

    replay_memory_capacity: NonNegativeInt
    replay_memory_min_size: PositiveInt
    prioritized_replay: bool

    nb_episodes: PositiveInt
    freq_test: PositiveInt
    nb_tests: PositiveInt
    freq_save: PositiveInt
    freq_verbose_train: NonNegativeInt
    freq_verbose_test: NonNegativeInt

    @property
    def config_env(self) -> ConfigEnv:
        return ConfigEnv(
            env=self.env,
            seed=self.seed,
            freq_verbose_train=self.freq_verbose_train,
            freq_verbose_test=self.freq_verbose_test,
            render_mode=self.render_mode,
        )

    @property
    def device_torch(self) -> torch.device:
        return torch.device(self.device)

    @model_validator(mode="after")
    def validate_target_update(self):
        if (self.target_update_freq == 0) is (self.target_soft_update == 0.0):
            msg = (
                "Choose either target_update_freq > 0 for hard updates "
                "or target_soft_update > 0.0 for soft updates."
            )
            raise ValueError(msg)
        return self


class ConfigPPO(BaseModel):
    env: str

    seed: PositiveInt
    device: str | int
    render_mode: Literal["human", "terminal", "both"] = "both"
    deterministic_config: bool = False

    max_length_train: PositiveInt
    max_length_test: PositiveInt

    num_updates: PositiveInt
    steps_per_update: PositiveInt
    stop_when_test_higher_than: float = float("inf")
    freq_test: PositiveInt
    nb_tests: PositiveInt
    freq_save: PositiveInt
    freq_verbose_train: NonNegativeInt
    freq_verbose_test: NonNegativeInt

    gamma: ZeroToOne  # discount factor
    gae_lambda: ZeroToOne  # GAE lambda parameter
    clip_epsilon: PositiveFloat  # PPO clipping epsilon
    beta_init: PositiveFloat  # initial value for entropy bonus coefficient
    target_kl: PositiveFloat  # target KL divergence for early stopping

    num_epochs: PositiveInt  # number of epochs per update
    batch_size: PositiveInt  # minibatch size for each epoch

    value_loss_coef: PositiveFloat  # coefficient for value loss
    entropy_coef: PositiveFloat  # coefficient for entropy bonus

    policy_lr: PositiveFloat  # learning rate for policy optimizer
    value_lr: PositiveFloat  # learning rate for value optimizer

    value_hidden_layers: list[int]
    value_hidden_layers_activation: PythonObject[type[torch.nn.Module]]
    value_activation: PythonObject[type[torch.nn.Module]]
    value_dropout: ZeroToOne

    policy_hidden_layers: list[int]
    policy_hidden_layers_activation: PythonObject[type[torch.nn.Module]]
    policy_activation: PythonObject[type[torch.nn.Module]]
    policy_dropout: ZeroToOne

    clip_grad_norm: PositiveFloat

    feature_extractor: PythonObject[type[FeatureExtractor]]

    @property
    def config_env(self) -> ConfigEnv:
        return ConfigEnv(
            env=self.env,
            seed=self.seed,
            freq_verbose_train=self.freq_verbose_train,
            freq_verbose_test=self.freq_verbose_test,
            render_mode=self.render_mode,
        )

    @property
    def device_torch(self) -> torch.device:
        return torch.device(self.device)


def load_config[T](config_path: Path, kind: type[T]) -> T:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return kind(**data)
