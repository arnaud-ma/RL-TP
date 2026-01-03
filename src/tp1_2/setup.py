import datetime
import random
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rich
import torch
from torch.utils.tensorboard import SummaryWriter

from tp1_2 import logging
from tp1_2.config import ConfigModel, load_config
from tp1_2.gym_env import DualEnvWrapper
from tp1_2.logging import TensorboardLogger


def init_random_seed(config: ConfigModel) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)  # noqa: NPY002
    torch.manual_seed(config.seed)
    if config.device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def get_outdir(run_name: str, env_name: str):
    time_str = datetime.datetime.now(tz=datetime.UTC).strftime("%d-%m-%Y_%H-%M-%S")
    outdir = Path("outputs") / env_name / f"{run_name}_{time_str}"
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


class SetupEnv(NamedTuple):
    """A named tuple containing the setup components for a reinforcement learning
    environment.

    Attributes:
        env (DualEnvWrapper): The wrapped environment instance used for training
            and evaluation.
        config (ConfigModel): The configuration model containing hyperparameters and
            settings.
        outdir (Path): The output directory path where results and logs will be saved.
        logger (logging.TensorboardLogger): The TensorBoard logger instance for
            tracking metrics.
    """

    env: DualEnvWrapper
    config: ConfigModel
    outdir: Path
    logger: logging.TensorboardLogger


def init_env(
    config_file: str | Path,
    name: str,
    *,
    launch_tensorboard: bool = True,
) -> SetupEnv:
    """
    Initialize the environment and related components for reinforcement
    learning.

    It:
    - loads configuration from the specified file
    - initializes random seed based on config
    - optionally sets deterministic CUDA behavior if specified in config
      ('deterministic_config' flag in the config file),
    - creates the output directory and copies the config file to it, and
      initializes the TensorBoard writer and logger for this output directory.

    Args:
        config_file (str | Path): Path to the configuration file.
            Can be a string or Path object. The path will be expanded and
            resolved to an absolute path.
        name (str): Name of the run, used for creating output
            directories and logging.

    Returns:
        - SetupEnv: A named tuple containing the initialized environment,
            configuration, output directory, and logger. See SetupEnv for details.
    """
    config_file = Path(config_file).expanduser().resolve()

    config = load_config(config_file)
    print("Loaded config:")
    rich.print(config)

    init_random_seed(config)
    if config.deterministic_config:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    outdir = get_outdir(run_name=name, env_name=config.env)
    shutil.copy2(config_file, outdir / config_file.name)

    writer = SummaryWriter(log_dir=outdir / "tensorboard")
    logger = TensorboardLogger(writer)
    if launch_tensorboard:
        logging.load_tensorboard(outdir / "tensorboard")

    env = DualEnvWrapper(config.config_env)
    env.reset()
    return SetupEnv(env, config, outdir, logger)
