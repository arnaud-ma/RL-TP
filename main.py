from pathlib import Path

import torch

from tp1_2 import agents, setup
from tp1_2.config import ConfigPPO


def main():

    # get nb of threads
    torch.set_num_threads(torch.get_num_threads() * 2 - 1)
    setup_env = setup.init_env(
        config_file="./configs/dqn-lunar.toml",
        name="dqn_lunar",
        launch_tensorboard=True,
        kind=setup.ConfigDQN,
    )
    # state, info = gym_env.reset()
    # print(state, state.shape, type(state))
    agent = agents.dqn.DoubleDQNAgent(setup_env)
    agent.run()


def main2():
    p = Path(
        "./outputs/LunarLander-v3/double_dqn_lunar_04-01-2026_23-26-28/",
    ).resolve()
    agent = agents.dqn.Agent.from_dir(p)
    while True:
        agent.evaluate_episode(render=True)


def main_ppo():
    setup_env = setup.init_env(
        config_file="./configs/ppo-lunar.toml",
        name="ppo_lunar",
        launch_tensorboard=True,
        kind=ConfigPPO,
    )
    agent = agents.ppo2.PPOAgent(setup_env)
    agent.run()


if __name__ == "__main__":
    main_ppo()
