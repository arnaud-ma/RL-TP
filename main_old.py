import itertools
import random
import statistics
from itertools import count

import torch

from tp1_2 import feature_extractor
from tp1_2.memory import ReplayMemory, Transition, TransposedTransition
from tp1_2.network import NeuralNetwork
from tp1_2.setup import init_env

config_path = "./configs/my_config_dqn_cartpole.toml"
env, config, outdir, logger = init_env(config_path, name="my_run")


BATCH_SIZE = config.batch_size
GAMMA = config.gamma
EPS_START = config.epsilon_start
EPS_END = config.epsilon_min
EPS_DECAY = config.epsilon_decay
LR = config.learning_rate

epsilon = EPS_START

memory = ReplayMemory(
    capacity=config.replay_memory_capacity,
    prioritized=config.prioritized_replay,
)

n_actions: int = env.action_space.n
state, info = env.reset()
n_observations = len(state)
print(f"Number of actions: {n_actions}")
print(f"Number of observations: {n_observations}")
print(f"Output directory: {outdir}")

policy_net = NeuralNetwork(
    input_size=n_observations,
    output_size=n_actions,
    layers=config.hidden_layers,
    activation=config.hidden_layers_activation,
    final_activation=config.final_layer_activation,
    dropout=config.dropout,
)

target_net = NeuralNetwork(
    input_size=n_observations,
    output_size=n_actions,
    layers=config.hidden_layers,
    activation=config.hidden_layers_activation,
    final_activation=config.final_layer_activation,
    dropout=config.dropout,
)
target_net.load_state_dict(policy_net.state_dict())

optimizer = config.optimizer(policy_net.parameters(), lr=LR)  # pyright: ignore[reportCallIssue]
feat_extractor: feature_extractor.FeatureExtractor = config.feature_extractor(env)


steps_train_done = 0
trains_done = 0
tests_done = 0


def select_action(state: torch.Tensor):
    global epsilon
    sample = random.random()
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    need_explore = sample < epsilon
    if need_explore:
        action = env.action_space.sample()
        return torch.tensor([[action]], device=state.device, dtype=torch.long)

    with torch.no_grad():
        q_values: torch.Tensor = policy_net(state)
        action = q_values.argmax(dim=1, keepdim=True)
        return action


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    indices, weights, transitions = memory.sample(BATCH_SIZE)
    batch = TransposedTransition(*zip(*transitions, strict=True))

    # non_final_mask = torch.tensor(
    #     [s is not None for s in batch.next_states],
    #     device=config.device,
    #     dtype=torch.bool,
    # )

    # non_final_next_states = torch.cat(
    #     [s for s in batch.next_states if s is not None],
    # )

    state_batch = torch.cat(batch.states)
    action_batch = torch.cat(batch.actions)
    reward_batch = torch.cat(batch.rewards)
    next_state_batch = torch.cat(batch.next_states)
    is_non_final_batch = ~torch.tensor(
        batch.dones,
        device=config.device,
        dtype=torch.bool,
    )
    non_terminated_next_states = next_state_batch[is_non_final_batch]

    # weights = torch.cat(weights)

    # Compute Q-values for all actions (shape: [batch_size, n_actions])
    q_values: torch.Tensor = policy_net(state_batch)
    # Select Q-values for the taken actions (shape: [batch_size, 1])
    state_action_values = q_values.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states. But only for non-final next states.
    # (the one that are not marked as done). Otherwise it's 0 by definition of V.
    next_state_values = torch.zeros(BATCH_SIZE, device=config.device)
    with torch.no_grad():
        next_q_values: torch.Tensor = target_net(non_terminated_next_states)
        non_terminated_next_state_values = next_q_values.max(dim=1).values
        next_state_values[is_non_final_batch] = non_terminated_next_state_values

    expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).detach()

    criterion = config.value_loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # loss = (weights * F.smooth_l1_loss(
    #     state_action_values,
    #     expected_state_action_values.unsqueeze(1),
    #     reduction="none",
    # )).mean()

    # optimize the model
    optimizer.zero_grad()
    loss.backward()

    # in-place gradient clipping in case of exploding gradients
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    logger["loss"] = loss.item()
    optimizer.step()


def test_agent(num_episodes: int):
    test_rewards: list[float] = []
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = feat_extractor.get_features(state)
        state = torch.tensor(
            state,
            device=config.device,
            dtype=torch.float32,
        )
        total_reward = 0.0
        for t in count():
            with torch.no_grad():
                q_values: torch.Tensor = policy_net(state)
                action = q_values.argmax(dim=1, keepdim=True)

            observation, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            total_reward += float(reward)

            if done:
                break

            next_state = feat_extractor.get_features(observation)
            state = torch.tensor(
                next_state,
                device=config.device,
                dtype=torch.float32,
            )
        test_rewards.append(total_reward)

    global tests_done
    logger.in_term = True
    logger.direct_log("test_reward_mean", statistics.fmean(test_rewards), tests_done)
    tests_done += 1


def train_agent(num_episode: int):
    for i_episode in range(num_episode):
        total_reward = 0.0

        state, info = env.reset()
        state = feat_extractor.get_features(state)
        state = torch.tensor(
            state,
            device=config.device,
            dtype=torch.float32,
        )
        for t in count():
            global steps_train_done
            steps_train_done += 1
            action = select_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=config.device)
            done = terminated or truncated
            total_reward += reward.item()

            next_state = feat_extractor.get_features(observation)
            next_state = torch.tensor(
                next_state,
                device=config.device,
                dtype=torch.float32,
            )
            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            state = next_state
            memory.push(transition)
            optimize_model()

            if steps_train_done % config.freq_optim == 0:
                target_net.load_state_dict(policy_net.state_dict())

            global trains_done

            logger.in_term = trains_done != 0 and (
                trains_done % config.freq_verbose == 0
            )
            if done or t >= config.max_length_train:
                logger["reward"] = total_reward
                logger["epsilon"] = epsilon
                logger.log(trains_done)
                trains_done += 1
                break


def main():
    num_episodes = config.nb_episodes
    test_interval = config.freq_test

    # batch means e.g. batched("ABCDEFGHIJ", 3) -> ["ABC", "DEF", "GHI", "J"]
    # in our case, batched([0..99], 10) -> [[0..9], [10..19], ..., [90..99]]
    for batch in itertools.batched(range(num_episodes), test_interval, strict=True):
        train_agent(len(batch))
        test_agent(config.nb_tests)


if __name__ == "__main__":
    main()
