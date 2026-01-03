from __future__ import annotations

import itertools
import random
import statistics
import sys
from itertools import count
from typing import TYPE_CHECKING

import torch

from tp1_2.memory import ReplayMemory, Transition, TransposedTransition
from tp1_2.network import NeuralNetwork
from tp1_2.setup import SetupEnv, init_env
from einops import rearrange

if TYPE_CHECKING:
    import numpy as np

    from tp1_2 import feature_extractor


class Agent:
    def __init__(self, setup_env: SetupEnv):
        self.env, self.config, self.outdir, self.logger = setup_env
        self._init_hyperparameters()
        self._init_memory()
        self._init_networks()
        self._init_optimizer()
        self._init_feature_extractor()
        self._init_counters()

    def _init_hyperparameters(self):
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.learning_rate = self.config.learning_rate

    def _init_memory(self):
        self.memory = ReplayMemory(
            capacity=self.config.replay_memory_capacity,
            prioritized=self.config.prioritized_replay,
        )

    def _init_networks(self):
        self.n_actions: int = self.env.action_space.n
        state, info = self.env.reset()
        self.n_observations = len(state)

        print(f"Number of actions: {self.n_actions}")
        print(f"Number of observations: {self.n_observations}")
        print(f"Output directory: {self.outdir}")

        self.policy_network = NeuralNetwork(
            input_size=self.n_observations,
            output_size=self.n_actions,
            layers=self.config.hidden_layers,
            activation=self.config.hidden_layers_activation,
            final_activation=self.config.final_layer_activation,
            dropout=self.config.dropout,
        )

        self.target_network = NeuralNetwork(
            input_size=self.n_observations,
            output_size=self.n_actions,
            layers=self.config.hidden_layers,
            activation=self.config.hidden_layers_activation,
            final_activation=self.config.final_layer_activation,
            dropout=self.config.dropout,
        )
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def _init_optimizer(self):
        self.optimizer = self.config.optimizer(
            self.policy_network.parameters(),
            lr=self.learning_rate,  # pyright: ignore[reportCallIssue]
        )

    def _init_feature_extractor(self):
        self.feat_extractor: feature_extractor.FeatureExtractor = (
            self.config.feature_extractor(self.env)
        )

    def _init_counters(self):
        self.steps_train_done = 0
        self.trains_done = 0
        self.tests_done = 0

    def _preprocess_state(self, state) -> torch.Tensor:
        """Convert raw state to tensor with feature extraction."""
        features = self.feat_extractor.get_features(state)
        return torch.tensor(
            features,
            device=self.config.device,
            dtype=torch.float32,
        )

    def select_action(
        self,
        state: torch.Tensor,
        *,
        maybe_explore: bool = True,
    ) -> torch.Tensor:
        """Select action using epsilon-greedy policy."""
        if maybe_explore:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
                return torch.tensor([[action]], device=state.device, dtype=torch.long)

        with torch.no_grad():
            q_values: torch.Tensor = self.policy_network(state)
            action = q_values.argmax(dim=1, keepdim=True)
            return action

    def store(self, transition: Transition):
        """Store transition in replay memory."""
        self.memory.push(transition)

    def time_to_learn(self) -> bool:
        """Check if we have enough samples to start learning."""
        return len(self.memory) >= self.batch_size

    def time_to_update_target(self) -> bool:
        """Check if it's time to update target network."""
        return self.steps_train_done % self.config.freq_optim == 0

    def learn(self):
        """Sample from memory and optimize the policy network."""
        if not self.time_to_learn():
            return

        indices, weights, transitions = self.memory.sample(self.batch_size)
        batch = TransposedTransition(*zip(*transitions, strict=True))

        # state_batch = torch.cat(batch.states)
        state_batch = rearrange(torch.stack(batch.states), "batch 1 features -> batch features")
        action_batch = torch.cat(batch.actions)
        reward_batch = torch.cat(batch.rewards)
        next_state_batch = torch.cat(batch.next_states)
        is_non_final_batch = ~torch.tensor(
            batch.dones,
            device=self.config.device,
            dtype=torch.bool,
        )

        # Compute current Q values
        q_values: torch.Tensor = self.policy_network(state_batch)
        state_action_values = q_values.gather(1, action_batch)

        # Compute next state values using target network
        next_state_values = torch.zeros(self.batch_size, device=self.config.device)
        if is_non_final_batch.any():
            non_terminated_next_states = next_state_batch[is_non_final_batch]
            with torch.no_grad():
                next_q_values: torch.Tensor = self.target_network(
                    non_terminated_next_states,
                )
                next_state_values[is_non_final_batch] = next_q_values.max(dim=1).values

        # Compute expected Q values
        expected_state_action_values = (
            (next_state_values * self.gamma + reward_batch).unsqueeze(1).detach()
        )

        # Compute loss and update
        criterion = self.config.value_loss(reduction="none")
        element_wise_loss = criterion(state_action_values, expected_state_action_values)
        loss = self.reduce_loss(element_wise_loss, weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger["loss"] = loss.item()

    def reduce_loss(
        self,
        element_wise_loss: torch.Tensor,
        weights: np.ndarray | None,
    ) -> torch.Tensor:
        """Reduce loss based on whether prioritized replay is used."""
        if weights is not None:
            weights_tensor = torch.tensor(
                weights,
                device=element_wise_loss.device,
                dtype=element_wise_loss.dtype,
            ).unsqueeze(1)
            return (element_wise_loss * weights_tensor).mean()
        return element_wise_loss.mean()

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train_episode(self) -> float:
        """Train for one episode and return total reward."""
        state, info = self.env.reset()
        state = self._preprocess_state(state)
        total_reward = 0.0

        for t in count():
            self.steps_train_done += 1

            # Select and execute action
            action = self.select_action(state, maybe_explore=True)
            observation, reward, terminated, truncated, info = self.env.step(
                action.item(),
            )
            done = terminated or truncated
            total_reward += float(reward)

            # Process next state
            next_state = self._preprocess_state(observation)
            reward_tensor = torch.tensor([reward], device=self.config.device)

            # Store transition
            transition = Transition(
                state=state,
                action=action,
                reward=reward_tensor,
                next_state=next_state,
                done=done,
            )
            self.store(transition)

            # Learn from experience
            self.learn()

            # Update target network
            if self.time_to_update_target():
                self.update_target_network()

            # Move to next state
            state = next_state

            # Check episode termination
            if done or t >= self.config.max_length_train:
                break

        return total_reward

    def train_agent(self, num_episodes: int):
        """Train for multiple episodes."""
        for i_episode in range(num_episodes):
            total_reward = self.train_episode()

            # Log metrics
            self.logger.in_term = self.trains_done != 0 and (
                self.trains_done % self.config.freq_verbose == 0
            )
            self.logger["reward"] = total_reward
            self.logger["epsilon"] = self.epsilon
            self.logger.log(self.trains_done)
            self.trains_done += 1

    def test_episode(self) -> float:
        """Test for a single episode and return total reward."""
        state, info = self.env.reset()
        state = self._preprocess_state(state)
        total_reward = 0.0

        for t in count():
            # Select action without exploration
            action = self.select_action(state, maybe_explore=False)
            observation, reward, terminated, truncated, info = self.env.step(
                action.item(),
            )
            done = terminated or truncated
            total_reward += float(reward)

            if done:
                break

            state = self._preprocess_state(observation)

        return total_reward

    def test_agent(self, num_episodes: int):
        """Test for multiple episodes and log mean reward.

        Returns:
            float: Mean reward over the test episodes. Not really used,
                we only need to log it in TensorBoard.
        """
        test_rewards = [self.test_episode() for _ in range(num_episodes)]
        mean_rewards = statistics.fmean(test_rewards)
        self.logger.in_term = True
        self.logger.direct_log(
            "test_reward_mean",
            mean_rewards,
            self.tests_done,
        )
        self.tests_done += 1
        return mean_rewards

    def save_model(self, path: str):
        """Save the policy network to the specified path."""
        torch.save(self.policy_network.state_dict(), path)

    def load_model(self, path: str):
        """Load the policy network from the specified path."""
        self.policy_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def run(self):
        """Main training loop with periodic testing."""
        num_episodes = self.config.nb_episodes
        test_interval = self.config.freq_test

        for batch in itertools.batched(range(num_episodes), test_interval, strict=True):
            self.train_agent(len(batch))
            self.test_agent(self.config.nb_tests)


def main():

    # get nb of threads
    n_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads * 2 - 1)
    config_path = "./configs/my_config_dqn_cartpole.toml"
    name = "dqn_cartpole"
    setup_env = init_env(config_path, name)
    agent = Agent(setup_env)
    agent.run()


if __name__ == "__main__":
    main()
