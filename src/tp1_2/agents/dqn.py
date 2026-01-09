from __future__ import annotations

import itertools
import random
import statistics
from itertools import count
from typing import TYPE_CHECKING

import torch

from tp1_2 import setup
from tp1_2.memory import (
    ReplayMemory,
    Transition,
    TransposedTransitionTensor,
)
from tp1_2.network import NeuralNetwork

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from tp1_2 import feature_extractor
    from tp1_2.config import ConfigDQN
    from tp1_2.setup import SetupEnv


class Agent:
    """Base class for RL agents. Use this agent only for evaluation, when loading
    a pre-trained model (`from_dir`).
    """

    def __init__(
        self,
        env: setup.DualEnvWrapper,
        config: ConfigDQN,
        outdir: Path,
    ):
        self.outdir = outdir
        self.config = config
        self.env = env
        self.policy_network: torch.nn.Module = NeuralNetwork(1, 1)
        self._init_feature_extractor()

    def load_model(self, path: Path | None = None):
        """Load policy network parameters θ."""
        if path is None:
            path = self.outdir / "policy_network.pth"
        if not path.exists():
            msg = f"No model found at {path}"
            raise FileNotFoundError(msg)
        self.policy_network = torch.load(
            path,
            weights_only=False,
            map_location=self.config.device_torch,
        )
        return self

    @classmethod
    def from_dir(
        cls,
        path: Path,
    ):
        config_file = path / "config.toml"
        gym_env, config = setup.init_raw_env(config_file=config_file)
        agent = cls(
            env=gym_env,
            config=config,
            outdir=path,
        )
        agent.load_model(path / "policy_network.pth")
        return agent

    def _init_feature_extractor(self):
        """Initialize feature extractor φ: observation → features."""
        self.feat_extractor: feature_extractor.FeatureExtractor = (
            self.config.feature_extractor(self.env)
        )

    def select_action(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            q_values: torch.Tensor = self.policy_network(state)
            return q_values.argmax(dim=1, keepdim=True)

    def evaluate_episode(self, *args, render: bool = False, **kwargs) -> float:
        """Run one episode with greedy policy (no exploration).

        Args:
            render (bool): Whether to render the environment in a human-readable way.
            *args, **kwargs: Additional arguments to pass to env.reset().

        Returns:
            total_reward: float. Reward accumulated during the episode.
        """
        self.policy_network.eval()
        if render:
            state, _info = self.env.reset_human(*args, **kwargs)
        else:
            state, _info = self.env.reset_test(*args, **kwargs)
        state = self._preprocess_state(state)
        total_reward = 0.0

        while True:
            # Greedy action selection
            action = self.select_action(state)
            observation, reward, terminated, truncated, _info = self.env.step(
                action.item(),
            )
            done = terminated or truncated
            total_reward += float(reward)

            if done:
                break

            state = self._preprocess_state(observation)

        self.policy_network.train()
        return total_reward

    def _preprocess_state(self, state) -> torch.Tensor:
        """Apply feature extraction φ(s) and convert to tensor."""
        features = self.feat_extractor.get_features(state)
        return torch.tensor(
            features,
            device=self.config.device,
            dtype=torch.float32,
        )


class DQNAgent(Agent):
    def __init__(self, setup_env: SetupEnv[ConfigDQN]):
        self.env, self.config, self.outdir, self.logger = setup_env
        self._init_hyperparameters()
        self._init_memory()
        self._init_networks()
        self._init_optimizer()
        self._init_feature_extractor()
        self._init_counters()

    @classmethod
    def from_model_path(
        cls,
        path: Path,
    ):
        setup_env = setup.init_env(
            config_file=path.parent / "config.toml",
            name="loaded_agent",
            launch_tensorboard=False,
            kind=ConfigDQN,
        )
        agent = cls(setup_env)
        agent.load_model(path)
        return agent

    def _init_hyperparameters(self):
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma  # γ: discount factor
        self.epsilon = self.config.epsilon_start  # ε: exploration rate
        self.epsilon_end = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.learning_rate = self.config.learning_rate  # α

    def _init_memory(self):
        """Initialize replay buffer D (optional).

        If replay_memory_capacity is 0 or None, memory is disabled
        and we do online learning (update after each transition).
        """
        self.memory = (
            ReplayMemory(
                capacity=self.config.replay_memory_capacity,
                prioritized=self.config.prioritized_replay,
            )
            if self.config.replay_memory_capacity
            else None
        )

    def _init_networks(self):
        """Initialize Q_θ(s,a) and Q_θ'(s,a) networks.

        Q_θ: S x A → R (policy network)
        Q_θ': S x A → R (target network)

        Implementation: Q_θ: S → R^|A|, output Q(s,·) for all actions
        """
        self.n_actions: int = self.env.action_space.n  # type: ignore[reportAttributeAccessIssue]
        state, _info = self.env.reset()
        self.n_observations = len(state)

        print(f"Number of actions: {self.n_actions}")
        print(f"Number of observations: {self.n_observations}")
        print(f"Output directory: {self.outdir}")

        self.policy_network = self._create_network()  # Q_θ
        if self.config.use_target_network:
            self.target_network = self._create_network()  # Q_θ'
            self.target_network.load_state_dict(self.policy_network.state_dict())
        else:
            self.target_network = self.policy_network

    def _create_network(self):
        """Create neural network for Q(s,a) approximation."""
        return NeuralNetwork(
            input_size=self.n_observations,
            output_size=self.n_actions,
            layers=self.config.hidden_layers,
            activation=self.config.hidden_layers_activation,
            final_activation=self.config.final_layer_activation,
            dropout=self.config.dropout,
        ).to(self.config.device)

    def _init_optimizer(self):
        """Initialize optimizer for gradient descent."""
        self.optimizer = self.config.optimizer(
            self.policy_network.parameters(),
            lr=self.learning_rate,  # pyright: ignore[reportCallIssue]
        )

    def _init_counters(self):
        self.steps_train_done = 0
        self.trains_done = 0
        self.tests_done = 0

    def select_action(
        self,
        state: torch.Tensor,
        *,
        maybe_explore: bool = False,
    ) -> torch.Tensor:
        """ε-greedy policy: π_ε(s).

        a_t = argmax_a Q_θ(s_t, a)     with probability 1-ε
              uniform random           with probability ε

        After selection: ε ← max(ε_min, ε · ε_decay)

        If maybe_explore is False, only used to evaluate the action:
        no exploration, no ε change.
        """
        if maybe_explore:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
                return torch.tensor([[action]], device=state.device, dtype=torch.long)

        # Greedy action: argmax_a Q_θ(s, a)
        return super().select_action(state)
        # with torch.no_grad():
        #     q_values: torch.Tensor = self.policy_network(state)
        #     return q_values.argmax(dim=1, keepdim=True)

    def store(self, transition: Transition):
        """Store transition (s, a, r, s', done) in replay buffer D.

        No-op if memory is disabled.
        """
        if self.memory is not None:
            self.memory.push(transition)

    @property
    def enough_samples(self) -> bool:
        """Check if |D| ≥ min_size.

        Always True if memory is disabled (online learning).
        """
        if self.memory is None:
            return True
        return len(self.memory) >= self.config.replay_memory_min_size

    @property
    def time_to_learn(self) -> bool:
        """Check if we should perform a gradient update.

        With memory: need enough samples and train_freq intervals
        Without memory: learn after every step
        """
        if self.memory is None:
            return True
        return self.enough_samples and (
            self.steps_train_done % self.config.train_freq == 0
        )

    @property
    def time_to_update_target(self) -> bool:
        """Check if we should update θ' ← θ (hard) or θ' ← τθ + (1-τ)θ' (soft)."""
        # skip if not target network
        if self.target_network is self.policy_network:
            return False

        # soft update: update every step
        if self.config.target_soft_update:
            return True

        # hard update: every target_update_freq steps
        # (only after enough samples if using memory)
        if self.memory is not None and not self.enough_samples:
            return False

        return self.steps_train_done % self.config.target_update_freq == 0

    # ---------------------------------------------------------------------------- #
    #               Core RL mathematics: Q-values, V-values, TD error              #
    # ---------------------------------------------------------------------------- #

    def compute_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """Compute Q_θ(s, ·) for all actions.

        Returns: tensor of shape (batch_size, n_actions)
        """
        return self.policy_network(states)

    def compute_q_target_values(self, states: torch.Tensor) -> torch.Tensor:
        """Compute Q_θ'(s, ·) for all actions using target network.

        Returns: tensor of shape (batch_size, n_actions)
        """
        return self.target_network(states)

    def compute_state_action_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q_θ(s, a) for given state-action pairs.

        Q(s_i, a_i) for i in batch

        Returns: tensor of shape (batch_size, 1)
        """
        q_values = self.compute_q_values(states)
        return q_values.gather(1, actions)

    def compute_v_values_target(self, states: torch.Tensor) -> torch.Tensor:
        """Compute V(s) = max_a Q_θ'(s, a) using target network.

        This is the estimated value of being in state s under the greedy policy.

        Returns: tensor of shape (batch_size,)
        """
        with torch.no_grad():
            q_values = self.compute_q_target_values(states)
            return q_values.max(dim=1).values

    def compute_next_state_values(
        self,
        next_states: torch.Tensor,
        is_non_terminal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute V(s') for next states (0 if terminal).

        V(s') = max_a Q_θ'(s', a)  if s' is non-terminal
                0                   if s' is terminal

        Override this method for Double DQN, etc.

        Returns: tensor of shape (batch_size,)
        """
        batch_size = len(next_states)
        next_values = torch.zeros(batch_size, device=self.config.device)

        if is_non_terminal.any():
            non_terminal_next_states = next_states[is_non_terminal]
            next_values[is_non_terminal] = self.compute_v_values_target(
                non_terminal_next_states,
            )

        return next_values

    def compute_temporal_diff_targets(
        self,
        rewards: torch.Tensor,
        next_state_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TD target: y = r + γ·V(s').

        This is the bootstrap estimate of Q(s,a).

        Returns: tensor of shape (batch_size, 1)
        """
        return (rewards + self.gamma * next_state_values).unsqueeze(1).detach()

    def compute_temporal_diff_error(
        self,
        current_Q: torch.Tensor,
        target_Q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal difference error given current and target Q-values.

        Element-wise TD errors (before reduction).
        Can be used for prioritized replay buffer updates.

        Returns: tensor of shape (batch_size, 1)
        """
        criterion = self.config.value_loss(reduction="none")
        return criterion(current_Q, target_Q)

    def compute_loss(
        self,
        batch: TransposedTransitionTensor,
        weights: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Compute the DQN loss: L(θ).

        Steps:
        1. Q_θ(s,a) = current Q-value estimates
        2. V(s') = max_a' Q_θ'(s', a') for non-terminal s'
        3. y = r + γ·V(s') = TD target
        4. l(θ) element-wise TD errors
        5. L(θ) = mean(l(θ)) or weighted mean if prioritized
        """
        # Step 1: Q_θ(s, a)
        current_Q = self.compute_state_action_values(batch.states, batch.actions)

        # Step 2: V(s') = max_a' Q_θ'(s', a') if non-terminal, 0 if terminal
        next_state_values = self.compute_next_state_values(
            batch.next_states,
            ~batch.dones,
        )

        # Step 3: y = r + γ·V(s')
        target_Q = self.compute_temporal_diff_targets(batch.rewards, next_state_values)

        # Step 4: l(θ)
        td_errors = self.compute_temporal_diff_error(current_Q, target_Q)

        # Step 5: L(θ)
        loss = self.reduce_loss(td_errors, weights)

        # Logging
        with torch.no_grad():
            self.logger["optim/loss"] = loss.item()
            self.logger["optim/td_error_mean"] = td_errors.mean().item()
            self.logger["train/q_value_mean"] = current_Q.mean().item()
            self.logger["train/target_q_value_mean"] = target_Q.mean().item()

        return loss

    def reduce_loss(
        self,
        element_wise_loss: torch.Tensor,
        weights: np.ndarray | None,
    ) -> torch.Tensor:
        """Reduce element-wise losses to scalar.

        - Standard replay: mean(losses)
        - Prioritized replay: mean(weights * losses)
        """
        if weights is not None:
            weights_tensor = torch.tensor(
                weights,
                device=self.config.device,
                dtype=element_wise_loss.dtype,
            ).unsqueeze(1)
            return (element_wise_loss * weights_tensor).mean()
        return element_wise_loss.mean()

    # ---------------------------------------------------------------------------- #
    #                               Learning methods                               #
    # ---------------------------------------------------------------------------- #

    def sample_transitions(
        self,
    ) -> tuple[np.ndarray | None, TransposedTransitionTensor]:
        """Sample transitions for learning.

        With memory: sample random minibatch from D
        Without memory: use the current transition buffer

        Returns:
            weights: importance sampling weights (None if not prioritized)
            batch: TransposedTransitionTensor containing the transitions
        """
        if self.memory is not None:
            _indices, weights, transitions = self.memory.sample(self.batch_size)
        else:
            weights = None
            transitions = self._current_transition_buffer

        batch = TransposedTransitionTensor.from_list_of_transitions(
            transitions,
            device=self.config.device,
        )
        return weights, batch

    def learn(self):
        """Perform one gradient descent step.

        With memory: sample minibatch from D
        Without memory: learn from current transition(s)
        """
        if not self.time_to_learn:
            return

        weights, batch = self.sample_transitions()

        # Compute loss
        loss = self.compute_loss(batch, weights)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.policy_network.parameters(),
                max_norm=self.config.clip_grad_norm,
            )

        self.optimizer.step()

    def update_target_network(self):
        """Update target network parameters θ'.

        Hard update: θ' ← θ
        Soft update: θ' ← τθ + (1-τ)θ'
        """
        # update only if target network is separate
        if self.target_network is self.policy_network:
            return
        tau = self.config.target_soft_update or 1.0
        target_net_state = self.target_network.state_dict()
        policy_net_state = self.policy_network.state_dict()

        for key in target_net_state:
            target_net_state[key] = (
                tau * policy_net_state[key] + (1 - tau) * target_net_state[key]
            )

        self.target_network.load_state_dict(target_net_state)

    # ---------------------------------------------------------------------------- #
    #                                 Training loop                                #
    # ---------------------------------------------------------------------------- #

    def train_episode(self) -> float:
        """Run one episode of training.

        With memory: store transitions in D, learn from random samples
        Without memory: learn immediately from each transition (online)
        """
        state, _info = self.env.reset_train()
        state = self._preprocess_state(state)
        total_reward = 0.0

        for t in count(start=1):
            self.steps_train_done += 1

            # Select action using ε-greedy policy
            action = self.select_action(state, maybe_explore=True)

            # Execute action in environment
            observation, reward, terminated, truncated, _info = self.env.step(
                action.item(),
            )
            done = terminated or truncated
            total_reward += float(reward)

            # Prepare transition
            next_state = self._preprocess_state(observation)
            reward_tensor = torch.tensor([reward], device=self.config.device)
            transition = Transition(
                state=state,
                action=action,
                reward=reward_tensor,
                next_state=next_state,
                done=torch.tensor([done], device=self.config.device, dtype=torch.bool),
            )

            # Store or buffer transition
            if self.memory is not None:
                self.store(transition)
            else:
                # For online learning, keep transition in buffer
                self._current_transition_buffer = [transition]

            # Perform learning step
            self.learn()

            # Update target network
            if self.time_to_update_target:
                self.update_target_network()

            # Transition to next state
            state = next_state

            # Check termination
            if done or t >= self.config.max_length_train:
                break

        return total_reward

    def train_agent(self, num_episodes: int):
        """Train for multiple episodes."""
        for _ in range(num_episodes):
            total_reward = self.train_episode()

            # Log metrics
            self.logger["train/reward"] = total_reward
            self.logger["misc/epsilon"] = self.epsilon
            verbose = self.config.freq_verbose_train > 0 and (
                self.trains_done % self.config.freq_verbose_train == 0
            )
            self.logger.log(
                self.trains_done,
                verbose=verbose,
            )
            self.trains_done += 1

            if (self.trains_done % self.config.freq_save) == 0:
                self.save_model()

    # ---------------------------------------------------------------------------- #
    #                                  Evaluation                                  #
    # ---------------------------------------------------------------------------- #

    def evaluate_episode(self, *, render: bool = False) -> float:
        """Run one episode with greedy policy (no exploration).

        Returns:
            total_reward: float. Reward accumulated during the episode.
        """
        if render:
            state, _info = self.env.reset_human()
        else:
            state, _info = self.env.reset_test()
        state = self._preprocess_state(state)
        total_reward = 0.0

        while True:
            # Greedy action selection
            action = self.select_action(state)
            observation, reward, terminated, truncated, _info = self.env.step(
                action.item(),
            )
            done = terminated or truncated
            total_reward += float(reward)

            if done:
                break

            state = self._preprocess_state(observation)

        return total_reward

    def test_agent(self, num_episodes: int):
        """Test for multiple episodes and log mean reward."""
        test_rewards = [self.evaluate_episode() for _ in range(num_episodes)]
        mean_rewards = statistics.fmean(test_rewards)
        verbose = self.config.freq_verbose_test > 0 and (
            self.tests_done % self.config.freq_verbose_test == 0
        )
        self.logger.direct_log(
            "test_reward_mean",
            mean_rewards,
            self.tests_done,
            verbose=verbose,
        )
        self.tests_done += 1
        return mean_rewards

    # ---------------------------------------------------------------------------- #
    #                                  Persistence                                  #
    # ---------------------------------------------------------------------------- #

    def save_model(self, path: Path | None = None):
        """Save policy network parameters θ."""
        if path is None:
            path = self.outdir / "policy_network.pth"
        torch.save(self.policy_network, path)
        return self

    def load_model(self, path: Path | None = None):
        """Load policy network parameters θ and sync target network θ'."""
        if path is None:
            path = self.outdir / "policy_network.pth"
        if not path.exists():
            msg = f"No model found at {path}"
            raise FileNotFoundError(msg)
        self.policy_network.load_state_dict(
            torch.load(path, weights_only=True, map_location=self.config.device_torch),
        )
        self.target_network.load_state_dict(self.policy_network.state_dict())
        return self

    # ---------------------------------------------------------------------------- #
    #                                   Main loop                                  #
    # ---------------------------------------------------------------------------- #

    def run(self):
        """Main training loop with periodic testing."""
        num_episodes = self.config.nb_episodes
        test_interval = self.config.freq_test

        for batch in itertools.batched(range(num_episodes), test_interval, strict=True):
            self.train_agent(len(batch))
            test_reward = self.test_agent(self.config.nb_tests)
            stop_threshold = self.config.stop_when_test_higher_than
            if test_reward >= stop_threshold:
                print(
                    f"Stopping training as test reward {test_reward} "
                    f"exceeds threshold {stop_threshold}.",
                )
                break
        self.save_model()
        print(f"Training complete. Model saved at {self.outdir}.")
        return self


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent."""

    def select_best_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Select argmax_a Q_θ(s, a) for each state 's'.

        Used in Double DQN to decouple action selection from evaluation.

        Returns: tensor of shape (batch_size, 1) (batch, action)
        """
        with torch.no_grad():
            q_values = self.compute_q_values(states)
            return q_values.argmax(dim=1, keepdim=True)

    def compute_next_state_values(
        self,
        next_states: torch.Tensor,
        is_non_terminal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute V(s') for next states (0 if terminal) using Double DQN.

        V(s') = Q_θ'(s', argmax_a' Q_θ(s', a'))  if s' is non-terminal
                0                             if s' is terminal

        Double DQN decouples action selection and evaluation to reduce
        overestimation bias. For each non-terminal next state:
        - Policy network Q_θ selects the best action
        - Target network Q_θ' evaluates that action's value

        Returns: tensor of shape (batch_size,)
        """
        batch_size = len(next_states)
        next_values = torch.zeros(batch_size, device=self.config.device)

        if is_non_terminal.any():
            non_terminal_next_states = next_states[is_non_terminal]
            # Select best actions using policy network
            with torch.no_grad():
                # Step 1: select best actions using policy network Q_θ
                policy_q_values = self.compute_q_values(non_terminal_next_states)
                best_actions = policy_q_values.argmax(dim=1, keepdim=True)

                # Step 2: evaluate best actions using target network Q_θ'
                target_q_values = self.compute_q_target_values(non_terminal_next_states)
                next_values[is_non_terminal] = target_q_values.gather(
                    1,
                    best_actions,
                ).squeeze(1)
                # need to squeeze to get shape (batch_size,)
                # ([batch_size,1] -> [batch_size])

        return next_values
