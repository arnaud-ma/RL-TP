from __future__ import annotations

import statistics
from itertools import count
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from tp1_2 import setup
from tp1_2.network import NeuralNetwork
from tp1_2.config import ConfigPPO

if TYPE_CHECKING:
    from pathlib import Path

    from tp1_2 import feature_extractor
    from tp1_2.setup import SetupEnv


class Trajectory:
    """Container for a single episode trajectory.

    Stores the sequence of (s_t, a_t, r_t, log_π(a_t|s_t)) for one episode.
    """

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ):
        """Add one timestep to trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)

    def to_tensors(self, device: torch.device):
        """Convert lists to tensors for batch processing.

        Returns:
            states: (T, state_dim)
            actions: (T, 1)
            rewards: (T,)
            log_probs: (T,)
            values: (T,)
            dones: (T,)
        """
        states = torch.cat(self.states, dim=0).to(device)
        actions = torch.cat(self.actions, dim=0).to(device)
        rewards = torch.tensor(self.rewards, device=device, dtype=torch.float32)
        log_probs = torch.cat(self.log_probs, dim=0).to(device)
        values = torch.cat(self.values, dim=0).to(device)
        dones = torch.tensor(self.dones, device=device, dtype=torch.bool)

        # Fix shapes if they have extra dimensions (e.g. from state being [1, 1, 8])
        if states.dim() == 3:  # (T, 1, D) -> (T, D)
            states = states.squeeze(1)

        if actions.dim() == 3:  # (T, 1, 1) -> (T, 1)
            actions = actions.squeeze(1)

        if log_probs.dim() == 2:  # (T, 1) -> (T,)
            log_probs = log_probs.squeeze(1)

        if values.dim() == 2:  # (T, 1) -> (T,)
            values = values.squeeze(1)

        return states, actions, rewards, log_probs, values, dones


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent with KL-adaptive learning.

    PPO learns a stochastic policy π_θ(a|s) that outputs action probabilities,
    along with a value function V_φ(s) that estimates expected returns.

    Key components:
    - Policy network π_θ: S → Δ(A) (probability distribution over actions)
    - Value network V_φ: S → ℝ (expected return from state)
    - Advantage function A^π(s,a) = Q^π(s,a) - V^π(s) (how good is action vs average)
    - KL divergence D_KL(π_old || π_new) (measure of policy change)
    """

    def __init__(self, setup_env: SetupEnv[ConfigPPO]):
        self.env, self.config, self.outdir, self.logger = setup_env
        self._init_hyperparameters()
        self._init_networks()
        self._init_optimizers()
        self._init_feature_extractor()
        self._init_counters()

    def _init_hyperparameters(self):
        """Initialize PPO hyperparameters."""
        # Discount and GAE
        self.gamma = self.config.gamma  # γ: discount factor for returns
        self.gae_lambda = self.config.gae_lambda  # λ: GAE parameter (0=TD, 1=MC)

        # PPO clipping
        self.clip_epsilon = self.config.clip_epsilon  # ε: clip range for ratio

        # KL-adaptive
        self.beta = self.config.beta_init  # β: KL penalty coefficient
        self.target_kl = self.config.target_kl  # δ: target KL divergence

        # Training
        self.num_epochs = self.config.num_epochs  # K: epochs per policy update
        self.batch_size = self.config.batch_size  # minibatch size

        # Loss coefficients
        self.value_loss_coef = self.config.value_loss_coef  # c₁
        self.entropy_coef = self.config.entropy_coef  # c₂

        # Learning rates
        self.policy_lr = self.config.policy_lr  # α for policy
        self.value_lr = self.config.value_lr  # α for value

    def _init_networks(self):
        """Initialize π_θ(a|s) and V_φ(s) networks.

        Policy network π_θ: S → R^|A| → Δ(A) via softmax
            - Input: state features
            - Output: logits for each action
            - Softmax converts to probabilities:
                π_θ(a|s) = exp(logit_a) / Σ_a' exp(logit_a')

        Value network V_φ: S → R
            - Input: state features
            - Output: scalar value estimate V_φ(s)
        """
        self.n_actions: int = self.env.action_space.n
        state, _info = self.env.reset()
        self.n_observations = len(state)

        print(f"Number of actions: {self.n_actions}")
        print(f"Number of observations: {self.n_observations}")
        print(f"Output directory: {self.outdir}")

        # Policy network: outputs logits for action probabilities
        self.policy_network = self._create_policy_network()

        # Value network: outputs scalar value estimate
        self.value_network = self._create_value_network()

    def _create_policy_network(self) -> torch.nn.Module:
        """Create policy network π_θ: S → R^|A|.

        The network outputs raw logits (unnormalized scores).
        Softmax is applied later: π_θ(a|s) = softmax(logits)_a
        """
        return NeuralNetwork(
            input_size=self.n_observations,
            output_size=self.n_actions,  # One logit per action
            layers=self.config.policy_hidden_layers,
            activation=self.config.policy_activation,
            final_activation=torch.nn.Identity,  # No activation (raw logits)
            dropout=self.config.policy_dropout,
        ).to(self.config.device)

    def _create_value_network(self) -> torch.nn.Module:
        """Create value network V_φ: S → R.

        The network outputs a single scalar: the estimated value V_φ(s).
        """
        return NeuralNetwork(
            input_size=self.n_observations,
            output_size=1,  # Single scalar output
            layers=self.config.value_hidden_layers,
            activation=self.config.value_activation,
            final_activation=torch.nn.Identity,  # No activation (raw value)
            dropout=self.config.value_dropout,
        ).to(self.config.device)

    def _init_optimizers(self):
        """Initialize optimizers for policy and value networks.

        Separate optimizers allow different learning rates and update schedules.
        """
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.policy_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=self.value_lr,
        )

    def _init_feature_extractor(self):
        """Initialize feature extractor φ: observation → features."""
        self.feat_extractor: feature_extractor.FeatureExtractor = (
            self.config.feature_extractor(self.env)
        )

    def _init_counters(self):
        """Initialize training counters."""
        self.steps_done = 0
        self.updates_done = 0
        self.episodes_done = 0

    def _preprocess_state(self, state) -> torch.Tensor:
        """Apply feature extraction φ(s) and convert to tensor.

        Returns: tensor of shape (1, state_dim) ready for network input
        """
        features = self.feat_extractor.get_features(state)
        return torch.tensor(
            features,
            device=self.config.device,
            dtype=torch.float32,
        ).unsqueeze(0)  # Add batch dimension

    # ---------------------------------------------------------------------------- #
    #                           Policy Network Operations                          #
    # ---------------------------------------------------------------------------- #

    def compute_action_logits(self, states: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for all actions: π_θ(·|s) before softmax.

        Args:
            states: (batch_size, state_dim)

        Returns:
            logits: (batch_size, n_actions)
        """
        return self.policy_network(states)

    def compute_action_probabilities(self, states: torch.Tensor) -> torch.Tensor:
        """Compute action probabilities π_θ(a|s) = softmax(logits).

        Args:
            states: (batch_size, state_dim)

        Returns:
            probs: (batch_size, n_actions) where each row sums to 1
        """
        logits = self.compute_action_logits(states)
        return F.softmax(logits, dim=-1)

    def compute_action_log_probabilities(self, states: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities log π_θ(a|s) = log_softmax(logits).

        More numerically stable than log(softmax(logits)).

        Args:
            states: (batch_size, state_dim)

        Returns:
            log_probs: (batch_size, n_actions)
        """
        logits = self.compute_action_logits(states)
        return F.log_softmax(logits, dim=-1)

    def select_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy π_θ(·|s).

        Action selection: a_t ~ π_θ(·|s_t)

        This is stochastic! Unlike DQN's argmax, we sample from the distribution.
        Exploration is inherent in the stochastic policy.

        Args:
            state: (1, state_dim)

        Returns:
            action: (1, 1) sampled action index
            log_prob: (1,) log probability log π_θ(a_t|s_t)
        """
        with torch.no_grad():
            probs = self.compute_action_probabilities(state)  # (1, n_actions)

            # Sample from categorical distribution
            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample()  # (1,)
            log_prob = distribution.log_prob(action)  # (1,)

            return action.unsqueeze(1), log_prob  # (1, 1), (1,)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log π_θ(a|s) and entropy H(π_θ(·|s)) for given state-action pairs.

        Used during learning to compute:
        - Log probabilities of actions taken
        - Policy entropy (for exploration bonus)

        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, 1)

        Returns:
            log_probs: (batch_size,) log π_θ(a|s) for each (s,a) pair
            entropy: (batch_size,) entropy H(π_θ(·|s)) = -Σ_a π(a|s)log(π(a|s))
        """
        log_probs_all = self.compute_action_log_probabilities(states)  # (B, A)
        probs = torch.exp(log_probs_all)  # (B, A)

        # Extract log prob for taken actions
        log_probs = log_probs_all.gather(1, actions).squeeze(1)  # (B,)

        # Compute entropy: H = -Σ_a π(a|s) log π(a|s)
        entropy = -(probs * log_probs_all).sum(dim=-1)  # (B,)

        return log_probs, entropy

    # ---------------------------------------------------------------------------- #
    #                           Value Network Operations                           #
    # ---------------------------------------------------------------------------- #

    def compute_values(self, states: torch.Tensor) -> torch.Tensor:
        """Compute state values V_φ(s).

        Args:
            states: (batch_size, state_dim)

        Returns:
            values: (batch_size,) estimated values V_φ(s)
        """
        return self.value_network(states).squeeze(-1)  # (B, 1) → (B,)

    # ---------------------------------------------------------------------------- #
    #                        Advantage Estimation (GAE)                            #
    # ---------------------------------------------------------------------------- #

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        GAE balances bias-variance tradeoff in advantage estimation:

        TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)

        GAE(λ): A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
               = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        Special cases:
        - λ=0: A_t = δ_t (TD advantage, high bias, low variance)
        - λ=1: A_t = Σ_l r_{t+l} - V(s_t) (MC advantage, low bias, high variance)

        Args:
            rewards: (T,) rewards r_t
            values: (T,) value estimates V(s_t)
            dones: (T,) episode termination flags
            next_value: scalar, V(s_{T+1}) for bootstrapping

        Returns:
            advantages: (T,) advantage estimates A_t
            returns: (T,) return estimates R_t = A_t + V(s_t)
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.config.device)

        # Compute advantages backwards from T-1 to 0
        last_gae = 0.0

        not_dones = ~dones

        for t in reversed(range(T)):
            next_value_t = next_value if t == T - 1 else values[t + 1]

            # Mask next value if episode ended
            next_value_t *= not_dones[t]

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t - values[t]

            # GAE recursive formula: A_t = δ_t + (γλ) * A_{t+1}
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * (not_dones[t]) * last_gae
            )

        # Returns: R_t = A_t + V(s_t)
        # This is the target for value function learning
        returns = advantages + values

        return advantages, returns

    # ---------------------------------------------------------------------------- #
    #                              PPO Loss Functions                              #
    # ---------------------------------------------------------------------------- #

    def compute_probability_ratio(
        self,
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor,
    ) -> torch.Tensor:
        """Compute probability ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t).

        The ratio measures how much the policy has changed for the taken actions.
        - r_t > 1: new policy assigns higher probability
        - r_t < 1: new policy assigns lower probability
        - r_t = 1: no change

        Args:
            log_probs_new: (batch_size,) log π_θ(a|s)
            log_probs_old: (batch_size,) log π_θ_old(a|s)

        Returns:
            ratio: (batch_size,) r_t = exp(log π_new - log π_old)
        """
        return torch.exp(log_probs_new - log_probs_old)

    def compute_clipped_surrogate_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute PPO clipped surrogate objective L^CLIP(θ).

        Standard policy gradient: L^PG = E[A_t * log π_θ(a_t|s_t)]

        PPO clips the ratio to prevent too large updates:
        L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

        where r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

        Intuition:
        - If A_t > 0 (good action): we want to increase π(a|s), but not too much
        - If A_t < 0 (bad action): we want to decrease π(a|s), but not too much
        - Clipping prevents destructively large updates

        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, 1)
            advantages: (batch_size,) estimated advantages A_t
            log_probs_old: (batch_size,) old log probabilities log π_θ_old(a_t|s_t)

        Returns:
            loss: scalar, -L^CLIP (negated for gradient ascent → descent)
            entropy: scalar, mean entropy H(π_θ) for exploration bonus
        """
        # Evaluate actions under current policy
        log_probs_new, entropy = self.evaluate_actions(states, actions)

        # Compute probability ratio r_t = π_new / π_old
        ratio = self.compute_probability_ratio(log_probs_new, log_probs_old)

        # Unclipped objective: r_t * A_t
        surrogate1 = ratio * advantages

        # Clipped objective: clip(r_t, 1-ε, 1+ε) * A_t
        ratio_clipped = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )
        surrogate2 = ratio_clipped * advantages

        # PPO objective: take minimum (pessimistic bound)
        # We negate because PyTorch minimizes, but we want to maximize
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        entropy_mean = entropy.mean()

        return policy_loss, entropy_mean

    def compute_value_loss(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value function loss L^VF(φ).

        Mean squared error between predicted values and target returns:
        L^VF = E[(V_φ(s_t) - R_t)²]

        where R_t = A_t + V(s_t) is the empirical return (GAE target).

        This is similar to DQN's TD error, but for the value function.

        Args:
            states: (batch_size, state_dim)
            returns: (batch_size,) target returns R_t

        Returns:
            loss: scalar MSE loss
        """
        values_pred = self.compute_values(states)
        return F.mse_loss(values_pred, returns)

    def compute_kl_divergence(
        self,
        states: torch.Tensor,
        log_probs_old: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence D_KL(π_θ_old || π_θ).

        KL divergence measures how much the policy has changed:
        D_KL(π_old || π_new) = E_{a~π_old}[log π_old(a|s) - log π_new(a|s)]
                              = E[log(π_old/π_new)]

        For discrete distributions:
        D_KL = Σ_a π_old(a|s) * log(π_old(a|s) / π_new(a|s))

        We use this to adapt the KL penalty coefficient β:
        - If D_KL too large: increase β (penalize changes more)
        - If D_KL too small: decrease β (allow more changes)

        Args:
            states: (batch_size, state_dim)
            log_probs_old: (batch_size, n_actions) old log probabilities

        Returns:
            kl: scalar, mean KL divergence
        """
        log_probs_new = self.compute_action_log_probabilities(states)
        probs_old = torch.exp(log_probs_old)

        # D_KL = Σ_a π_old(a|s) * (log π_old(a|s) - log π_new(a|s))
        kl = (probs_old * (log_probs_old - log_probs_new)).sum(dim=-1).mean()

        return kl

    def compute_total_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_all_old: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total PPO loss with all components.

        Total loss:
        L = L^CLIP - c₂·H(π) + c₁·L^VF - β·D_KL

        where:
        - L^CLIP: clipped policy loss (maximize)
        - H(π): entropy bonus (maximize for exploration)
        - L^VF: value function loss (minimize)
        - D_KL: KL penalty (minimize to keep policy stable)

        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, 1)
            advantages: (batch_size,)
            returns: (batch_size,)
            log_probs_old: (batch_size,) log π_old for taken actions
            log_probs_all_old: (batch_size, n_actions) log π_old for all actions

        Returns:
            loss: scalar total loss
            info: dict with loss components for logging
        """
        # Policy loss with entropy
        policy_loss, entropy = self.compute_clipped_surrogate_loss(
            states,
            actions,
            advantages,
            log_probs_old,
        )

        # Value loss
        value_loss = self.compute_value_loss(states, returns)

        # KL divergence
        kl = self.compute_kl_divergence(states, log_probs_all_old)

        # Total loss: policy + value - entropy_bonus + kl_penalty
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
            + self.beta * kl
        )

        info = {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/entropy": entropy.item(),
            "loss/kl": kl.item(),
            "beta": self.beta,
        }

        return total_loss, info

    # ---------------------------------------------------------------------------- #
    #                              Training Loop                                   #
    # ---------------------------------------------------------------------------- #

    def collect_trajectory(self) -> Trajectory:
        """Collect one episode trajectory using current policy π_θ.

        Execute episode and store (s_t, a_t, r_t, log π(a_t|s_t), V(s_t), done_t).

        Returns:
            trajectory: Trajectory object with full episode data
        """
        trajectory = Trajectory()

        state, _info = self.env.reset_train()
        state = self._preprocess_state(state)

        for t in count():
            self.steps_done += 1

            # Select action from policy
            with torch.no_grad():
                action, log_prob = self.select_action(state)
                value = self.compute_values(state)

            # Execute action
            observation, reward, terminated, truncated, _info = self.env.step(
                action.item(),
            )
            done = terminated or truncated

            # Store transition
            trajectory.add(
                state=state,
                action=action,
                reward=float(reward),
                log_prob=log_prob,
                value=value.view(-1),
                done=done,
            )

            if done or t >= self.config.max_length_train:
                break

            # Next state
            state = self._preprocess_state(observation)

        return trajectory

    def collect_trajectories(self, num_steps: int) -> list[Trajectory]:
        """Collect multiple trajectories until we have at least num_steps.

        Args:
            num_steps: minimum number of timesteps to collect

        Returns:
            trajectories: list of Trajectory objects
        """
        trajectories = []
        total_steps = 0

        while total_steps < num_steps:
            trajectory = self.collect_trajectory()
            trajectories.append(trajectory)
            total_steps += len(trajectory)
            self.episodes_done += 1

        return trajectories

    def process_trajectories(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[torch.Tensor, ...]:
        """Process collected trajectories for learning.

        Steps:
        1. Convert trajectory lists to tensors
        2. Compute advantages using GAE
        3. Normalize advantages (reduces variance)
        4. Compute old log probabilities for all actions (for KL)

        Args:
            trajectories: list of Trajectory objects

        Returns:
            states: (total_steps, state_dim)
            actions: (total_steps, 1)
            advantages: (total_steps,) normalized advantages
            returns: (total_steps,) target returns
            log_probs_old: (total_steps,) old log probs for taken actions
            log_probs_all_old: (total_steps, n_actions) old log probs for all actions
        """
        all_states = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_log_probs = []
        all_log_probs_all = []

        for trajectory in trajectories:
            states, actions, rewards, log_probs, values, dones = trajectory.to_tensors(
                torch.device(self.config.device),
            )

            # Compute next value for bootstrapping
            # If episode ended naturally, next_value = 0
            # If truncated, next_value = V(last_state)
            if dones[-1]:
                next_value = torch.tensor(0.0, device=self.config.device)
            else:
                with torch.no_grad():
                    # Get the last state (after the episode ended)
                    next_value = values[-1]  # Use last value as approximation

            # Compute advantages and returns using GAE
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)

            # Store old log probabilities for all actions (needed for KL)
            with torch.no_grad():
                log_probs_all = self.compute_action_log_probabilities(states)

            all_states.append(states)
            all_actions.append(actions)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_log_probs.append(log_probs)
            all_log_probs_all.append(log_probs_all)

        # Concatenate all trajectories
        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)
        log_probs_old = torch.cat(all_log_probs, dim=0)
        log_probs_all_old = torch.cat(all_log_probs_all, dim=0)

        # Normalize advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, advantages, returns, log_probs_old, log_probs_all_old

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_all_old: torch.Tensor,
    ):
        """Update policy and value networks using PPO.

        Performs K epochs of minibatch SGD on the collected data.

        Args:
            states: (N, state_dim) all states
            actions: (N, 1) all actions
            advantages: (N,) all advantages
            returns: (N,) all returns
            log_probs_old: (N,) old log probs
            log_probs_all_old: (N, n_actions) old log probs for all actions
        """
        N = len(states)
        info = None
        # Multiple epochs on same data
        for epoch in range(self.num_epochs):
            # Shuffle data
            indices = torch.randperm(N)

            # Minibatch SGD
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_indices = indices[start:end]

                # Extract minibatch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_log_probs_all_old = log_probs_all_old[batch_indices]

                # Compute loss
                loss, info = self.compute_total_loss(
                    batch_states,
                    batch_actions,
                    batch_advantages,
                    batch_returns,
                    batch_log_probs_old,
                    batch_log_probs_all_old,
                )

                # Update both networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (optional)
                if self.config.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_network.parameters(),
                        self.config.clip_grad_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.value_network.parameters(),
                        self.config.clip_grad_norm,
                    )

                self.policy_optimizer.step()
                self.value_optimizer.step()

            # Log metrics (last minibatch of last epoch)
            if info is None:
                msg = "No info collected during policy update."
                raise ValueError(msg)

            if epoch == self.num_epochs - 1:
                for key, value in info.items():
                    self.logger[key] = value

        if info is None:
            msg = "No info collected during policy update."
            raise ValueError(msg)

        # Adapt KL coefficient β based on observed KL divergence
        self.adapt_kl_coefficient(info["loss/kl"])

    def adapt_kl_coefficient(self, kl: float):
        """Adapt KL penalty coefficient β based on observed KL divergence.

        From Algorithm 1:
        - If D_KL(π_old || π_new) ≥ 1.5δ: β ← 2β (increase penalty)
        - If D_KL(π_old || π_new) ≤ δ/1.5: β ← 0.5β (decrease penalty)

        This adapts the strength of the KL constraint based on how much
        the policy is changing.

        Args:
            kl: observed KL divergence from last update
        """
        if kl >= 1.5 * self.target_kl:
            self.beta = min(self.beta * 2.0, 10.0)  # Cap maximum β
        elif kl <= self.target_kl / 1.5:
            self.beta = max(self.beta * 0.5, 0.001)  # Cap minimum β

    def learn(self, num_steps: int):
        """Perform one PPO update cycle.

        Steps:
        1. Collect trajectories using current policy π_θ
        2. Compute advantages using GAE
        3. Update policy and value networks for K epochs
        4. Adapt KL coefficient β

        Args:
            num_steps: number of environment steps to collect before update
        """
        # Step 1: Collect trajectories
        trajectories = self.collect_trajectories(num_steps)

        # Compute total reward for logging
        total_rewards = [sum(traj.rewards) for traj in trajectories]
        mean_reward = statistics.fmean(total_rewards)
        self.logger["train/reward"] = mean_reward
        self.logger["train/episodes"] = len(trajectories)

        # Step 2: Process trajectories (compute advantages, returns)
        (
            states,
            actions,
            advantages,
            returns,
            log_probs_old,
            log_probs_all_old,
        ) = self.process_trajectories(trajectories)

        # Step 3: Update networks
        self.update_policy(
            states,
            actions,
            advantages,
            returns,
            log_probs_old,
            log_probs_all_old,
        )

        self.updates_done += 1

        # Log
        verbose = self.config.freq_verbose_train > 0 and (
            self.updates_done % self.config.freq_verbose_train == 0
        )
        self.logger.log(self.updates_done, verbose=verbose)

    # ---------------------------------------------------------------------------- #
    #                                  Evaluation                                  #
    # ---------------------------------------------------------------------------- #

    def evaluate_episode(self, *, render: bool = False) -> float:
        """Run one episode with greedy policy (no exploration).

        For evaluation, we take the most likely action (argmax) instead of sampling.

        Returns:
            total_reward: reward accumulated during the episode
        """
        self.policy_network.eval()
        self.value_network.eval()

        if render:
            state, _info = self.env.reset_human()
        else:
            state, _info = self.env.reset_test()

        state = self._preprocess_state(state)
        total_reward = 0.0

        while True:
            # Greedy action selection: argmax π_θ(·|s)
            with torch.no_grad():
                probs = self.compute_action_probabilities(state)
                action = probs.argmax(dim=-1, keepdim=True)

            observation, reward, terminated, truncated, _info = self.env.step(
                action.item(),
            )
            done = terminated or truncated
            total_reward += float(reward)

            if done:
                break

            state = self._preprocess_state(observation)

        self.policy_network.train()
        self.value_network.train()

        return total_reward

    def test_agent(self, num_episodes: int) -> float:
        """Test agent for multiple episodes and log mean reward.

        Args:
            num_episodes: number of test episodes

        Returns:
            mean_reward: average reward over test episodes
        """
        test_rewards = [self.evaluate_episode() for _ in range(num_episodes)]
        mean_reward = statistics.fmean(test_rewards)

        self.logger.direct_log(
            "test/reward_mean",
            mean_reward,
            self.updates_done,
            verbose=True,
        )

        return mean_reward

    # ---------------------------------------------------------------------------- #
    #                                  Persistence                                 #
    # ---------------------------------------------------------------------------- #

    def save_model(self, path: Path | None = None):
        """Save policy and value network parameters.

        Args:
            path: directory to save models (defaults to self.outdir)
        """
        if path is None:
            path = self.outdir

        torch.save(self.policy_network.state_dict(), path / "policy_network.pth")
        torch.save(self.value_network.state_dict(), path / "value_network.pth")

        return self

    def load_model(self, path: Path | None = None):
        """Load policy and value network parameters.

        Args:
            path: directory containing saved models (defaults to self.outdir)
        """
        if path is None:
            path = self.outdir

        policy_path = path / "policy_network.pth"
        value_path = path / "value_network.pth"

        if not policy_path.exists():
            raise FileNotFoundError(f"No policy model found at {policy_path}")
        if not value_path.exists():
            raise FileNotFoundError(f"No value model found at {value_path}")

        self.policy_network.load_state_dict(
            torch.load(
                policy_path,
                weights_only=True,
                map_location=self.config.device_torch,
            ),
        )
        self.value_network.load_state_dict(
            torch.load(
                value_path,
                weights_only=True,
                map_location=self.config.device_torch,
            ),
        )

        return self

    @classmethod
    def from_model_path(cls, path: Path):
        """Load agent from saved model directory.

        Args:
            path: path to model file or directory containing config.toml

        Returns:
            agent: loaded PPOAgent
        """
        if path.is_file():
            path = path.parent

        setup_env = setup.init_env(
            config_file=path / "config.toml",
            name="loaded_agent",
            launch_tensorboard=False,
            kind=ConfigPPO,
        )
        agent = cls(setup_env)
        agent.load_model(path)
        return agent

    # ---------------------------------------------------------------------------- #
    #                                   Main Loop                                  #
    # ---------------------------------------------------------------------------- #

    def run(self):
        """Main training loop with periodic testing.

        Training loop:
        1. Collect trajectories and update policy
        2. Periodically evaluate on test episodes
        3. Save model checkpoints
        4. Stop when performance threshold reached
        """
        num_updates = self.config.num_updates
        test_interval = self.config.freq_test
        steps_per_update = self.config.steps_per_update

        for update in range(num_updates):
            # Training update
            self.learn(steps_per_update)

            # Periodic testing
            if (update + 1) % test_interval == 0:
                test_reward = self.test_agent(self.config.nb_tests)

                # Early stopping
                stop_threshold = self.config.stop_when_test_higher_than
                if test_reward >= stop_threshold:
                    print(
                        f"Stopping training as test reward {test_reward:.2f} "
                        f"exceeds threshold {stop_threshold}.",
                    )
                    break

            # Periodic saving
            if (update + 1) % self.config.freq_save == 0:
                self.save_model()

        self.save_model()
        print(f"Training complete. Model saved at {self.outdir}.")
        return self
