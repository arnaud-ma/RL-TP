# from __future__ import annotations

# import statistics
# from typing import TYPE_CHECKING, NamedTuple

# import torch
# import torch.nn.functional as F

# from tp1_2 import setup
# from tp1_2.config import ConfigPPO
# from tp1_2.network import NeuralNetwork

# if TYPE_CHECKING:
#     from pathlib import Path

#     from tp1_2.setup import SetupEnv


# class Transition(NamedTuple):
#     """Represents a single step in a PPO trajectory."""

#     state: torch.Tensor
#     action: torch.Tensor
#     reward: torch.Tensor
#     log_prob: torch.Tensor  # log π(a|s)
#     value: torch.Tensor  # V(s)
#     done: torch.Tensor


# class TrajectoryBatch(NamedTuple):
#     """Represents a batch of transitions for PPO update."""

#     states: torch.Tensor
#     actions: torch.Tensor
#     rewards: torch.Tensor
#     log_probs: torch.Tensor
#     values: torch.Tensor
#     dones: torch.Tensor

#     # Computed fields for update
#     returns: torch.Tensor | None = None
#     advantages: torch.Tensor | None = None

#     @classmethod
#     def from_list_of_transitions(
#         cls,
#         transitions: list[Transition],
#         device: torch.device,
#     ) -> TrajectoryBatch:
#         """Convert a list of transitions into a batched tensor object."""
#         batch = cls(
#             *(torch.cat(x).to(device) for x in zip(*transitions, strict=True)),
#         )
#         return batch

#     def with_returns_and_advantages(
#         self,
#         returns: torch.Tensor,
#         advantages: torch.Tensor,
#     ) -> TrajectoryBatch:
#         """Return a new batch with computed returns and advantages."""
#         return TrajectoryBatch(
#             states=self.states,
#             actions=self.actions,
#             rewards=self.rewards,
#             log_probs=self.log_probs,
#             values=self.values,
#             dones=self.dones,
#             returns=returns,
#             advantages=advantages,
#         )


# class PPOAgent:
#     """
#     Implementation of Algorithm 1: PPO with Adaptive KL Penalty.
#     """

#     def __init__(self, setup_env: SetupEnv[ConfigPPO]):
#         self.env, self.config, self.outdir, self.logger = setup_env
#         self._init_hyperparameters()
#         self._init_networks()
#         self._init_optimizers()
#         self._init_feature_extractor()
#         self._init_counters()

#     def _init_hyperparameters(self):
#         # Line 1: β0 ← 1
#         self.beta = 1.0

#         # Input: KL cible δ (target_kl)
#         self.target_kl = self.config.target_kl
#         # Input: pas d'apprentissage α (policy_lr)
#         self.policy_lr = self.config.policy_lr
#         self.value_lr = self.config.value_lr
#         # Input: nombre d'étapes d'optimisation K (num_epochs)
#         self.K_epochs = self.config.num_epochs

#         self.gamma = self.config.gamma
#         self.gae_lambda = self.config.gae_lambda
#         self.batch_size = self.config.batch_size

#     def _init_networks(self):
#         # Input: Paramètres initiaux θ0 et φ
#         self.n_actions = self.env.action_space.n  # type: ignore[reportAttributeAccessIssue]
#         state, _ = self.env.reset()
#         self.n_observations = len(state)

#         # Policy Network πθ
#         self.policy_network = NeuralNetwork(
#             input_size=self.n_observations,
#             output_size=self.n_actions,
#             layers=self.config.policy_hidden_layers,
#             activation=self.config.policy_activation,
#             final_activation=torch.nn.Identity,  # Output logits
#             dropout=self.config.policy_dropout,
#         ).to(self.config.device)

#         # Value Network Vφ
#         self.value_network = NeuralNetwork(
#             input_size=self.n_observations,
#             output_size=1,
#             layers=self.config.value_hidden_layers,
#             activation=self.config.value_activation,
#             final_activation=torch.nn.Identity,  # Output scalar value
#             dropout=self.config.value_dropout,
#         ).to(self.config.device)

#     def _init_optimizers(self):
#         self.policy_optimizer = torch.optim.Adam(
#             self.policy_network.parameters(),
#             lr=self.policy_lr,
#         )
#         self.value_optimizer = torch.optim.Adam(
#             self.value_network.parameters(),
#             lr=self.value_lr,
#         )

#     def _init_feature_extractor(self):
#         self.feat_extractor = self.config.feature_extractor(self.env)

#     def _init_counters(self):
#         self.updates_done = 0

#     def _preprocess_state(self, state) -> torch.Tensor:
#         features = self.feat_extractor.get_features(state)
#         return torch.tensor(
#             features,
#             device=self.config.device,
#             dtype=torch.float32,
#         ).unsqueeze(0)

#     # ---------------------------------------------------------------------------------
#     #                                     Algorithm Steps
#     # ---------------------------------------------------------------------------------

#     def collect_trajectories(self, num_steps: int) -> TrajectoryBatch:
#         """
#         - Collecte d'un ensemble de trajectoires Dk selon politique πθk.
#         - Calcul des avantages A_hat pour toutes les transitions de Dk selon TD(λ).
#         """
#         all_transitions: list[Transition] = []
#         total_steps = 0

#         while total_steps < num_steps:
#             state_tensor = self._preprocess_state(self.env.reset_train()[0])
#             episode_transitions: list[Transition] = []

#             for _ in range(self.config.max_length_train):
#                 with torch.no_grad():
#                     logits = self.policy_network(state_tensor)
#                     probs = F.softmax(logits, dim=-1)
#                     dist = torch.distributions.Categorical(probs)
#                     action = dist.sample()
#                     log_prob = dist.log_prob(action)
#                     value = self.value_network(state_tensor)

#                 next_obs, reward, terminated, truncated, _ = self.env.step(
#                     action.item(),
#                 )
#                 done = terminated or truncated

#                 # Create transition object
#                 transition = Transition(
#                     state=state_tensor,
#                     action=action.unsqueeze(0),
#                     reward=torch.tensor([reward], dtype=torch.float32),
#                     log_prob=log_prob.unsqueeze(0),
#                     value=value.view(-1),
#                     done=torch.tensor([done], dtype=torch.bool),
#                 )
#                 episode_transitions.append(transition)

#                 state_tensor = self._preprocess_state(next_obs)
#                 total_steps += 1
#                 if done:
#                     break

#             # Bootstrap value for GAE
#             if not done:  # truncated
#                 with torch.no_grad():
#                     next_value = self.value_network(state_tensor).item()
#             else:
#                 next_value = 0.0

#             all_transitions.extend(episode_transitions)

#             # We calculate advantages later on the full batch, but we need next_values
#             # stored correctly or handle episodes individually.
#             # Simplified: we just collect raw transitions first.

#         # Convert list of transitions to batch (on device)
#         batch = TrajectoryBatch.from_list_of_transitions(
#             all_transitions,
#             self.config.device_torch,
#         )

#         # Line 4: Update Advantage estimates (GAE/TD(λ))
#         # Note: Ideally GAE is computed per episode. For simplicity here we treat the
#         # collected batch as a stream, but proper implementation requires
#         # episode boundaries.
#         # Since 'batch' concatenates everything, we'll re-calculate GAE using
#         # the dones flag.
#         returns, advantages = self._compute_gae_vectorized(
#             batch.rewards,
#             batch.values,
#             batch.dones,
#             next_value=0.0,  # Approximation: last collected step is usually terminal or truncated
#         )

#         # Normalize advantages
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#         return batch.with_returns_and_advantages(returns, advantages)

#     def _compute_gae_vectorized(
#         self,
#         rewards: torch.Tensor,
#         values: torch.Tensor,
#         dones: torch.Tensor,
#         next_value: float,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Compute GAE for the entire batch vectorized."""
#         advantages = torch.zeros_like(rewards)
#         gae = 0.0
#         # Iterate backwards
#         for t in reversed(range(len(rewards))):
#             next_val = next_value if t == len(rewards) - 1 else values[t + 1]

#             # If done at t, next_val should differ (usually 0 unless truncated)
#             # here we use 'not done' mask.
#             mask = ~dones[t]

#             delta = rewards[t] + self.gamma * next_val * mask - values[t]
#             gae = delta + self.gamma * self.gae_lambda * mask * gae
#             advantages[t] = gae

#         returns = advantages + values
#         return returns, advantages

#     def update_policy(self, Dk: TrajectoryBatch) -> float:
#         """
#         Line 6: for s de 1 à K do
#         Line 7: θ ← θ + α∇(L - β*KL)
#         Line 8: L = ratio * Advantage
#         Line 9: KL approx
#         """
#         assert Dk.advantages is not None
#         assert Dk.returns is not None

#         dataset_size = len(Dk.states)
#         indices = torch.arange(dataset_size)
#         total_kl = 0.0

#         for _ in range(self.K_epochs):  # Line 6
#             indices = indices[torch.randperm(dataset_size)]

#             for start in range(0, dataset_size, self.batch_size):
#                 idx = indices[start : start + self.batch_size]
#                 states = Dk.states[idx]
#                 actions = Dk.actions[idx]
#                 advantages = Dk.advantages[idx]
#                 old_log_probs = Dk.log_probs[idx]

#                 # Current policy πθ
#                 logits = self.policy_network(states)
#                 dist = torch.distributions.Categorical(logits=logits)
#                 new_log_probs = dist.log_prob(actions.squeeze())

#                 # Line 8: Lθk(θ)
#                 # ratio = πθ(a|s) / πθk(a|s)
#                 ratio = torch.exp(new_log_probs - old_log_probs.squeeze())
#                 L_theta = (ratio * advantages).mean()

#                 # Line 9: KL(θk | θ)
#                 # Approx: DKL ≈ E[log πθk - log πθ]
#                 kl_div = (old_log_probs.squeeze() - new_log_probs).mean()

#                 # Line 7: Objective = L - β * KL
#                 loss = -(L_theta - self.beta * kl_div)

#                 self.policy_optimizer.zero_grad()
#                 loss.backward()
#                 self.policy_optimizer.step()

#                 total_kl += kl_div.item()

#         return total_kl / (self.K_epochs * (dataset_size / self.batch_size))

#     def update_kl_parameter(self, kl_div: float):
#         """Line 12-17: Adaptation du paramètre β sur base de KL."""
#         if kl_div >= 1.5 * self.target_kl:
#             self.beta = min(self.beta * 2.0, 10.0)
#         elif kl_div <= self.target_kl / 1.5:
#             self.beta = max(self.beta * 0.5, 0.001)

#     def update_value_function(self, Dk: TrajectoryBatch):
#         """
#         Line 18: Mise à jour de Vφ selon TD(λ) sur Dk.
#         (Minimizing MSE between Vφ(s) and Returns computed via GAE).
#         """
#         assert Dk.returns is not None
#         dataset_size = len(Dk.states)
#         indices = torch.arange(dataset_size)

#         for _ in range(self.K_epochs):
#             indices = indices[torch.randperm(dataset_size)]
#             for start in range(0, dataset_size, self.batch_size):
#                 idx = indices[start : start + self.batch_size]
#                 states = Dk.states[idx]
#                 target_values = Dk.returns[idx]

#                 current_values = self.value_network(states).squeeze()
#                 loss = F.mse_loss(current_values, target_values)

#                 self.value_optimizer.zero_grad()
#                 loss.backward()
#                 self.value_optimizer.step()

#     def step(self):
#         """Line 2: for k = 0, 1, 2... (Single iteration)."""
#         # Line 3 & 4
#         Dk = self.collect_trajectories(self.config.steps_per_update)

#         # Line 5 (Implicit: θk stored in Dk as old_log_probs) & Lines 6-10
#         avg_kl = self.update_policy(Dk)

#         # Line 11 (Implicit: θ updated in place) & Lines 12-17
#         self.update_kl_parameter(avg_kl)

#         # Line 18
#         self.update_value_function(Dk)

#         self.updates_done += 1

#         # Logging
#         if self.updates_done % self.config.freq_verbose_train == 0:
#             assert Dk.returns is not None
#             self.logger["train/reward"] = Dk.returns.mean().item()
#             self.logger["train/beta"] = self.beta
#             self.logger["train/kl"] = avg_kl
#             self.logger.log(self.updates_done, verbose=True)

#     # -------------------------------------------------------------------------------
#     #                                   Evaluation & Utilities
#     # -------------------------------------------------------------------------------

#     def evaluate_episode(self, *, render=False) -> float:
#         self.policy_network.eval()
#         state, _ = self.env.reset_human() if render else self.env.reset_test()
#         state = self._preprocess_state(state)
#         total_reward = 0.0

#         while True:
#             with torch.no_grad():
#                 logits = self.policy_network(state)
#                 action = logits.argmax(dim=-1)

#             obs, reward, terminated, truncated, _ = self.env.step(action.item())
#             total_reward += float(reward)
#             if terminated or truncated:
#                 break
#             state = self._preprocess_state(obs)

#         self.policy_network.train()
#         return total_reward

#     def test_agent(self, nb_episodes) -> float:
#         rewards = [self.evaluate_episode() for _ in range(nb_episodes)]
#         return statistics.mean(rewards)

#     def save_model(self, path=None):
#         path = path or self.outdir
#         torch.save(self.policy_network.state_dict(), path / "policy_network.pth")
#         torch.save(self.value_network.state_dict(), path / "value_network.pth")

#     def load_model(self, path=None):
#         path = path or self.outdir
#         self.policy_network.load_state_dict(
#             torch.load(
#                 path / "policy_network.pth",
#                 weights_only=True,
#                 map_location=self.config.device_torch,
#             ),
#         )
#         self.value_network.load_state_dict(
#             torch.load(
#                 path / "value_network.pth",
#                 weights_only=True,
#                 map_location=self.config.device_torch,
#             ),
#         )
#         return self

#     @classmethod
#     def from_model_path(cls, path: Path):
#         setup_env = setup.init_env(
#             path.parent / "config.toml",
#             "loaded_ppo",
#             launch_tensorboard=False,
#             kind=ConfigPPO,
#         )
#         agent = cls(setup_env)
#         agent.load_model(path)
#         return agent

#     def run(self):
#         """Main training loop."""
#         for update in range(self.config.num_updates):
#             self.step()
#             if (update + 1) % self.config.freq_test == 0:
#                 reward = self.test_agent(self.config.nb_tests)
#                 self.logger.direct_log(
#                     "test/reward",
#                     reward,
#                     self.updates_done,
#                     verbose=True,
#                 )
#                 if reward >= self.config.stop_when_test_higher_than:
#                     print(
#                         "Stopping: "
#                         f"{reward} >= {self.config.stop_when_test_higher_than}",
#                     )
#                     break
#         self.save_model()
#         print(f"Training complete. Model saved as {self.outdir}")
