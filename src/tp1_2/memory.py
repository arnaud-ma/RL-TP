from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch


class Transition(NamedTuple):
    state: torch.Tensor  # shape (nb_features,)
    action: torch.Tensor  # int of shape (1,)
    reward: torch.Tensor  # float of shape (1,)
    next_state: torch.Tensor  # shape (nb_features,)
    done: torch.Tensor  # shape (1,) (boolean)


class TransposedTransitionTensor(NamedTuple):
    states: torch.Tensor  # shape (batch_size, nb_features)
    actions: torch.Tensor  # shape (batch_size, 1)
    rewards: torch.Tensor  # shape (batch_size, 1)
    next_states: torch.Tensor  # shape (batch_size, nb_features)
    dones: torch.Tensor  # shape (batch_size,) (boolean mask)

    @classmethod
    def from_list_of_transitions(
        cls,
        transitions: list[Transition],
        device: str | int | torch.device,
    ) -> TransposedTransitionTensor:
        """
        Create a TransposedTransitionTensor from a list of Transition objects.

        This class method takes a list of individual transitions and combines them into
        a single batched tensor representation, with all tensors moved to the specified device.

        Args:
            transitions (list[Transition]): A list of Transition objects to be combined.
            device (str | int | torch.device): The device to move the tensors to. Can be
                a string like 'cpu' or 'cuda', an integer representing GPU index, or a
                torch.device object.

        Returns:
            TransposedTransitionTensor: A batched tensor representation of all transitions,
                with each field concatenated along the batch dimension.

        Example:
            >>> transition1 = Transition(state=torch.tensor([1.0]), action=torch.tensor([0]))
            >>> transition2 = Transition(state=torch.tensor([2.0]), action=torch.tensor([1]))
            >>> transitions = [transition1, transition2]
            >>> batched = TransposedTransitionTensor.from_list_of_transitions(
            ...     transitions, device='cpu'
            ... )
            >>> batched.state.shape
            torch.Size([2, 1])
        """
        # zip([a, b, c], [d, e, f]) -> [(a, d), (b, e), (c, f)]
        # so in our case, we get:
        # zip([[state1, action1, reward1, next_state1, done1],
        #      [state2, action2, reward2, next_state2, done2],
        #      ...])
        # -> [(state1, state2, ...),
        #     (action1, action2, ...),
        #     (reward1, reward2, ...),
        #     (next_state1, next_state2, ...),
        #     (done1, done2, ...)]
        # and then we concatenate each of these tuples, moving to the specified device,
        # to have [states, actions, rewards, next_states, dones]
        return cls(
            *(torch.cat(x).to(device) for x in zip(*transitions, strict=True)),
        )


class SumTree:
    """Binary tree data structure where the parent's value is the sum
    of its children."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data: list[Transition | None] = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def update(self, idx, p):
        """Update priority of a leaf node and propagate changes up the tree."""
        tree_idx = idx + self.capacity - 1
        diff = p - self.tree[tree_idx]
        self.tree[tree_idx] += diff

        # propagate change up to the root
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def add(self, priority: float, data: Transition):
        """Add a new data point with given priority."""
        idx = self.write_idx
        self.data[idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def get_next_idx(self):
        return self.write_idx

    def sample(self, value) -> tuple[int, float, Transition]:
        """Sample a leaf node based on the given value."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value < self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1

        data_idx = idx - (self.capacity - 1)
        current_data = self.data[data_idx]
        if current_data is None:
            msg = "Sampled data is None"
            raise ValueError(msg)
        return data_idx, self.tree[idx], current_data

    @property
    def total_priority(self) -> np.float64:
        """Sum of all priorities."""
        return self.tree[0]

    @property
    def max_priority(self):
        """Maximum priority among leaf nodes."""
        return np.max(self.tree[-self.capacity :])

    @property
    def min_priority(self):
        """Minimum priority among leaf nodes."""
        return np.min(self.tree[-self.capacity :])


class ReplayMemory:
    """Replay buffer with optional prioritized experience replay, for DQN agents with
    importance-sampling weights."""

    def __init__(
        self,
        capacity: int,
        p_upper: float = 1.0,
        epsilon: float = 0.01,
        alpha: float = 1,
        beta: float = 1,
        *,
        prioritized: bool = True,
    ):
        """Initialize the replay memory.

        Args:
            capacity (int): Maximum number of transitions to store.
            prioritized (bool): Whether to use prioritized experience replay.
            p_upper (int): Maximum priority value to avoid excessively large priorities.
            epsilon (float): Small value to ensure non-zero priorities.
            alpha (float): Priority exponent
                (0=uniform, 1=full prioritization).
            beta (float): Importance-sampling exponent
                (0=no corrections, 1=full correction).
        """
        self.capacity = capacity
        self.p_upper = p_upper
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.prioritized = prioritized
        self.nb_entities = 0
        # self.dict={}
        # self.data_len = 2 * feature_size + 2
        if prioritized:
            self.tree = SumTree(self.capacity)
            self.data_tree = [None] * capacity
        else:
            self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> int:
        """Save a transition with maximum priority (if prioritized) or append to memory.

        Returns:
            Index where the transition was stored.
        """

        if self.prioritized:
            max_priority = self.tree.max_priority
            if max_priority == 0:
                max_priority = self.p_upper
            return self.tree.add(max_priority, transition)

        self.memory.append(transition)
        return len(self.memory) - 1

    def sample(
        self,
        batch_size,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32] | None, list[Transition]]:
        """Sample a batch of transitions.
        Args:
            batch_size (int): Number of transitions to sample.
        Returns:
            indices (np.ndarray): Indices of sampled transitions.
            weights (np.ndarray | None): Importance-sampling weights
                (None if not prioritized).
            transitions (list[Transition]): Sampled transitions.
        """
        if not self.prioritized:
            # just uniformly sample without replacement
            indices = np.random.choice(len(self), batch_size, replace=False)  # noqa: NPY002
            transitions = [self.memory[idx] for idx in indices]
            return indices, None, transitions

        # prioritized sampling:
        # - sample based on priorities
        # - compute importance-sampling weights

        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        transitions = []

        # divide the total priority into segments
        segment_size = self.tree.total_priority / batch_size

        # calculate minimum priority for normalization
        min_priority = self.tree.min_priority
        if min_priority == 0:
            min_priority = self.epsilon**self.alpha
        max_weights = (min_priority / self.tree.total_priority) ** (-self.beta)

        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)
            value = random.uniform(a, b)

            idx, priority, data = self.tree.sample(value)
            indices[i] = idx
            transitions.append(data)

            # calculate importance-sampling weight
            sampling_prob = priority / self.tree.total_priority
            weights[i] = (sampling_prob) ** (-self.beta)

        # normalize weights
        weights /= max_weights
        return indices, weights, transitions

    def update_priorities(
        self,
        indices: npt.NDArray[np.int32],
        td_errors: npt.NDArray[np.float32],
    ):
        """Update priorities of sampled transitions based on their TD errors."""
        if not self.prioritized:
            return
        # p = (|td_error| + epsilon)^alpha
        priorities = np.abs(td_errors) + self.epsilon
        priorities = np.minimum(priorities, self.p_upper)
        priorities = np.power(priorities, self.alpha)
        for idx, p in zip(indices, priorities, strict=True):
            self.tree.update(idx, p)

    @property
    def next_idx(self):
        return (
            self.tree.get_next_idx()
            if self.prioritized
            else len(self.memory) % self.capacity
        )

    def get_data(self, idx: int) -> Transition:
        if idx >= len(self):
            msg = "Index out of range"
            raise IndexError(msg)
        if self.prioritized:
            current_data = self.tree.data[idx]
            if current_data is None:
                msg = "Retrieved data is None"
                raise ValueError(msg)
            return current_data
        return self.memory[idx]

    def __getitem__(self, idx: int) -> Transition:
        return self.get_data(idx)

    def __len__(self):
        return self.tree.n_entries if self.prioritized else len(self.memory)
