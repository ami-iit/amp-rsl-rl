# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch
import numpy as np
from typing import Generator, Tuple


class ReplayBuffer:
    """
    Fixed-size circular buffer to store state and next-state experience tuples.

    Attributes:
        states (Tensor): Buffer of current states.
        next_states (Tensor): Buffer of next states.
        buffer_size (int): Maximum number of elements in the buffer.
        device (str): Device where tensors are stored.
        step (int): Current write index.
        num_samples (int): Total number of inserted samples (up to buffer_size).
    """

    def __init__(self, obs_dim: int, buffer_size: int, device: str) -> None:
        """
        Initialize a ReplayBuffer object.

        Args:
            obs_dim (int): Dimension of the observation space.
            buffer_size (int): Maximum number of transitions to store.
            device (str): Torch device where buffers are allocated ('cpu' or 'cuda').
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """
        Add a batch of states and next_states to the buffer.

        Args:
            states (Tensor): Batch of current states (batch_size, obs_dim).
            next_states (Tensor): Batch of next states (batch_size, obs_dim).
        """
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states

        if end_idx > self.buffer_size:
            # Wrap around the buffer end
            upper_len = self.buffer_size - self.step
            self.states[self.step : self.buffer_size] = states[:upper_len]
            self.next_states[self.step : self.buffer_size] = next_states[:upper_len]
            self.states[: end_idx - self.buffer_size] = states[upper_len:]
            self.next_states[: end_idx - self.buffer_size] = next_states[upper_len:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yield mini-batches of (state, next_state) tuples from the buffer.

        Args:
            num_mini_batch (int): Number of mini-batches to generate.
            mini_batch_size (int): Number of samples per mini-batch.

        Yields:
            Tuple[Tensor, Tensor]: A mini-batch of states and next_states.
        """
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(
                self.num_samples, size=mini_batch_size, replace=False
            )
            yield (
                self.states[sample_idxs].to(self.device),
                self.next_states[sample_idxs].to(self.device),
            )
