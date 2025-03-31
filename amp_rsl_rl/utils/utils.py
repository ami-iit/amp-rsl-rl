# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from typing import Tuple, Union

import torch
import numpy as np


class RunningMeanStd:
    """
    Calculates the running mean and standard deviation of a data stream.
    Based on the parallel algorithm for calculating variance:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Args:
        epsilon (float): Small constant to initialize the count for numerical stability.
        shape (Tuple[int, ...]): Shape of the data (e.g., observation shape).
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        """
        Updates the running statistics using a new batch of data.

        Args:
            arr (np.ndarray): Batch of data (batch_size, *shape).
        """
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """
        Updates statistics using precomputed batch mean, variance, and count.

        Args:
            batch_mean (np.ndarray): Mean of the batch.
            batch_var (np.ndarray): Variance of the batch.
            batch_count (int): Number of samples in the batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class Normalizer(RunningMeanStd):
    """
    A normalizer that uses running statistics to normalize inputs, with optional clipping.

    Args:
        input_dim (Tuple[int, ...]): Shape of the input observations.
        epsilon (float): Small constant added to variance to avoid division by zero.
        clip_obs (float): Maximum absolute value to clip the normalized observations.
    """

    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
    ) -> None:
        shape = (input_dim,) if isinstance(input_dim, int) else input_dim
        super().__init__(epsilon=epsilon, shape=shape)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input: np.ndarray) -> np.ndarray:
        """
        Normalizes input using running mean and std, and clips the result.

        Args:
            input (np.ndarray): Input array to normalize.

        Returns:
            np.ndarray: Normalized and clipped array.
        """
        return np.clip(
            (input - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def normalize_torch(self, input: torch.Tensor, device: str) -> torch.Tensor:
        """
        Torch version of normalize(), for use in PyTorch graphs.

        Args:
            input (torch.Tensor): Input tensor to normalize.
            device (str): Device on which to place the computation ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Normalized and clipped tensor.
        """
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(
            torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32)
        )
        return torch.clamp(
            (input - mean_torch) / std_torch,
            -self.clip_obs,
            self.clip_obs,
        )

    def update_normalizer(self, rollouts, expert_loader) -> None:
        """
        Updates running statistics using samples from both policy and expert trajectories.

        Args:
            rollouts: Object with method `feed_forward_generator_amp(...)`.
            expert_loader: Dataloader or similar object providing expert batches.
        """
        policy_data_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size
        )
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(
            expert_loader.batch_size
        )

        for expert_batch, policy_batch in zip(
            expert_data_generator, policy_data_generator
        ):
            self.update(
                torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy()
            )
