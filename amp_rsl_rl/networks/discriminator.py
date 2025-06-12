# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd
from rsl_rl.utils import utils


class Discriminator(nn.Module):
    """Discriminator implements the discriminator network for the AMP algorithm.

    This network is trained to distinguish between expert and policy-generated data.
    It also provides reward signals for the policy through adversarial learning.

    Args:
        input_dim (int): Dimension of the concatenated input state (state + next state).
        hidden_layer_sizes (list): List of hidden layer sizes.
        reward_scale (float): Scale factor for the computed reward.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 0.0001,
        device: str = "cpu",
    ):
        super(Discriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.reward_clamp_epsilon = reward_clamp_epsilon
        layers = []
        curr_in_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers).to(device)
        self.linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.linear.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x (Tensor): Input tensor (batch_size, input_dim).

        Returns:
            Tensor: Discriminator output logits.
        """
        h = self.trunk(x)
        d = self.linear(h)
        return d

    def compute_grad_pen(
        self,
        expert_state_part: torch.Tensor,
        expert_next_state_part: torch.Tensor,
        expert_goal_vector: torch.Tensor,
        normalizer, # Expected to be an instance of amp_rsl_rl.utils.Normalizer
        lambda_: float = 10,
    ) -> torch.Tensor:
        """Computes the gradient penalty used to regularize the discriminator.
           Assumes expert_state_part and expert_next_state_part are unnormalized.
           The normalizer is applied to state parts before concatenation with goal.

        Args:
            expert_state_part (Tensor): Batch of expert state parts (unnormalized).
            expert_next_state_part (Tensor): Batch of expert next state parts (unnormalized).
            expert_goal_vector (Tensor): Batch of expert goal vectors.
            normalizer: Normalizer instance for state parts.
            lambda_ (float): Penalty coefficient.

        Returns:
            Tensor: Gradient penalty value.
        """
        if normalizer is not None:
            expert_state_part_norm = normalizer.normalize(expert_state_part)
            expert_next_state_part_norm = normalizer.normalize(expert_next_state_part)
        else:
            expert_state_part_norm = expert_state_part
            expert_next_state_part_norm = expert_next_state_part

        expert_data = torch.cat([expert_state_part_norm, expert_next_state_part_norm, expert_goal_vector], dim=-1)
        expert_data.requires_grad = True

        disc = self.forward(expert_data)
        ones = torch.ones(disc.size(), device=disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    # TODO: remove the complete function
    def predict_reward_old(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        normalizer=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Old version of the reward prediction function (deprecated).

        Args:
            state (Tensor): Current state tensor.
            next_state (Tensor): Next state tensor.
            normalizer (Optional): Optional state normalizer.

        Returns:
            Tuple[Tensor, Tensor]: Computed reward and discriminator output.
        """
        with torch.no_grad():
            if normalizer is not None:
                state = normalizer.normalize(state)
                next_state = normalizer.normalize(next_state)

            d = self.forward(torch.cat([state, next_state], dim=-1))
            reward = torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)

        return reward.squeeze(), d

    def predict_reward(
        self,
        state_part: torch.Tensor,
        next_state_part: torch.Tensor,
        goal_vector: torch.Tensor,
        normalizer=None, # Expected to be an instance of amp_rsl_rl.utils.Normalizer
    ) -> torch.Tensor:
        """Predicts reward based on discriminator output using a log-style formulation.
           Assumes state_part and next_state_part are unnormalized.
           The normalizer is applied to state parts before concatenation with goal.

        Args:
            state_part (Tensor): Current state part tensor (unnormalized).
            next_state_part (Tensor): Next state part tensor (unnormalized).
            goal_vector (Tensor): Goal vector tensor.
            normalizer (Optional): Optional state normalizer for state parts.

        Returns:
            Tensor: Computed adversarial reward.
        """
        with torch.no_grad():
            state_part_norm = state_part
            next_state_part_norm = next_state_part
            if normalizer is not None:
                state_part_norm = normalizer.normalize(state_part)
                next_state_part_norm = normalizer.normalize(next_state_part)

            d_input = torch.cat([state_part_norm, next_state_part_norm, goal_vector], dim=-1)
            discriminator_logit = self.forward(d_input)
            prob = torch.sigmoid(discriminator_logit)

            # Avoid log(0) by clamping the input to a minimum threshold
            reward = -torch.log(
                torch.maximum(
                    1 - prob,
                    torch.tensor(self.reward_clamp_epsilon, device=self.device),
                )
            )

            reward = self.reward_scale * reward
            return reward.squeeze()
