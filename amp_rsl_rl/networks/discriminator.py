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
        loss_type: str = "BCEWithLogits",
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
        loss_type = "wgan"
        self.loss_type = loss_type
        if loss_type == "BCEWithLogits":
            self.loss_fun = torch.nn.BCEWithLogitsLoss()
        elif loss_type == "wgan":
            self.loss_fun = None
        else:
            raise ValueError(
                f"Unsupported loss type: {loss_type}. Supported types are 'BCEWithLogits' and 'wgan'."
            )
        # self.loss_fun = torch.nn.BCEWithLogitsLoss()

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

    # def compute_grad_pen(
    #     self,
    #     expert_state: torch.Tensor,
    #     expert_next_state: torch.Tensor,
    #     lambda_: float = 10,
    # ) -> torch.Tensor:
    #     """Computes the gradient penalty used to regularize the discriminator.

    #     Args:
    #         expert_state (Tensor): Batch of expert states.
    #         expert_next_state (Tensor): Batch of expert next states.
    #         lambda_ (float): Penalty coefficient.

    #     Returns:
    #         Tensor: Gradient penalty value.
    #     """
    #     expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    #     expert_data.requires_grad = True

    #     disc = self.forward(expert_data)
    #     ones = torch.ones(disc.size(), device=disc.device)

    #     grad = autograd.grad(
    #         outputs=disc,
    #         inputs=expert_data,
    #         grad_outputs=ones,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #     )[0]

    #     grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
    #     return grad_pen

    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        normalizer=None,
    ) -> torch.Tensor:
        """Predicts reward based on discriminator output using a log-style formulation.

        Args:
            state (Tensor): Current state tensor.
            next_state (Tensor): Next state tensor.
            normalizer (Optional): Optional state normalizer.

        Returns:
            Tensor: Computed adversarial reward.
        """
        with torch.no_grad():
            if normalizer is not None:
                state = normalizer.normalize(state)
                next_state = normalizer.normalize(next_state)

            discriminator_logit = self.forward(torch.cat([state, next_state], dim=-1))

            if self.loss_type == "wgan":
                # reward ∝  -D(x)  (policy wants to *raise* D)
                return (self.reward_scale * discriminator_logit).squeeze()

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

    def policy_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the discriminator when classifying policy-generated transitions.
        Uses binary cross-entropy loss where the target label for policy transitions is 0.

        Parameters
        ----------
        discriminator_output : torch.Tensor
            The raw logits output from the discriminator for policy data.

        Returns
        -------
        torch.Tensor
            The computed policy loss.
        """
        expected = torch.zeros_like(discriminator_output, device=self.device)
        return self.loss_fun(discriminator_output, expected)

    def expert_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the discriminator when classifying expert transitions.
        Uses binary cross-entropy loss where the target label for expert transitions is 1.

        Parameters
        ----------
        discriminator_output : torch.Tensor
            The raw logits output from the discriminator for expert data.

        Returns
        -------
        torch.Tensor
            The computed expert loss.
        """
        expected = torch.ones_like(discriminator_output, device=self.device)
        return self.loss_fun(discriminator_output, expected)

    def compute_loss(
        self, policy_d, expert_d, sample_amp_expert, sample_amp_policy=None
    ):

        # Compute gradient penalty to stabilize discriminator training.
        grad_pen_loss = self.compute_grad_pen(
            expert_states=sample_amp_expert,
            policy_states=sample_amp_policy,
            lambda_=10,
        )
        if self.loss_type == "BCEWithLogits":
            expert_loss = self.loss_fun(expert_d, torch.ones_like(expert_d))
            policy_loss = self.loss_fun(policy_d, torch.zeros_like(policy_d))
            # AMP loss is the average of expert and policy losses.
            amp_loss = 0.5 * (expert_loss + policy_loss)
        elif self.loss_type == "wgan":
            amp_loss = self.wgan_loss(
                policy_d=policy_d, expert_d=expert_d
            )
        return amp_loss, grad_pen_loss

    def compute_grad_pen(
        self,
        expert_states: tuple[torch.Tensor, torch.Tensor],
        policy_states: tuple[torch.Tensor, torch.Tensor] = None,
        lambda_: float = 10,
    ) -> torch.Tensor:
        """Computes the gradient penalty used to regularize the discriminator.

        Args:
            expert_state (Tensor): Batch of expert states.
            expert_next_state (Tensor): Batch of expert next states.
            lambda_ (float): Penalty coefficient.

        Returns:
            Tensor: Gradient penalty value.
        """
        expert = torch.cat(expert_states, -1)

        if self.loss_type == "wgan":
            policy = torch.cat(policy_states, -1)
            alpha = torch.rand(expert.size(0), 1, device=expert.device)
            data = alpha * expert + (1 - alpha) * policy
            offset = 0
        elif self.loss_type == "BCEWithLogits":
            data = expert
            offset = 1
        else:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. Supported types are 'BCEWithLogits' and 'wgan'."
            )

        data.requires_grad = True
        scores = self.forward(data)
        grad = autograd.grad(
            outputs=scores,
            inputs=data,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return lambda_ * (grad.norm(2, dim=1) - offset).pow(2).mean()

    @staticmethod
    def wgan_loss(policy_d, expert_d):
        """
        Wasserstein critic loss (to *minimize*):
        Args:
            policy_d (Tensor): Discriminator output for policy data.
            expert_d (Tensor): Discriminator output for expert data.
        """
        return policy_d.mean() - expert_d.mean()

    @staticmethod
    def wgan_pi_loss(policy_d):
        """
        Wasserstein policy loss (to *maximize*):
        Args:
            policy_d (Tensor): Discriminator output for policy data.
        """
        return -policy_d.mean()
