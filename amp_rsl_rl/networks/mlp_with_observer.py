# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticWithObserver(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        num_estimator_obs: int,
        num_estimator_outputs: int,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        estimator_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "ActorCriticWithObserver ignored unexpected arguments: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()

        if num_estimator_obs <= 0:
            raise ValueError(
                "num_estimator_obs must be > 0 to build the observer network"
            )
        if num_estimator_outputs <= 0:
            raise ValueError(
                "num_estimator_outputs must be > 0 to define the observer prediction size"
            )
        if num_actor_obs <= num_estimator_obs:
            raise ValueError(
                "num_actor_obs must be larger than num_estimator_obs so that policy-specific observations remain"
            )

        self.policy_obs_dim = num_actor_obs - num_estimator_obs
        self.estimator_obs_dim = num_estimator_obs
        self.estimator_output_dim = num_estimator_outputs
        self.num_actions = num_actions

        activation_layer = resolve_nn_activation(activation)

        self.actor = self._build_mlp(
            input_dim=self.policy_obs_dim + self.estimator_output_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation_layer,
        )

        self.critic = self._build_mlp(
            input_dim=num_critic_obs,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation_layer,
        )

        self.estimator = self._build_mlp(
            input_dim=self.estimator_obs_dim,
            hidden_dims=estimator_hidden_dims,
            output_dim=self.estimator_output_dim,
            activation=activation_layer,
        )

        print("Actor network with observer:")
        print(self.actor)
        print("Critic network:")
        print(self.critic)
        print("Estimator network:")
        print(self.estimator)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Expected 'scalar' or 'log'."
            )

        self.distribution: Normal | None = None
        self._last_estimator_output: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int,
        activation: nn.Module,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*layers)

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for idx in range(len(hidden_dims)):
            in_dim = hidden_dims[idx] if idx == 0 else hidden_dims[idx]
            if idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _split_actor_observations(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_obs = observations[..., : self.policy_obs_dim]
        estimator_obs = observations[..., self.policy_obs_dim :]
        return policy_obs, estimator_obs

    def _compute_actor_mean(self, observations: torch.Tensor) -> torch.Tensor:
        policy_obs, estimator_obs = self._split_actor_observations(observations)
        estimator_output = self.estimator(estimator_obs)
        self._last_estimator_output = estimator_output
        actor_input = torch.cat((policy_obs, estimator_output.detach()), dim=-1)
        return self.actor(actor_input)

    @property
    def action_mean(self):
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not updated. Call act() or update_distribution() first."
            )
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not updated. Call act() or update_distribution() first."
            )
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not updated. Call act() or update_distribution() first."
            )
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self._compute_actor_mean(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not updated. Call act() before requesting log probabilities."
            )
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._compute_actor_mean(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def predict_estimator(self, estimator_observations: torch.Tensor) -> torch.Tensor:
        return self.estimator(estimator_observations)

    def policy_parameters(self):
        noise_params = [self.std] if self.noise_std_type == "scalar" else [self.log_std]
        return itertools.chain(
            self.actor.parameters(),
            self.critic.parameters(),
            noise_params,
        )

    def estimator_parameters(self):
        return self.estimator.parameters()

    @property
    def last_estimator_output(self) -> torch.Tensor | None:
        return self._last_estimator_output
