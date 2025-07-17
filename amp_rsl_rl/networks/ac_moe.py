from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


class MLP_net(nn.Sequential):
    def __init__(self, in_dim, hidden_dims, out_dim, act):
        layers = [nn.Linear(in_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act])
        super().__init__(*layers)


class ActorMoE(nn.Module):
    """
    Mixture-of-Experts actor:  ⎡expert_1(x) … expert_K(x)⎤·softmax(gate(x))
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims,
        num_experts: int = 4,
        gate_hidden_dims: list[int] | None = None,
        activation="elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        act = resolve_nn_activation(activation)

        # experts
        self.experts = nn.ModuleList(
            [MLP_net(obs_dim, hidden_dims, act_dim, act) for _ in range(num_experts)]
        )

        # gating network
        gate_layers = []
        last_dim = obs_dim
        gate_hidden_dims = gate_hidden_dims or []
        for h in gate_hidden_dims:
            gate_layers += [nn.Linear(last_dim, h), act]
            last_dim = h
        gate_layers.append(nn.Linear(last_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)
        self.softmax = nn.Softmax(dim=-1)  # kept separate for ONNX clarity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, obs_dim]
        Returns:
            mean action: [batch, act_dim]
        """
        expert_out = torch.stack([e(x) for e in self.experts], dim=-1)
        gate_logits = self.gate(x)  # [batch, K]
        weights = self.softmax(gate_logits).unsqueeze(1)  # [batch, 1, K]
        return (expert_out * weights).sum(-1)  # weighted sum -> [batch, A]


class ActorCriticMoE(nn.Module):
    """Actor-critic with Mixture-of-Experts policy."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        num_experts: int = 4,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                (
                    "ActorCriticMoE.__init__ ignored unexpected arguments: "
                    + str(list(kwargs.keys()))
                )
            )
        super().__init__()
        act = resolve_nn_activation(activation)

        # Actor (Mixture-of-Experts)
        self.actor = ActorMoE(
            obs_dim=num_actor_obs,
            act_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=actor_hidden_dims[:-1],  # last layer is output
            activation=activation,
        )

        # Critic
        self.critic = MLP_net(num_critic_obs, critic_hidden_dims, 1, act)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Actor (MoE) structure:\n{self.actor}")
        print(f"Critic MLP structure:\n{self.critic}")

    def reset(self, dones=None):  # noqa: D401
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:  # "log"
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # deterministic (mean) action
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    # unchanged load_state_dict so checkpoints from the old class still load
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
