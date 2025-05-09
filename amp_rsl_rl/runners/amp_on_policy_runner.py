# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
import rsl_rl.utils
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state

from amp_rsl_rl.utils import Normalizer
from amp_rsl_rl.utils import AMPLoader
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils import export_policy_as_onnx


class AMPOnPolicyRunner:
    """
    AMPOnPolicyRunner is a high-level orchestrator that manages the training and evaluation
    of a policy using Adversarial Motion Priors (AMP) combined with on-policy reinforcement learning (PPO).

    It brings together multiple components:
    - Environment (`VecEnv`)
    - Policy (`ActorCritic`, `ActorCriticRecurrent`)
    - Discriminator (Discriminator)
    - Expert dataset (AMPLoader)
    - Reward combination (task + style)
    - Logging and checkpointing

    ---
    🔧 Configuration
    ----------------
    The class expects a `train_cfg` dictionary structured with keys:
    - "policy": configuration for the policy network, including `"class_name"`
    - "algorithm": configuration for PPO/AMP_PPO, including `"class_name"`
    - "discriminator": configuration for the AMP discriminator
    - "amp_data_path": path to folder containing expert dataset(s)
    - "dataset_names": list of dataset filenames (without `.npy`)
    - "dataset_weights": list of float weights used to sample from datasets
    - "slow_down_factor": slowdown applied to real motion data to match sim dynamics
    - "num_steps_per_env": rollout horizon per environment
    - "save_interval": frequency (in iterations) for model checkpointing
    - "empirical_normalization": whether to apply running observation normalization
    - "logger": one of "tensorboard", "wandb", or "neptune"

    ---
    📦 Dataset format
    ------------------
    The expert motion datasets loaded via `AMPLoader` must be `.npy` files with a dictionary containing:

    - `"joints_list"`: List[str] — ordered list of joint names
    - `"joint_positions"`: List[np.ndarray] — joint configurations per timestep (1D arrays)
    - `"root_position"`: List[np.ndarray] — base position in world coordinates
    - `"root_quaternion"`: List[np.ndarray] — base orientation in **`xyzw`** format (SciPy default)
    - `"fps"`: float — original dataset frame rate

    Internally:
    - Quaternions are interpolated via SLERP and converted to **`wxyz`** format before being used by the model (to match Isaac Gym convention).
    - Velocities are estimated with finite differences.
    - All data is converted to torch tensors and placed on the desired device.

    ---
    🎓 AMP Reward
    -------------
    During each training step, the runner collects AMP-specific observations and computes
    a discriminator-based "style reward" from the expert dataset. This is combined
    with the environment reward as:

        `reward = 0.5 * task_reward + 0.5 * style_reward`

    This can be later generalized into a weighted or learned reward mixing policy.

    ---
    🔁 Training loop
    ----------------
    The `learn()` method performs:
    - `rollout`: collects data via `self.alg.act()` and `env.step()`
    - `style_reward`: computed from discriminator via `predict_reward(...)`
    - `storage update`: via `process_env_step()` and `process_amp_step()`
    - `return computation`: via `compute_returns()`
    - `update`: performs backpropagation with `self.alg.update()`
    - Logging via TensorBoard/WandB/Neptune

    ---
    💾 Saving and ONNX export
    --------------------------
    At each `save_interval`, the runner:
    - Saves the full state (`model`, `optimizer`, `discriminator`, `normalizer`, etc.)
    - Optionally exports the policy as an ONNX model for deployment
    - Uploads checkpoints to logging services if enabled

    ---
    📤 Inference policy
    -------------------
    `get_inference_policy()` returns a callable that takes an observation and returns an action.
    If empirical normalization is enabled, observations are normalized before inference.

    ---
    🛠️ Additional tools
    -------------------
    - Git integration via `store_code_state()` to track code changes
    - Logging of learning statistics, reward breakdown, discriminator metrics
    - Compatible with multi-task setups via dataset weights

    ---
    📚 Notes
    --------
    - This runner assumes an AMP-compatible VecEnv, providing `observations["amp"]`
    - AMP uses both current and next state to train the discriminator
    - Logging behavior is separated from core logic (WandB, Neptune, TensorBoard)
    - The Discriminator and AMP_PPO must follow expected APIs

    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.device = device
        self.env = env

        # Get the size of the observation space
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticMoE = (
            actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
        )

        actuated_joint_names = self.env.cfg.actions.joint_positions.joint_names

        delta_t = self.env.cfg.sim.dt * self.env.cfg.decimation

        # Initilize all the ingredients required for AMP (discriminator, dataset loader)
        num_amp_obs = extras["observations"]["amp"].shape[1]
        amp_data = AMPLoader(
            self.device,
            self.cfg["amp_data_path"],
            self.cfg["dataset_names"],
            self.cfg["dataset_weights"],
            delta_t,
            self.cfg["slow_down_factor"],
            actuated_joint_names,
        )

        # self.env.unwrapped.scene["robot"].joint_names)

        # amp_data = AMPLoader(num_amp_obs, self.device)
        self.amp_normalizer = Normalizer(num_amp_obs, device=self.device)
        self.discriminator = Discriminator(
            num_amp_obs
            * 2,  # the discriminator takes in the concatenation of the current and next observation
            self.discriminator_cfg["hidden_dims"],
            self.discriminator_cfg["reward_scale"],
            device=self.device,
        ).to(self.device)

        # Initialize the PPO algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # AMP_PPO

        self.alg: AMP_PPO = alg_class(
            actor_critic=actor_critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            amp_normalizer=self.amp_normalizer,
            device=self.device,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=[num_critic_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                import wandb

                # Update the run name with a sequence number. This function is useful to
                # replicate the same behaviour of rsl-rl-lib before v2.3.0
                def update_run_name_with_sequence(prefix: str) -> None:
                    # Retrieve the current wandb run details (project and entity)
                    project = wandb.run.project
                    entity = wandb.run.entity

                    # Use wandb's API to list all runs in your project
                    api = wandb.Api()
                    runs = api.runs(f"{entity}/{project}")

                    max_num = 0
                    # Iterate through runs to extract the numeric suffix after the prefix.
                    for run in runs:
                        if run.name.startswith(prefix):
                            # Extract the numeric part from the run name.
                            numeric_suffix = run.name[
                                len(prefix) :
                            ]  # e.g., from "prefix564", get "564"
                            try:
                                run_num = int(numeric_suffix)
                                if run_num > max_num:
                                    max_num = run_num
                            except ValueError:
                                continue

                    # Increment to get the new run number
                    new_num = max_num + 1
                    new_run_name = f"{prefix}{new_num}"

                    # Update the wandb run's name
                    wandb.run.name = new_run_name
                    print("Updated run name to:", wandb.run.name)

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                update_run_name_with_sequence(prefix=self.cfg["wandb_project"])

                wandb.gym.monitor()
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10
                )
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        amp_obs = extras["observations"]["amp"]
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout

            mean_style_reward_log = 0
            mean_task_reward_log = 0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    self.alg.act_amp(amp_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    _, extras = self.env.get_observations()
                    next_amp_obs = extras["observations"]["amp"]
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(
                            infos["observations"]["critic"]
                        )
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    next_amp_obs = next_amp_obs.to(self.device)

                    # Process the AMP reward
                    style_rewards = self.discriminator.predict_reward(
                        amp_obs, next_amp_obs, normalizer=self.amp_normalizer
                    )

                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    # Combine the task and style rewards (TODO this can be a hyperparameters)
                    rewards = 0.5 * rewards + 0.5 * style_rewards

                    self.alg.process_env_step(rewards, dones, infos)
                    self.alg.process_amp_step(next_amp_obs)

                    # The next observation becomes the current observation for the next step
                    amp_obs = torch.clone(next_amp_obs)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env
            mean_total_reward_log = mean_style_reward_log + mean_task_reward_log

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
            save_onnx=True,
        )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )

        # Adding logging due to AMP
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar(
            "Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"]
        )

        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"]
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string

        # make the eta in H:M:S
        eta_seconds = (
            self.tot_time
            / (locs["it"] + 1)
            * (locs["num_learning_iterations"] - locs["it"])
        )

        # Convert seconds to H:M:S
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None, save_onnx=False):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = (
                self.critic_obs_normalizer.state_dict()
            )
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

        if save_onnx:
            # Save the model in ONNX format
            # extract the folder path
            onnx_folder = os.path.dirname(path)

            # extract the iteration number from the path. The path is expected to be in the format
            # model_{iteration}.pt
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            onnx_model_name = f"policy_{iteration}.onnx"

            export_policy_as_onnx(
                self.alg.actor_critic,
                normalizer=self.obs_normalizer,
                path=onnx_folder,
                filename=onnx_model_name,
            )

            if self.logger_type in ["neptune", "wandb"]:
                self.writer.save_model(
                    os.path.join(onnx_folder, onnx_model_name),
                    self.current_learning_iteration,
                )

    def load(self, path, load_optimizer=True, weights_only=False):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=weights_only)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]

        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(
                loaded_dict["critic_obs_norm_state_dict"]
            )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(
                self.obs_normalizer(x)
            )  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        self.alg.discriminator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
