# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import warnings
from dataclasses import asdict

from torch.utils.tensorboard import SummaryWriter

try:
    import mlflow
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "MLflow is required to log to MLflow. Install it with: pip install mlflow"
    ) from None


class MLflowSummaryWriter(SummaryWriter):
    """Summary writer for MLflow tracking.

    This class is a drop-in replacement for :class:`WandbSummaryWriter`.
    It inherits from TensorBoard's ``SummaryWriter`` so that local TB logs
    are still produced, while simultaneously forwarding metrics, configs,
    artifacts and videos to an MLflow Tracking server.

    Configuration is read from ``cfg["mlflow_kwargs"]`` with the following keys:

    - ``"experiment_name"`` (required): MLflow experiment name.
    - ``"tracking_uri"`` (optional): MLflow tracking URI. Falls back to the
      ``MLFLOW_TRACKING_URI`` environment variable or ``"./mlruns"``.
    - ``"run_name"`` (optional): explicit run name; defaults to the last
      component of *log_dir*.
    - ``"tags"`` (optional): dict of tags forwarded to ``mlflow.start_run``.
    - ``"notes"`` (optional): description string stored as the run description.
    """

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        mlflow_kwargs: dict = cfg.get("mlflow_kwargs", {})

        # --- experiment name (required) ---
        experiment_name = mlflow_kwargs.get("experiment_name")
        if experiment_name is None:
            raise KeyError("Please specify 'experiment_name' in cfg['mlflow_kwargs'].")

        # --- tracking URI ---
        tracking_uri = mlflow_kwargs.get(
            "tracking_uri",
            os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"),
        )
        mlflow.set_tracking_uri(tracking_uri)

        # --- run name ---
        run_name = mlflow_kwargs.get("run_name", os.path.split(log_dir)[-1])

        # --- tags / description ---
        tags = mlflow_kwargs.get("tags", {})
        notes = mlflow_kwargs.get("notes", "")

        # --- start (or resume) a run ---
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            description=notes,
        )

        # Store log dir as a run param
        mlflow.log_param("log_dir", log_dir)

        # Replicate the name_map from WandbSummaryWriter for metric-key
        # sanitisation (MLflow metrics may not contain certain characters).
        self.name_map: dict[str, str] = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        # Keep track of already-logged video files (same pattern as wandb writer)
        self.video_files: list[str] = []

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def store_config(
        self,
        env_cfg: dict | object,
        runner_cfg: dict,
        alg_cfg: dict,
        policy_cfg: dict,
    ) -> None:
        """Persist configuration dicts as MLflow params."""

        def _flatten(d: dict, prefix: str = "") -> dict:
            """Flatten a nested dict into dot-separated keys."""
            items: dict = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict):
                    items.update(_flatten(v, key))
                else:
                    items[key] = v
            return items

        def _safe_log_params(params: dict) -> None:
            """Log params while truncating values that exceed MLflow's limit."""
            max_len = 500
            for k, v in params.items():
                str_v = str(v)
                if len(str_v) > max_len:
                    str_v = str_v[:max_len] + "..."
                try:
                    mlflow.log_param(k, str_v)
                except mlflow.exceptions.MlflowException:
                    pass  # param already logged – ignore

        _safe_log_params(_flatten(runner_cfg, "runner_cfg"))
        _safe_log_params(_flatten(policy_cfg, "policy_cfg"))
        _safe_log_params(_flatten(alg_cfg, "alg_cfg"))

        try:
            env_dict = (
                env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
            )
            _safe_log_params(_flatten(env_dict, "env_cfg"))
        except Exception:
            warnings.warn("Could not log env_cfg to MLflow params.")

    def log_config(
        self,
        env_cfg: dict | object,
        runner_cfg: dict,
        alg_cfg: dict,
        policy_cfg: dict,
    ) -> None:
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------
    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        # Forward to TensorBoard
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        # Sanitise metric name for MLflow
        metric_name = self.name_map.get(tag, tag)
        mlflow.log_metric(metric_name, scalar_value, step=global_step)

    # ------------------------------------------------------------------
    # Video logging
    # ------------------------------------------------------------------
    def add_video_files(self, log_dir: str, step: int) -> None:
        """Log new ``.mp4`` video files found under *log_dir* as MLflow artifacts."""
        if not os.path.exists(log_dir):
            return

        for root, _dirs, files in os.walk(log_dir):
            for video_file in files:
                if video_file.endswith(".mp4") and video_file not in self.video_files:
                    self.video_files.append(video_file)
                    video_path = os.path.join(root, video_file)
                    mlflow.log_artifact(video_path, artifact_path="videos")

    # ------------------------------------------------------------------
    # Model / file saving
    # ------------------------------------------------------------------
    def save_model(self, model_path: str, iter: int) -> None:
        """Log a model checkpoint as an MLflow artifact."""
        mlflow.log_artifact(model_path, artifact_path="models")

    def save_file(self, path: str) -> None:
        """Log an arbitrary file as an MLflow artifact."""
        mlflow.log_artifact(path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
