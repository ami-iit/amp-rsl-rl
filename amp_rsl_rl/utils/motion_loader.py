# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from pathlib import Path
from typing import List, Union, Tuple, Generator
from dataclasses import dataclass

import torch
import numpy as np
import random
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


def download_amp_dataset_from_hf(
    destination_dir: Path,
    robot_folder: str,
    files: list,
    repo_id: str = "ami-iit/amp-dataset",
) -> list:
    """
    Downloads AMP dataset files from Hugging Face and saves them to `destination_dir`.
    Ensures real file copies (not symlinks or hard links).

    Args:
        destination_dir (Path): Local directory to save the files.
        robot_folder (str): Folder in the Hugging Face dataset repo to pull from.
        files (list): List of filenames to download.
        repo_id (str): Hugging Face repository ID. Default is "ami-iit/amp-dataset".

    Returns:
        List[str]: List of dataset names (without .npy extension).
    """

    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = []
    for file in files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{robot_folder}/{file}",
            repo_type="dataset",
            local_files_only=False,
        )
        local_copy = destination_dir / file

        # Deep copy to avoid symlinks
        with open(file_path, "rb") as src_file:
            with open(local_copy, "wb") as dst_file:
                dst_file.write(src_file.read())

        dataset_names.append(file.replace(".npy", ""))

    return dataset_names


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    This class stores joint positions and velocities, base velocities (both in local
    and mixed/world frames), and base orientation (as quaternion). It offers utilities
    for preparing data in AMP-compatible format, as well as environment reset states.

    Attributes:
        - joint_positions: shape (T, N)
        - joint_velocities: shape (T, N)
        - base_lin_velocities_mixed: linear velocity in world frame
        - base_ang_velocities_mixed: (currently zeros)
        - base_lin_velocities_local: linear velocity in local (body) frame
        - base_ang_velocities_local: (currently zeros)
        - base_quat: orientation quaternion as torch.Tensor in wxyz order

    Notes:
        - The quaternion is expected in the dataset as `xyzw` format (SciPy default),
          and it is converted internally to `wxyz` format to be compatible with IsaacLab conventions.
        - All data is converted to torch.Tensor on the specified device during initialization.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    device: str = "cpu"

    def __post_init__(self) -> None:
        # convert numpy arrays to torch tensors
        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = torch.tensor(
                self.joint_positions, device=self.device, dtype=torch.float32
            )
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = torch.tensor(
                self.joint_velocities, device=self.device, dtype=torch.float32
            )
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = torch.tensor(
                self.base_lin_velocities_mixed, device=self.device, dtype=torch.float32
            )
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = torch.tensor(
                self.base_ang_velocities_mixed, device=self.device, dtype=torch.float32
            )
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = torch.tensor(
                self.base_lin_velocities_local, device=self.device, dtype=torch.float32
            )
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = torch.tensor(
                self.base_ang_velocities_local, device=self.device, dtype=torch.float32
            )
        if isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()
            self.base_quat = torch.tensor(
                quat_xyzw[:, [3, 0, 1, 2]], device=self.device, dtype=torch.float32
            )

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the AMP observation tensor for given indices.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Concatenated observation tensor
        """
        return torch.cat(
            (
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
            ),
            dim=1,
        )

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns the full state needed for environment reset.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
    """
    Loader and processor for humanoid motion capture datasets in AMP format.

    This class is responsible for:
      - Loading `.npy` files containing motion data
      - Optionally reordering joint names to match expected ones
      - Resampling trajectories to match the simulator's timestep
      - Computing derived quantities (velocities, local-frame motion)
      - Returning torch-friendly `MotionData` instances

    Dataset format:
        Each dataset is a `.npy` file containing a dictionary with the following keys:
        - "joints_list": List[str]         → names of joints
        - "joint_positions": List[np.ndarray]  → per-frame joint positions (array of shape (N,))
        - "root_position": List[np.ndarray]    → per-frame base position in world (3D)
        - "root_quaternion": List[np.ndarray]  → per-frame base orientation in `xyzw` format
        - "fps": float                        → original sampling rate (frames per second)

    Internally:
        - Data is resampled via interpolation to match the simulation timestep
        - Quaternion data is interpolated using SLERP (spherical interpolation)
        - The `xyzw` quaternions are converted to `wxyz` format before conversion to torch.Tensor
        - Velocities are estimated via finite differences (naïve backward difference)

    Args:
        device: Target torch device ('cpu' or 'cuda')
        dataset_path_root: Directory containing the `.npy` motion files
        dataset_names: List of dataset filenames (without extension)
        dataset_weights: List of sampling weights (used for minibatch sampling)
        simulation_dt: Timestep used by the simulator
        slow_down_factor: Integer factor used to slow down original dataset motion
        expected_joint_names: (Optional) target joint name order. If provided, joint data will be permuted accordingly.
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str],
        dataset_weights: List[float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
    ) -> None:
        self.device = device
        self.motion_data: List[MotionData] = []
        self.dataset_weights = torch.tensor(
            dataset_weights, dtype=torch.float32, device=self.device
        )
        self.dataset_weights /= self.dataset_weights.sum()

        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.npy"
            self.motion_data.append(
                self.load_data(
                    dataset_path, simulation_dt, slow_down_factor, expected_joint_names
                )
            )

    def _resample_data_Rn(
        self, data: List[np.ndarray], original_keyframes, target_keyframes
    ) -> np.ndarray:
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    def _resample_data_SO3(
        self, raw_quaternions: List[np.ndarray], original_keyframes, target_keyframes
    ) -> Rotation:

        # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    def _compute_raw_derivative(
        self, data: List[np.ndarray], dt: float
    ) -> List[np.ndarray]:
        velocities = [(data[i + 1] - data[i]) / dt for i in range(len(data) - 1)]
        return velocities

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> MotionData:
        """
        Loads and processes one motion dataset.

        Returns:
            MotionData instance
        """
        data = np.load(str(dataset_path), allow_pickle=True).item()
        dataset_joint_names = data["joints_list"]

        if expected_joint_names is not None:
            permutation_matrix = np.zeros(
                (len(expected_joint_names), len(dataset_joint_names))
            )
            for i, joint_name in enumerate(expected_joint_names):
                if joint_name in dataset_joint_names:
                    permutation_matrix[i, dataset_joint_names.index(joint_name)] = 1
        else:
            permutation_matrix = np.eye(len(dataset_joint_names))

        joint_positions = [permutation_matrix @ jp for jp in data["joint_positions"]]

        dt = 1.0 / data["fps"] / float(slow_down_factor)

        original_keyframes = np.linspace(
            0, len(joint_positions) * dt, len(joint_positions)
        )
        target_keyframes = np.linspace(
            0, len(joint_positions) * dt, int(len(joint_positions) * dt / simulation_dt)
        )

        resampled_joint_positions = self._resample_data_Rn(
            joint_positions, original_keyframes, target_keyframes
        )
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )
        resampled_joint_velocities.append(resampled_joint_velocities[-1])

        original_keyframes = np.linspace(
            0, len(joint_positions) * dt, len(joint_positions)
        )
        target_keyframes = np.linspace(
            0,
            len(resampled_joint_positions) * simulation_dt,
            len(resampled_joint_positions),
        )

        resampled_base_positions = self._resample_data_Rn(
            data["root_position"], original_keyframes, target_keyframes
        )
        resampled_base_orientations = self._resample_data_SO3(
            data["root_quaternion"], original_keyframes, target_keyframes
        )

        resampled_base_lin_velocities_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )
        resampled_base_lin_velocities_mixed.append(
            resampled_base_lin_velocities_mixed[-1]
        )

        resampled_base_lin_velocities_local = [
            R.as_matrix().T @ v
            for R, v in zip(
                resampled_base_orientations, resampled_base_lin_velocities_mixed
            )
        ]

        return MotionData(
            joint_positions=np.array(resampled_joint_positions),
            joint_velocities=np.array(resampled_joint_velocities),
            base_lin_velocities_mixed=np.array(resampled_base_lin_velocities_mixed),
            base_ang_velocities_mixed=np.zeros_like(
                resampled_base_lin_velocities_mixed
            ),
            base_lin_velocities_local=np.array(resampled_base_lin_velocities_local),
            base_ang_velocities_local=np.zeros_like(
                resampled_base_lin_velocities_local
            ),
            base_quat=resampled_base_orientations,
            device=self.device,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yields mini-batches of (state, next_state) pairs for training.

        Yields:
            Tuple[Tensor, Tensor]: (current_state, next_state)
        """
        sampled_indices = torch.multinomial(
            self.dataset_weights, mini_batch_size, replacement=True
        )
        counts = torch.bincount(sampled_indices, minlength=len(self.dataset_weights))
        mini_batch_size_per_dataset = counts.long()

        for _ in range(num_mini_batch):
            indices = [
                torch.randint(0, len(data) - 1, (size,), device=self.device)
                for data, size in zip(self.motion_data, mini_batch_size_per_dataset)
            ]
            states, next_states = [], []
            for data, idx in zip(self.motion_data, indices):
                states.append(data.get_amp_dataset_obs(idx))
                next_states.append(data.get_amp_dataset_obs(idx + 1))
            yield torch.cat(states), torch.cat(next_states)

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples full states for environment resets.

        Args:
            number_of_samples (int): Number of samples to return.

        Returns:
            Tuple of Tensors: (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        sampled_indices = torch.multinomial(
            self.dataset_weights, number_of_samples, replacement=True
        )
        counts = torch.bincount(sampled_indices, minlength=len(self.dataset_weights))
        number_of_samples_per_dataset = counts.long()

        states = []
        for data, n in zip(self.motion_data, number_of_samples_per_dataset):
            indices = torch.randint(0, len(data), (n,), device=self.device)
            states.append(data.get_state_for_reset(indices))

        random.shuffle(states)
        (
            quat,
            joint_positions,
            joint_velocities,
            base_lin_velocities,
            base_ang_velocities,
        ) = zip(*states)
        return (
            torch.cat(quat),
            torch.cat(joint_positions),
            torch.cat(joint_velocities),
            torch.cat(base_lin_velocities),
            torch.cat(base_ang_velocities),
        )
