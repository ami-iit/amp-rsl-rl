# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from huggingface_hub import hf_hub_download
from pathlib import Path
import os
import shutil
import tempfile
from amp_rsl_rl.utils import AMPLoader, download_amp_dataset
import torch

# Configuration
repo_id = "ami-iit/amp-dataset"
robot_folder = "ergocub"
files = [
    "ergocub_stand_still.npy",
    "ergocub_walk_left0.npy",
    "ergocub_walk.npy",
    "ergocub_walk_right2.npy",
]

# Create temporary directory for dataset
with tempfile.TemporaryDirectory() as tmpdirname:
    local_dir = Path(tmpdirname)
    dataset_names = download_amp_dataset(local_dir, robot_folder=robot_folder, files=files)


    # Use AMPLoader to load and process the dataset
    loader = AMPLoader(
        device="cpu",
        dataset_path_root=local_dir,
        dataset_names=dataset_names,
        dataset_weights=[1.0] * len(dataset_names),
        simulation_dt=1 / 60.0,
        slow_down_factor=1,
        expected_joint_names=None,
    )

    # Example usage
    motion = loader.motion_data[0]
    print("Loaded dataset with", len(motion), "frames.")
    print("Sample AMP observation:", motion.get_amp_dataset_obs(torch.tensor([0])))
