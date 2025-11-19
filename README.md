# AMP-RSL-RL

AMP-RSL-RL is a reinforcement learning library that extends the Proximal Policy Optimization (PPO) implementation of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) to incorporate Adversarial Motion Priors (AMP). This framework enables humanoid agents to learn motor skills from motion capture data using adversarial imitation learning techniques.

---

## 📦 Installation

The repository is available on PyPI under the package name **amp-rl-rsl**. You can install it directly using pip:

```bash
pip install amp-rsl-rl
```

Alternatively, if you prefer to clone the repository and install it locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/amp_rsl_rl.git
    cd amp_rsl_rl
    ```

2. Install the package:
    ```bash
    pip install .
    ```

For editable/development mode:

```bash
pip install -e .
```

If you want to run the examples, please install with:

```bash
pip install .[examples]
```

The required dependencies include:

- `numpy`
- `scipy`
- `torch`
- `rsl-rl-lib`

These will be automatically installed via pip.

---

## 📂 Project Structure

```
amp_rsl_rl/
│
├── algorithms/        # AMP and PPO implementations
├── networks/          # Neural networks for policy and discriminator
├── runners/           # Training and evaluation routines
├── storage/           # Replay buffer for experience collection
├── utils/             # Dataset loaders and motion tools
```

---

## 📁 Dataset Structure

The AMP-RSL-RL framework expects motion capture datasets in `.npy` format. Each `.npy` file must contain a Python dictionary with the following keys:

- **`joints_list`**: `List[str]`  
  A list of joint names. These should correspond to the joint order expected by the agent.

- **`joint_positions`**: `List[np.ndarray]`  
  A list where each element is a NumPy array representing the joint positions at a frame. All arrays should have the same shape `(N,)`, where `N` is the number of joints.

- **`root_position`**: `List[np.ndarray]`  
  A list of 3D vectors representing the position of the base (root) of the agent in world coordinates for each frame.

- **`root_quaternion`**: `List[np.ndarray]`  
  A list of unit quaternions in **`xyzw`** format (SciPy convention), representing the base orientation of the agent for each frame.

- **`fps`**: `float`  
  The number of frames per second in the original dataset. This is used to resample the data to match the simulator's timestep.

### Example

Here’s an example of how the structure might look when loaded in Python:

```python
{
    "joints_list": ["hip", "knee", "ankle"],
    "joint_positions": [np.array([0.1, -0.2, 0.3]), np.array([0.11, -0.21, 0.31]), ...],
    "root_position": [np.array([0.0, 0.0, 1.0]), np.array([0.01, 0.0, 1.0]), ...],
    "root_quaternion": [np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.1, 0.99]), ...],
    "fps": 120.0
}
```

All lists must have the same number of entries (i.e. one per frame). The dataset should represent smooth motion captured over time.

---

## 📚 Supported Dataset

For a ready-to-use motion capture dataset, you can use the [AMP Dataset on Hugging Face](https://huggingface.co/datasets/ami-iit/amp-dataset). This dataset is curated to work seamlessly with the AMP-RSL-RL framework.

---

## ♻️ Symmetry Augmentation

You can mirror demonstrations and policy rollouts by declaring the robot's kinematic symmetry in the training configuration. Provide a `symmetry_cfg` dictionary when building the runner:

```python
symmetry_cfg = {
  "joint_pairs": [
    ["left_hip_yaw", "right_hip_yaw"],
    ["left_knee", "right_knee"],
    # ... add all mirrored joint names here
  ],
  "center_joints": ["torso_yaw"],
  "joint_sign_overrides": {
    "left_hip_roll": -1.0,
    "right_hip_roll": -1.0,
  },
  "base_linear_sign": [1.0, -1.0, 1.0],
  "base_angular_sign": [1.0, -1.0, -1.0],
}
```

Key fields:

- **`joint_pairs`** – mirrored joint name tuples (left, right).
- **`center_joints`** – joints lying on the mirror plane (remain unchanged).
- **`joint_sign_overrides`** – optional per-joint multipliers (±1) for axes that reverse direction (e.g., roll joints).
- **`base_linear_sign` / `base_angular_sign`** – sign flips for base velocities when reflected across the sagittal plane.

When supplied, the dataset loader automatically augments motion clips with mirrored copies. The PPO agent mirrors policy-generated AMP observations as well, so the discriminator receives both original and reflected trajectories without extra code.

---

## 🧑‍💻 Authors

- **Giulio Romualdi** – [@GiulioRomualdi](https://github.com/GiulioRomualdi)
- **Giuseppe L'Erario** – [@Giulero](https://github.com/Giulero)

---

## 📄 License

BSD 3-Clause License © 2025 Istituto Italiano di Tecnologia
