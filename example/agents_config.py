from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

@configclass
class YourConfigName(RslRlOnPolicyRunnerCfg):
    experiment_name = "XXX"
    discriminator = {
        "hidden_dims": [512, 256],
        "reward_scale": 1.0,
        "loss_type": "BCEWithLogits"  # Choose between BCEWithLogits or Wasserstein
    }
    
    # Weights for combining task and style rewards
    task_reward_weight = 0.5
    style_reward_weight = 0.5
    
    amp_data_path = "path of the dataset folder"
    dataset_names = ["files", 
                     "without", 
                     ".npy",
                     ]
    dataset_weights = [1.0 for i in range(len(dataset_names))]
    slow_down_factor = 1.0
    
    def __post_init__(self):
        self.algorithm.class_name = "AMP_PPO"