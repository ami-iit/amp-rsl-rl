import os
import warnings
import wandb
from rsl_rl.utils.wandb_utils import WandbSummaryWriter as RslWandbSummaryWriter
from torch.utils.tensorboard import SummaryWriter

class WandbSummaryWriter(RslWandbSummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        SummaryWriter.__init__(self, log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]
        
        # Thanks to https://github.com/leggedrobotics/rsl_rl/pull/80/
        try:
            project = cfg['wandb_kwargs']["project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None

        try:
            entity = cfg['wandb_kwargs']["entity"]
        except KeyError:
            entity = None
            warnings.warn("wandb_entity not specified in the runner config.")
        
        try:
            group = cfg['wandb_kwargs']["group"]
        except KeyError:
            warnings.warn("wandb_group not specified in the runner config. Using default group.")

        # Initialize wandb
        wandb.init(
            project=project, 
            entity=entity, 
            name=run_name,
            group=group,
            notes=cfg['wandb_kwargs']['notes'],
        )

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})