import os
import wandb
from rsl_rl.utils.wandb_utils import WandbSummaryWriter as RslWandbSummaryWriter

class WandbSummaryWriter(RslWandbSummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_files = []

    # To save video files to wandb explicitly
    # Thanks to https://github.com/leggedrobotics/rsl_rl/pull/84    
    def add_video_files(self, log_dir: str, step: int, fps: int = 30):
        # Check if there are video files in the video directory
        if os.path.exists(log_dir):
            # append the new video files to the existing list
            for root, dirs, files in os.walk(log_dir):
                for video_file in files:
                    if video_file.endswith(".mp4") and video_file not in self.video_files:
                        self.video_files.append(video_file)
                        # add the new video file to wandb only if video file is not updating
                        video_path = os.path.join(root, video_file)
                        wandb.log(
                            {"Video": wandb.Video(video_path, fps=fps, format="mp4")},
                            step = step
                        )


