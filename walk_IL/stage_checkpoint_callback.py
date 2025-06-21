from typing import List, Optional
import os
from stable_baselines3.common.callbacks import BaseCallback


class StageCheckpointCallback(BaseCallback):
    """Save the model whenever the environment reports a new curriculum stage.

    Assumes that each env.step(info) dictionary contains a key 'stage'.
    Saves the model to ``save_dir/ppo_stage_{stage}_steps_{timesteps}.zip`` and records
    each transition timestep in ``self.stage_change_steps`` for later plotting.
    """

    def __init__(self, save_dir: str = "stage_checkpoints", verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.prev_stage: Optional[int] = None
        self.stage_change_steps: List[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # We look at the first env in the vector (works for VecEnv as all share stage)
        stage = infos[0].get("stage")
        if stage is None:
            return True

        if self.prev_stage is None:
            self.prev_stage = stage
            return True

        if stage != self.prev_stage:
            # Stage transition detected
            self.prev_stage = stage
            self.stage_change_steps.append(self.num_timesteps)

            if self.verbose:
                print(f"[StageCheckpoint] Stage changed to {stage} at step {self.num_timesteps}")

            # Save model
            filename = os.path.join(self.save_dir, f"ppo_stage_{stage}_steps_{self.num_timesteps}.zip")
            self.model.save(filename)
            if self.verbose:
                print(f"[StageCheckpoint] Model saved to {filename}")

        return True

    def _on_training_end(self) -> None:
        if self.verbose and self.stage_change_steps:
            print("[StageCheckpoint] Stage change steps:", self.stage_change_steps) 