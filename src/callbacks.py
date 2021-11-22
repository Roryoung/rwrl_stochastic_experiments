import numpy as np
import imageio
import os
import json

from stable_baselines3.common.callbacks import BaseCallback

from utils import clear_line


class SaveBestModelStep(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.best_step = 0

    def _on_step(self) -> bool:
        self.best_step = self.parent.n_calls
        return super()._on_step()


class LoggerCallback(BaseCallback):
    def __init__(self, verbose=0, *args, **kwargs):
        super(LoggerCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        print(f"\r{self.n_calls }", end="")
        return super()._on_step()

    def _on_training_end(self) -> None:
        clear_line()


class SaveModelCallback(BaseCallback):
    def __init__(self, trainer, backup_frequency=1000 * 50, verbose=0, *args, **kwargs):
        super(SaveModelCallback, self).__init__(verbose)
        self.trainer = trainer
        self.backup_frequency = backup_frequency  

    def _on_step(self) -> bool:
        if (self.n_calls  - 1) % self.backup_frequency == 0:
            save_name = (self.n_calls -1) if self.n_calls  < 1001 else f"{int((self.n_calls  - 1)/1000)}k"

            self.model.save(f"{self.trainer.exp_dir}/ckpt/trial_{self.trainer.trial_no}/backup/{save_name}")
            self.model.save(f"{self.trainer.exp_dir}/ckpt/trial_{self.trainer.trial_no}/final_model")

        return True

    def _on_training_end(self) -> None:
        self.model.save(f"{self.trainer.exp_dir}/ckpt/trial_{self.trainer.trial_no}/final_model")
       



class SampleTrajectoryCallback(BaseCallback):
    def __init__(self, sample_frequency=None, sample_steps=None, final_steps=1000, reward_color=True, constraints_color=True, render_camera_ids=[0,1], verbose=0, *args, **kwargs):
        super(SampleTrajectoryCallback, self).__init__(verbose)
        self.sample_frequency = sample_frequency
        self.sample_steps = sample_steps
        self.final_steps = final_steps

        self.reward_color = reward_color
        self.constraints_color = constraints_color
        self.render_camera_ids = render_camera_ids

        self.images = []

        if sample_frequency is not None and sample_steps is not None:
            assert(sample_frequency > sample_steps+1)
        else:
            assert(sample_frequency is None)

    
    def _on_step(self) -> bool:
        if self.sample_frequency is not None:
            self.model.env.env_method("render_colors", reward=self.reward_color, constraints=self.constraints_color)
            self.model.env.env_method("render_angles", render_camera_ids=self.render_camera_ids)
            if self.n_calls  % self.sample_frequency < self.sample_steps:
                img = self.model.env.render(mode="rgb_array")
                self.images.append(np.array(img))
            elif self.n_calls  % self.sample_frequency == self.sample_steps:
                name = str(self.n_calls ) if self.n_calls  < 1000 else f"{str(int(self.n_calls /1000))}k" 
                imageio.mimsave(f"{self.model.exp_dir}/samples/{name}.gif", self.images, fps=50)

                self.images = []


    def _on_training_end(self) -> None:
        env = self.model.env

        env.env_method("render_colors", reward=self.reward_color, constraints=self.constraints_color)
        env.env_method("render_angles", render_camera_ids=self.render_camera_ids)

        images = []
        obs = env.reset()
        img = env.render(mode='rgb_array')

        for i in range(self.final_steps):
            images.append(np.array(img))

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _ ,_ = env.step(action)
            img = env.render(mode='rgb_array')
 
        imageio.mimsave(f"{self.model.exp_dir}/evaluation/sample_trajectory.gif", images, fps=50)        
    
