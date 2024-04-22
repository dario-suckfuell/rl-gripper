import torch
import random
import torch.nn as nn
from torchvision.models import resnet, resnet18
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np


class TensorboardCallback(BaseCallback):
    # Custom callback for plotting additional values in tensorboard.
    def __init__(self, model, verbose=0):
        super().__init__(verbose=0)
        self.model = model

    def _on_step(self) -> bool:
        # Log additional stats
        success_rate = self.training_env.unwrapped.get_attr("success_rate")[0]
        self.logger.record('custom/success_rate', success_rate)

        workspace = self.training_env.unwrapped.get_attr("workspace")[0]
        self.logger.record('custom/workspace', workspace.xMax - workspace.xMin)

        # if 'approx_kl' in self.locals:
        #     approx_kl = self.locals['approx_kl']
        #     self.logger.record('Approx_kl', approx_kl)

        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, model, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.model = model  # Store the model instance
        self.eval_freq = 1000  # Evaluate every 1000 steps
        self.threshold_for_increase = 0.8
        self.threshold_for_decrease = 0.3
        self.success_rate = 0
        self.n_steps = 0

    def _on_step(self) -> bool:
        """This method will be called by Stable Baselines3 at each environment step."""
        self.n_steps += 1
        # Check if it's time to evaluate the performance
        if self.n_steps % self.eval_freq == 0:
            self.success_rate = self.training_env.unwrapped.get_attr("success_rate")[0]
            self.adjust_difficulty()
        return True

    def adjust_difficulty(self):
        """Adjust the difficulty of the environment based on the agent's success rate."""
        if self.success_rate > self.threshold_for_increase:
            self.training_env.env_method('increase_difficulty', indices=None)  # Apply to all envs

    def on_training_end(self):
        """Optional: Do something at the end of training."""
        print("Training ends. Final difficulty adjustments can be made here.")
