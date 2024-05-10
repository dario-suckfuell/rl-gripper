import torch
import random
import torch.nn as nn
from torchvision.models import resnet, resnet18
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os
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
        self.logger.record('custom/workspace_area', workspace.xMax - workspace.xMin)

        gripper_start_pos = self.training_env.unwrapped.get_attr("gripper_start_pos")[0]
        self.logger.record('custom/gripper_start_height', gripper_start_pos[2])

        picking_height = self.training_env.unwrapped.get_attr("picking_height")[0]
        self.logger.record('custom/picking_height', picking_height)

        # if 'approx_kl' in self.locals:
        #     approx_kl = self.locals['approx_kl']
        #     self.logger.record('Approx_kl', approx_kl)

        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, model, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.model = model  # Store the model instance
        self.eval_freq = 1000  # Evaluate every 1000 steps
        self.threshold_for_increase = 0.7
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
        if self.success_rate > self.threshold_for_increase:  #or self.n_steps < 2001:
            self.training_env.env_method('increase_difficulty', indices=None)  # Apply to all envs


class SaveNormalizationCallback(BaseCallback):
    """
    Custom callback to save normalization parameters when a new best model is found.
    """
    def __init__(self, train_env, save_path, verbose=0):
        super(SaveNormalizationCallback, self).__init__(verbose)
        self.train_env = train_env
        self.save_path = save_path

    def _on_step(self):
        print("VecNorm parameters saved!")
        self.model.get_vec_normalize_env().save(os.path.join(self.save_path, "best_model_vec_normalize.pkl"))
        return True

