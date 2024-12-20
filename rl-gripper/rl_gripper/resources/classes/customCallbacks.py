import torch
import random
import torch.nn as nn
from torchvision.models import resnet, resnet18
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import os
import gym
import numpy as np


class TensorboardCallback(BaseCallback):
    # Custom callback for plotting additional values in tensorboard.
    def __init__(self, model, log_dir, verbose=0):
        super().__init__(verbose=0)
        self.model = model
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self) -> None:
        # Initialize the SummaryWriter on training start
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _on_step(self) -> bool:
        # Log additional stats
        success_rate = self.training_env.unwrapped.get_attr("success_rate")[0]
        self.logger.record('custom/success_rate', success_rate)

        self.logger.record('custom/action_noise_sigma', self.model.action_noise._sigma[0])

        workspace = self.training_env.unwrapped.get_attr("workspace")[0]
        self.logger.record('curriculum/workspace_area', workspace.xMax - workspace.xMin)

        gripper_start_pos = self.training_env.unwrapped.get_attr("gripper_start_pos")[0]
        self.logger.record('curriculum/gripper_start_height', gripper_start_pos[2])

        lifting_height = self.training_env.unwrapped.get_attr("lifting_height")[0]
        self.logger.record('curriculum/lifting_height', lifting_height)

        curriculum = self.training_env.unwrapped.get_attr("curriculum")[0]
        self.logger.record('curriculum/curriculum_level', curriculum.laps_counter)

        self.logger.record('sac/critic_grad_norm', self.model.critic_grad_norm_before_clip.item())
        self.logger.record('sac/actor_grad_norm', self.model.actor_grad_norm_before_clip.item())
        batch_mean_qf_pi = self.model.min_qf_pi.mean()
        self.logger.record('sac/batch_mean_qf_pi', batch_mean_qf_pi.item())

        # object_stats = self.training_env.unwrapped.get_attr("object_stats")[0]
        # for obj_name, stats in object_stats.items():
        #     successes = stats["successes"]
        #     attempts = stats["attempts"]
        #     obj_sr = stats["obj_sr"]
        #
        #     # self.logger.record(f"object_stats/{obj_name}", successes, attempts)
        #     self.writer.add_scalar(f"object_stats/{obj_name}", successes, attempts)
        #     self.writer.add_scalar(f"object_sr/{obj_name}", obj_sr, attempts)


        return True

    def _on_training_end(self) -> None:
        # Close the SummaryWriter at the end of training
        if self.writer:
            self.writer.close()

class CurriculumCallback(BaseCallback):
    def __init__(self, model, threshold_for_increase=0.7, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.model = model  # Store the model instance
        self.eval_freq = 1000  # Evaluate every 1000 steps
        self.threshold_for_increase = threshold_for_increase
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
            # self.training_env.env_method('increase_dataset', indices=None)  # Apply to all envs

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


class ActionNoiseDecayCallback(BaseCallback):
    def __init__(self, action_noise, initial_sigma, min_sigma, total_timesteps, verbose=0):
        super(ActionNoiseDecayCallback, self).__init__(verbose)
        self.action_noise = action_noise
        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Decay the noise sigma
        # new_sigma = np.maximum(self.min_sigma, self.initial_sigma + ((self.min_sigma-self.initial_sigma)/self.total_timesteps) * self.num_timesteps) #Linear


        # 1/x decay
        T_over_tau = 0.05 #increase for slower decay
        K = (self.initial_sigma - self.min_sigma) * (T_over_tau) / T_over_tau
        B = self.initial_sigma - K
        t_over_tau = self.num_timesteps / (self.total_timesteps*0.1)
        new_sigma = np.maximum(self.min_sigma, K / (t_over_tau + 1) + B)


        self.model.action_noise._sigma = new_sigma
        return True


