import torch
import random
import torch.nn as nn
from torchvision.models import resnet, resnet18
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np


class CustomCNN(BaseFeaturesExtractor):
    # Custom Feature Extractor: Custom CNN
    def __init__(self, observation_space, features_dim):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observation: torch.tensor) -> torch.Tensor:
        return self.linear(self.cnn(observation))


class CustomCNN_attention(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomCNN_attention, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dummy input to calculate flat features size
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # Feature dimension for self-attention, reshaping is needed
        self.feature_dim = n_flatten // 64  # assuming last conv outputs 64 channels
        self.reshape = nn.Unflatten(1, (64, self.feature_dim))

        # Adding the self-attention layer
        self.self_attention = SelfAttention(feature_dim=self.feature_dim)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observation)
        x = self.reshape(x)
        x = self.self_attention(x)
        x = x.flatten(start_dim=1)  # Flatten back before feeding into linear layer
        return self.linear(x)


class CustomCNN_maxPooling(BaseFeaturesExtractor):
    # Custom Feature Extractor: Custom CNN
    def __init__(self, observation_space, features_dim):
        super(CustomCNN_maxPooling, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # First MaxPooling layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second MaxPooling layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Typically, a MaxPooling layer is not added after the last Conv layer right before flattening,
            # especially if the feature map has become quite small.
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observation: torch.tensor) -> torch.Tensor:
        return self.linear(self.cnn(observation))


class CustomCNN_big(BaseFeaturesExtractor):
    # Custom Feature Extractor: Custom CNN
    def __init__(self, observation_space, features_dim):
        super(CustomCNN_big, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observation: torch.tensor) -> torch.Tensor:
        return self.linear(self.cnn(observation))


### VORTRAINIERTES RESNET ###
class CustomResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super(CustomResNetFeatureExtractor, self).__init__(observation_space, features_dim)

        # Preprocessing layer to convert n-channel images to 3-channel
        n_input_channels = observation_space.shape[0]
        self.preprocess = nn.Conv2d(n_input_channels, 3, kernel_size=1)

        # Load a pre-trained ResNet model
        self.resnet = resnet18(weights="DEFAULT")

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer
        num_features_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features_in, features_dim)

        # for name, param in self.resnet.named_parameters():
        #     if param.requires_grad:
        #         print(name)

    def forward(self, observations):
        # Apply the preprocessing layer
        observations = self.preprocess(observations)

        # Pass through the ResNet model
        return self.resnet(observations)



class TensorboardCallback(BaseCallback):
    # Custom callback for plotting additional values in tensorboard.
    def __init__(self, verbose=0):
        super().__init__(verbose=0)

    def _on_step(self) -> bool:
        # Log additional stats
        success_rate = self.training_env.unwrapped.get_attr("success_rate")[0]
        self.logger.record('custom/success_rate', success_rate)

        gripperStartPos = self.training_env.unwrapped.get_attr("cube_start_pos")[0]
        self.logger.record('custom/cube_start_pos', gripperStartPos[2])

        # if 'approx_kl' in self.locals:
        #     approx_kl = self.locals['approx_kl']
        #     self.logger.record('Approx_kl', approx_kl)

        return True


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=3):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, x):
        # Permute batch and sequence dimensions:
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, features] -> [seq_len, batch_size, features]
        # Self-attention
        x, _ = self.attention(x, x, x)
        # Permute back to the original order:
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, features] -> [batch_size, seq_len, features]
        return x


class CurriculumCallback(BaseCallback):
    def __init__(self, model, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.model = model  # Store the model instance
        self.eval_freq = 1000  # Evaluate every 1000 steps
        self.threshold_for_increase = 0.6
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
