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

        # Existing CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),  # Produces the attention map
            nn.Sigmoid(),  # Normalizes the attention scores
        )

        # Flatten the feature maps before passing to the linear layer
        self.flatten = nn.Flatten()

        # Compute the shape after flattening
        with torch.no_grad():
            n_flatten = self.flatten(self.cnn(torch.as_tensor(observation_space.sample()[None]).float())).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observation: torch.tensor) -> torch.Tensor:
        # Extract features
        feature_map = self.cnn(observation)

        # Compute attention map
        attention_map = self.attention(feature_map)

        # Apply attention to the feature map
        attended_features = feature_map * attention_map

        # Flatten the attended features
        flattened = self.flatten(attended_features)

        # Pass through the linear layer
        return self.linear(flattened)


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
        value = random.randint(-1, 1)
        self.logger.record('Random Value', value)
        print(locals())

        # if 'approx_kl' in self.locals:
        #     approx_kl = self.locals['approx_kl']
        #     self.logger.record('Approx_kl', approx_kl)

        return True


class RewardStandardizationWrapper(gym.RewardWrapper):
    '''
    Reward Standardization over one Episode
    '''

    def __init__(self, env):
        super().__init__(env)
        self.rewards = []

    def reset(self, **kwargs):
        # Reset the environment and clear reward history at the start of each episode
        self.rewards = []
        return self.env.reset(**kwargs)

    def reward(self, reward):
        # Append the reward to the history for this episode
        self.rewards.append(reward)

        # Compute the mean and std of rewards for the current episode
        mean_reward = np.mean(self.rewards)
        std_reward = np.std(self.rewards) if np.std(self.rewards) > 0 else 1

        # Standardize the reward, maintaining the sign
        normalized_reward = abs(reward - mean_reward) / (std_reward + 1e-8)  # Normalize the magnitude of the reward
        normalized_reward = normalized_reward if reward > 0 else -normalized_reward  # Re-apply the original sign

        return normalized_reward
