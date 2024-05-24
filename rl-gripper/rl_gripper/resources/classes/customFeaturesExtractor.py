import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
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


class CustomCNN_attention(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomCNN_attention, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 24, kernel_size=6, stride=4, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout after activation
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout after activation
            nn.Conv2d(32, 48, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout after activation
            nn.Flatten(),
        )

        # Dummy input to calculate flat features size
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # Feature dimension for self-attention, reshaping is needed
        self.feature_dim = n_flatten // 48  # assuming last conv outputs 48 channels
        self.reshape = nn.Unflatten(1, (48, self.feature_dim))

        # Adding the self-attention layer
        self.self_attention = SelfAttention(feature_dim=self.feature_dim, num_heads=4)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout after activation
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observation)
        x = self.reshape(x)
        x = self.self_attention(x)
        x = x.flatten(start_dim=1)  # Flatten back before feeding into linear layer
        return self.linear(x)


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=2):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, x):
        # Permute batch and sequence dimensions:
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, features] -> [seq_len, batch_size, features]

        # Self-attention
        attn_output, attn_output_weights = self.attention(x, x, x)
        #visualize_attention_weights(attn_output_weights)

        # Permute back to the original order:
        attn_output = attn_output.permute(1, 0, 2)  # [seq_len, batch_size, features] -> [batch_size, seq_len, features]
        return attn_output


def visualize_attention_weights(attn_weights, seq_labels=None):
    """
    Visualizes the mean of the attention weights across all heads.

    Args:
    attn_weights: The attention weights tensor of shape [num_heads, seq_len, seq_len].
    seq_labels: Optional labels for the sequence positions (list of strings).
    """
    # Calculate the mean of the attention weights across all heads
    attn_weights_mean = torch.mean(attn_weights, dim=0)  # Shape: [seq_len, seq_len]
    attn_weights_mean_nparray = attn_weights_mean.cpu()
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(attn_weights_mean_nparray.detach().numpy(), cmap='bone')
    fig.colorbar(cax)

    if seq_labels:
        # Set up axes with labels, ensuring correct alignment
        ax.set_xticks(range(len(seq_labels)))
        ax.set_yticks(range(len(seq_labels)))
        ax.set_xticklabels([''] + seq_labels, rotation=90)
        ax.set_yticklabels([''] + seq_labels)

    # Show plot
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.title("Mean Attention Map Across All Heads")
    plt.show()
