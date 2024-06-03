import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from torchvision.models import resnet, resnet18
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np
import time


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


class EfficientNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(EfficientNetFeatureExtractor, self).__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "EfficientNet requires 3D input (H, W, C)"
        self.c, self.h, self.w = observation_space.shape
        assert self.c == 1, "EfficientNet expects single-channel depth input"

        # Define the transformations
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.to_three_channels = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))

        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.efficientnet._fc.in_features #1280
        self.efficientnet._fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(in_features, features_dim)

        # Freeze EfficientNet parameters
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Only train the parameters of the last layer (self.fc)
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Apply transformations
        observations = self.resize(observations)
        observations = self.to_three_channels(observations)

        start_time = time.time()
        with torch.no_grad():
            features = self.efficientnet(observations)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"EffNet Time: {elapsed_time:.6f} seconds")
        return self.fc(features)


class CustomCNN_attention(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomCNN_attention, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 24, kernel_size=6, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
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
        self.self_attention = SelfAttention(feature_dim=self.feature_dim, num_heads=2, dropout=0.1)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        start_time = time.time()

        x = self.cnn(observation)
        x = self.reshape(x)
        x = self.self_attention(x)

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Full Feature Extraction Time: {elapsed_time:.6f} seconds")

        x = x.flatten(start_dim=1)  # Flatten back before feeding into linear layer
        return self.linear(x)


class CustomCNN_attentionBIG(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomCNN_attentionBIG, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 24, kernel_size=6, stride=4, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout after activation
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout after activation
            nn.Conv2d(48, 64, kernel_size=2, stride=1, padding=0),
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
        self.feature_dim = n_flatten // 64  # assuming last conv outputs 48 channels
        self.reshape = nn.Unflatten(1, (64, self.feature_dim))

        # Adding the self-attention layer
        self.self_attention = SelfAttention(feature_dim=self.feature_dim, num_heads=8, dropout=0.3)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout after activation
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observation)
        x = self.reshape(x)
        x = self.self_attention(x)
        x = x.flatten(start_dim=1)  # Flatten back before feeding into linear layer
        return self.linear(x)


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=2, dropout=0.2):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout)

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
