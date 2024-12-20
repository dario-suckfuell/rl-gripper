import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from networkx import communicability_exp
from torch.nn.functional import layer_norm
from torchvision.models import resnet, resnet18, resnet34
from torchvision.models import ResNet34_Weights
from gymnasium import spaces
from rl_gripper.resources.classes.attention_layer import MultiHeadProjection2L, MultiHeadProjection
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np
import time
import math
import torch.nn.functional as F

class CustomAdapterNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512): # Total Features
        super(CustomAdapterNet, self).__init__(observation_space, features_dim)

        # Assuming observation_space is already flattened
        self.feedforward = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),  # Dropout added here
            nn.Linear(features_dim * 2, features_dim),
            nn.GELU()
        )
        self.layer_norm1 = nn.LayerNorm(512)


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split the observation tensor
        # resnet_features = observations[:, :512]
        # proprioceptive_data = observations[:, 512:517]

        # Pass the Resnet features through the adapter network
        x = self.feedforward(observations)
        x = self.layer_norm1(x)

        # Concatenate the transformed features with the proprioceptive data
        # combined_features = torch.cat((x, proprioceptive_data), dim=1)

        return x

class AdapterNetworkMHP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512,
                 num_heads: int = 4):  # Total Features
        super(AdapterNetworkMHP, self).__init__(observation_space, features_dim)
        self.num_heads = num_heads
        self.attention = MultiHeadProjection(embed_dim=features_dim, num_heads=num_heads)

        # Optional: add a position-wise feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),  # Dropout added here
            nn.Linear(features_dim * 2, features_dim),
            nn.GELU()
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)

        # Define the dropout layer for attention output
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, observations):
        # Split the observation tensor
        resnet_features = observations[:, :512]  # [bs, embedd_dim]
        # proprioceptive_data = observations[:, 512:518]

        # Reshape input: (batch_size, feature_dim) -> (1, batch_size, feature_dim)
        # resnet_features = resnet_features.unsqueeze(0)

        # Self-attention
        attn_output, _ = self.attention(resnet_features, resnet_features, resnet_features)
        attn_output = self.dropout(attn_output)
        resnet_features = resnet_features + attn_output.squeeze(1)  # Residual connection
        resnet_features = self.layer_norm1(resnet_features)

        # Feedforward
        ff_output = self.feedforward(resnet_features)
        resnet_features = resnet_features + ff_output  # Residual connection
        resnet_features = self.layer_norm2(resnet_features)

        # Reshape output: (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        resnet_features = resnet_features.squeeze(0)

        # combined_features = torch.cat((resnet_features, proprioceptive_data), dim=1)

        return resnet_features


class AdapterNetwork2L(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512,
                 num_heads: int = 8):  # Total Features
        super(AdapterNetwork2L, self).__init__(observation_space, features_dim)
        self.num_heads = num_heads
        self.attention = MultiHeadProjection2L(embed_dim=features_dim, num_heads=num_heads)

        # Optional: add a position-wise feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),  # Dropout added here
            nn.Linear(features_dim * 2, features_dim),
            nn.GELU()
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)

        # Define the dropout layer for attention output
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, observations):
        # Split the observation tensor
        resnet_features = observations[:, :512]  # [bs, embedd_dim]
        # proprioceptive_data = observations[:, 512:518]

        # Reshape input: (batch_size, feature_dim) -> (1, batch_size, feature_dim)
        # resnet_features = resnet_features.unsqueeze(0)

        # Self-attention
        attn_output, _ = self.attention(resnet_features, resnet_features, resnet_features)
        attn_output = self.dropout(attn_output)
        resnet_features = resnet_features + attn_output.squeeze(1)  # Residual connection
        resnet_features = self.layer_norm1(resnet_features)

        # Feedforward
        ff_output = self.feedforward(resnet_features)
        resnet_features = resnet_features + ff_output  # Residual connection
        resnet_features = self.layer_norm2(resnet_features)

        # Reshape output: (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        resnet_features = resnet_features.squeeze(0)

        # combined_features = torch.cat((resnet_features, proprioceptive_data), dim=1)

        return resnet_features

class AttentionAdapter(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512,
                 num_heads: int = 8):  # Total Features
        super(AttentionAdapter, self).__init__(observation_space, features_dim)
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=features_dim, num_heads=num_heads)

        # Optional: add a position-wise feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(features_dim, features_dim*2),
            nn.GELU(),
            nn.Dropout(p=0.2),  # Dropout added here
            nn.Linear(features_dim*2, features_dim),
            nn.GELU()
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)

        # Define the dropout layer for attention output
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, observations):
        # Split the observation tensor
        resnet_features = observations[:, :512]
        # proprioceptive_data = observations[:, 512:518]

        # Reshape input: (batch_size, feature_dim) -> (1, batch_size, feature_dim)
        resnet_features = resnet_features.unsqueeze(0)

        # Self-attention
        attn_output, _ = self.attention(resnet_features, resnet_features, resnet_features)
        attn_output = self.dropout(attn_output)
        resnet_features = resnet_features + attn_output  # Residual connection
        resnet_features = self.layer_norm1(resnet_features)

        # Feedforward
        ff_output = self.feedforward(resnet_features)
        resnet_features = resnet_features + ff_output  # Residual connection
        resnet_features = self.layer_norm2(resnet_features)

        # Reshape output: (1, batch_size, feature_dim) -> (batch_size, feature_dim)
        resnet_features = resnet_features.squeeze(0)

        # combined_features = torch.cat((resnet_features, proprioceptive_data), dim=1)

        return resnet_features


class SimbaAdapter(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512):  # Total Features
        super(SimbaAdapter, self).__init__(observation_space, features_dim)
        self.SimbaBlock = nn.Sequential(
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim, bias=True),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim, bias=True),
        )
        self.layer_norm1 = nn.LayerNorm(512)

    def forward(self, x):
        residual = x
        x = self.SimbaBlock(x)
        x = self.layer_norm1(residual + x)
        return x


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


class ResNet34FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet34FeatureExtractor, self).__init__()

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load ResNet34 and remove the final fully connected layer
        self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove the final fully connected layer

        # Freeze ResNet34 parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img):
        # Preprocess
        img_tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            features = self.model(img_tensor)

        return features.squeeze(0).cpu().numpy()  # 512 out


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()

        # Define the transformations
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.to_three_channels = transforms.Lambda(lambda x: x.expand(3, -1, -1))
        self.normalize = transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                              std=[0.229, 0.229, 0.229])

        # Load EfficientNet-B0 and remove the final fully connected layer
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Identity()  # Remove the final fully connected layer

        # Freeze EfficientNet parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img: np.ndarray) -> torch.Tensor:
        # Convert numpy array to torch tensor and reshape

        img_tensor = torch.tensor(img).permute(2, 0, 1).float()  # From (48, 48, 1) to (1, 48, 48)

        # Apply transformations
        img_tensor = self.resize(img_tensor)
        img_tensor = self.to_three_channels(img_tensor)
        img_tensor = self.normalize(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = self.model(img_tensor)

        return features.squeeze(0).cpu().numpy()  # Convert to 1280 numpy array



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

