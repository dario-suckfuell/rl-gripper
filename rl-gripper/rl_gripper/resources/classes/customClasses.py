import torch
import random
import torch.nn as nn
#from torchvision.models import resnet18, ResNet18_Weights  #Macht irgendeinen komischen fehler! bzw eine torch version

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback


class CustomCNN(BaseFeaturesExtractor):
    # Custom Feature Extractor: Custom CNN
    def __int__(self, observation_space, features_dim):
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

        self.fc = nn.Sequential(
            nn.Linear(self.features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observation):
        return self.fc(self.cnn(observation))


### VORTRAINIERTES RESNET ###
class CustomResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        super(CustomResNetFeatureExtractor, self).__init__(observation_space, features_dim)

        # Preprocessing layer to convert 1-channel images to 3-channel
        self.preprocess = nn.Conv2d(1, 3, kernel_size=1)

        # Load a pre-trained ResNet model
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace the last fully connected layer to match the desired feature size
        num_features_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features_in, features_dim)

    def forward(self, observations):
        # Apply the preprocessing layer
        observations = self.preprocess(observations)

        # Pass through the ResNet model
        return self.resnet(observations)


class TensorboardCallback(BaseCallback):
    # Custom callback for plotting additional values in tensorboard.
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional stats
        value = random.randint(-1, 1)
        self.logger.record('Random Value', value)
        print(locals())

        # if 'approx_kl' in self.locals:
        #     approx_kl = self.locals['approx_kl']
        #     self.logger.record('Approx_kl', approx_kl)

        return True
