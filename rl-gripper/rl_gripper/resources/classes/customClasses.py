import torch
import random
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

### VORTRAINIERTES RESNET VERWENDEN ###
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
