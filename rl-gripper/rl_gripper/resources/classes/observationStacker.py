import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

class LatentStacker:
    def __init__(self, stack_size=4, latent_dim=512):
        self.stack_size = stack_size
        self.latent_dim = latent_dim
        self.obs_dim = latent_dim
        self.frames = deque(maxlen=stack_size)

    def reset(self):
        self.frames.clear()
        # Fill with zero-padded frames
        for _ in range(self.stack_size):
            self.frames.append(np.zeros(self.obs_dim, dtype=np.float16))

    def add_observation(self, latent_obs):
        self.frames.append(latent_obs)

    def get_stacked_obs(self):
        return np.array(self.frames)  # Shape: (4, 518)

    def visualize(self):
        # plotting the heatmap
        hm = sns.heatmap(data=np.array(self.frames)[:,500:],
                         annot=True)

        # displaying the plotted heatmap
        plt.show()


class LatentStackerWithProprio:
    def __init__(self, stack_size=4, latent_dim=512, proprio_dim=6):
        self.stack_size = stack_size
        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.obs_dim = latent_dim + proprio_dim
        self.frames = deque(maxlen=stack_size)

    def reset(self):
        self.frames.clear()
        # Fill with zero-padded frames
        for _ in range(self.stack_size):
            self.frames.append(np.zeros(self.obs_dim, dtype=np.float16))

    def add_observation(self, latent_obs, proprio_obs):
        combined_obs = np.concatenate([latent_obs, proprio_obs])
        self.frames.append(combined_obs)

    def get_stacked_obs(self):
        return np.array(self.frames)  # Shape: (4, 518)

    def visualize(self):
        # plotting the heatmap
        hm = sns.heatmap(data=np.array(self.frames)[:,500:],
                         annot=True)

        # displaying the plotted heatmap
        plt.show()
