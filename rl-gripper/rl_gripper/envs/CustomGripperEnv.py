import gymnasium as gym
import pybullet as p
from gymnasium.spaces import Box
import cv2
import numpy as np
import random
import yaml
import time
import torch
import shutil
import os
import math
import torch as th
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, gaussian_filter
from collections import deque
import matplotlib.pyplot as plt
from rl_gripper.resources.classes.plane import Plane
from rl_gripper.resources.classes.curriculum import Curriculum
from rl_gripper.resources.classes.robot import Robot
from rl_gripper.resources.classes.dataset import Cube, RandomObject, YCB
from rl_gripper.resources.classes.workspace import Workspace
from rl_gripper.resources.functions.helper import load_config, tictoc
from rl_gripper.resources.classes.customFeaturesExtractor import EfficientNetFeatureExtractor, ResNet34FeatureExtractor
from rl_gripper.resources.classes.observationStacker import LatentStacker

config = load_config()

sim_length = config['env']['sim_length']
sr_window_size = config['env']['sr_window_size']

min_gripper_height = config['curriculum']['min_gripper_height']
max_gripper_height = config['curriculum']['max_gripper_height']

min_lifting_height = config['curriculum']['min_lifting_height']
max_lifting_height = config['curriculum']['max_lifting_height']

height = config['camera']['height']
width = config['camera']['width']


class GripperEnv(gym.Env):
    # metadata = {'render_modes': ['GUI', 'DIRECT']
    #             'curriculum_enabled': [True, False]
    #             'dataset': ['CUBE', 'RNG', 'YCB']
    #             'mode': ['TRAIN', 'EVAL', 'TEST']}

    def __init__(self, render_mode='GUI', curriculum_enabled=False, dataset='CUBE', mode='TRAIN', reward_scale=1):

        self.dataset = dataset
        self.mode = mode
        self.object_stats = {}
        self.reward_scale = reward_scale

        self.latent_stacker = LatentStacker(stack_size=4, latent_dim=512)

        # ACTION SPACE
        self.action_space = Box(
            low=np.array([-1, -1, -1, -1, -1], dtype=np.float16),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float16))

        self.observation_space = Box(low=-float(5), high=float(5), shape=(512,), dtype=np.float16)

        self.sim_length = sim_length
        self.last_results = deque(maxlen=sr_window_size)  # Results of the last 50 Episodes for the success rate

        self.sim_steps_per_env_step = 5
        self.time_step = 1/240

        self.feature_extractor = ResNet34FeatureExtractor().to('cuda', non_blocking=True)

        self._seed = self.seed(333)

        self.terminated = False
        self.truncated = False

        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False
        self.dist_to_goal = 100
        self.object_start_height = 0

        self.curriculum = Curriculum(curriculum_enabled)

        if self.curriculum.enabled:
            self.gripper_start_pos = [0.42, 0.0, min_gripper_height]
            self.lifting_height = min_lifting_height
        else:
            self.gripper_start_pos = [0.42, 0.0, max_gripper_height]
            self.lifting_height = max_lifting_height

        self.workspace = Workspace()
        self.workspace.define_workspace(self.curriculum)

        self.client = p.connect(p.GUI if render_mode == 'GUI' else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)


        self.robot = None
        self.plane = None
        self.object = None

    def step(self, action):
        ### ACTION ###
        #print("IN STEP FUNCTION")
        #self.robot.apply_action(action, self.dist_to_goal)
        self.robot.apply_action_jacobian(action)

        for _ in range(self.sim_steps_per_env_step):
            p.stepSimulation(physicsClientId=self.client)

        ### OBSERVATION ###
        obs = self.get_full_observation()

        ### REWARD ###
        tcp_world = self.robot.get_tcp_world()
        goal_xyz = self.object.get_pos()

        self.dist_to_goal = math.sqrt(((tcp_world[0] - goal_xyz[0]) ** 2 +
                                       (tcp_world[1] - goal_xyz[1]) ** 2 +
                                       (tcp_world[2] - goal_xyz[2]) ** 2))

        reward = self.calculate_reward_perception_only()

        self.sim_length -= 1

        if self.sim_length == 0 and self.terminated == False:
            self.last_results.append(0)
            self.object_stats[self.object.model_name]["recent_attempts"].append(0)
            self.terminated = True

        if self.terminated:
            # Calculate moving success rate over the last 50 attempts
            recent_successes = sum(self.object_stats[self.object.model_name]["recent_attempts"])
            recent_attempts = len(self.object_stats[self.object.model_name]["recent_attempts"])
            self.object_stats[self.object.model_name]["obj_sr"] = recent_successes / recent_attempts

        return obs, reward, self.terminated, False, dict()

    def seed(self, seed=None):
        np_random, _ = gym.utils.seeding.np_random(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        return seed

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self.latent_stacker.reset()
        self._seed +=1
        self.seed(self._seed)

        self.sim_length = sim_length
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False

        self.plane = Plane(self.client)
        self.robot = Robot(self.client, self.gripper_start_pos)

        if self.dataset == 'CUBE':
            self.object = Cube(self.client, self.workspace)
        elif self.dataset == 'YCB':
            self.object = YCB(self.client, self.workspace, self.mode)
        elif self.dataset == "RNG":
            self.object = RandomObject(self.client, self.workspace, self.mode)


        if self.object.model_name in self.object_stats:
            self.object_stats[self.object.model_name]['attempts'] += 1
        else:
            self.object_stats[self.object.model_name] = {'attempts': 1,
                                                         'successes': 0,
                                                         'obj_sr': 0.0,
                                                         'recent_attempts': deque(maxlen=20)}

        # Wait 1s, so objects drop to the ground
        for _ in range(240):
            p.stepSimulation(physicsClientId=self.client)

        self.object_start_height = self.object.get_pos()[2]
        # print(self.object_start_height)

        # Observation to start
        obs = self.get_full_observation()

        return obs, dict()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)

    def check_for_collisions(self):
        # p.performCollisionDetection(self.client)
        collision_robot_plane = p.getContactPoints(self.robot.id, self.plane.id, physicsClientId=self.client)
        collision_robot_robot = p.getContactPoints(self.robot.id, self.robot.id, physicsClientId=self.client)
        if len(collision_robot_plane) != 0 or len(collision_robot_robot) != 0:
            self.COLLISION_FLAG = True
            # print("Collision")

    def check_for_grasping(self):
        right_finger_and_cube = p.getContactPoints(self.robot.id, self.object.id, 12, physicsClientId=self.client)
        left_finger_and_cube = p.getContactPoints(self.robot.id, self.object.id, 9, physicsClientId=self.client)
        if len(right_finger_and_cube) != 0 and len(left_finger_and_cube) != 0:
            self.GRASPING_FLAG = True
            # print("Grasping detected")
        else:
            self.GRASPING_FLAG = False


    def calculate_reward_perception_only(self):
        ### SHAPED REWARD PERSONAL ###
        reward = -2  # Time penalty

        self.check_for_grasping()
        self.check_for_collisions()

        if self.COLLISION_FLAG:
            reward -= 90
            self.terminated = True
            self.last_results.append(0)
            self.object_stats[self.object.model_name]["recent_attempts"].append(0)

            print("CRASH")

        if self.GRASPING_FLAG:
            reward += 1

            # Lifting reward
            if self.object.get_pos()[2] > self.object_start_height:
                reward += (self.object.get_pos()[2] - self.object_start_height) * 10

            # Normal-Distribution for goal height: VERIFIZIEREN, könnte OF minimieren
            actual_lifting_height = np.random.normal(self.lifting_height, self.lifting_height/4)

            if self.object.get_pos()[2] > self.object_start_height + actual_lifting_height:
                reward += 300
                self.terminated = True

                self.last_results.append(1)
                self.object_stats[self.object.model_name]["successes"] += 1
                self.object_stats[self.object.model_name]["recent_attempts"].append(1)

                print("DONE")

        return reward * self.reward_scale


    def get_full_observation(self):
        ### Coordinate Observation ###
        # tcp_world = self.robot.get_tcp_world()
        # goal_world = self.object.get_pos()
        # obs = np.array([*tcp_world, *goal_world], dtype=np.float32)

        rgb, depth, mask = self.robot.get_camera_data()

        # proprioceptive_data = self.robot.get_proprioceptive_data() #Joint States
        # state_data = self.robot.get_state() #[X, Y, Z, Yaw, Gw]

        #Filter depth img
        #depth[mask == self.plane.id] = 0.0  #Filter the Plane
        depth[mask == self.robot.id] = 0.0  #Filter the Robot
        #depth_tensor = torch.tensor(depth).permute(2, 0, 1)

        #Filter rgp img OHNE FILTER BESSERE ERGEBNISSE
        # rgb[mask == self.robot.id] = 0.0
        # rgb[mask == self.plane.id] = 0.0

        rgb = self.image_augmentation(rgb)

        rgb = torch.tensor(rgb).permute(2, 0, 1).float().to('cuda', non_blocking=True)  # From (H, W, C) to (C, H, W)
        pre_features = self.feature_extractor(rgb) # Pre-Feature-Extractor (ResNet Model) before RL Pipeline

        # self.visualize_mean_feature_maps(self.feature_extractor, rgb, 'layer4')
        # self.visualize_feature_maps(self.feature_extractor, rgb, 'layer4')
        # self.visualize_grad_cam(self.feature_extractor, rgb, 'layer1')

        # combined_features = np.concatenate((pre_features, proprioceptive_data[0:6]), axis=0)
        return np.array(pre_features, dtype=np.float16) #512 + 6 Tensor (518,1)

        # self.latent_stacker.add_observation(pre_features.astype(np.float16))
        # return self.latent_stacker.get_stacked_obs() #(4, 512)


    @property
    def success_rate(self):
        if not self.last_results:
            return 0
        return sum(self.last_results) / len(self.last_results)

    @staticmethod #Visualize CNN Mean Feature Maps
    def visualize_mean_feature_maps(model, img, layer_name):
        def hook_fn(module, input, output):
            feature_maps.append(output)

        # Register hook to the desired layer
        feature_maps = []
        hook = getattr(model.model, layer_name).register_forward_hook(hook_fn)

        # Pass the image through the model to get the feature maps
        model.forward(img)

        # Remove the hook
        hook.remove()

        # Convert feature maps to numpy array
        feature_maps = feature_maps[0].squeeze(0).cpu().numpy()

        # Sum the feature maps across the channel dimension
        mean_feature_map = np.mean(feature_maps, axis=0)

        # Plot the summed feature map
        plt.figure(figsize=(8, 8))
        plt.imshow(mean_feature_map, cmap='viridis')
        # plt.colorbar()
        # plt.title(f'Mean Feature Map of {layer_name}')
        plt.axis('off')
        plt.show()

    @staticmethod #Visualize CNN Feature Maps
    def visualize_feature_maps(model, img, layer_name):
        def hook_fn(module, input, output):
            feature_maps.append(output)

        # Register hook to the desired layer
        feature_maps = []
        hook = getattr(model.model, layer_name).register_forward_hook(hook_fn)

        # Pass the image through the model to get the feature maps
        model.forward(img)

        # Remove the hook
        hook.remove()

        # Convert feature maps to numpy array
        feature_maps = feature_maps[0].squeeze(0).cpu().numpy()

        # Plot the feature maps
        num_feature_maps = feature_maps.shape[0]
        size = int(np.ceil(np.sqrt(num_feature_maps)))

        fig, axes = plt.subplots(size, size, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(num_feature_maps):
            axes[i].imshow(feature_maps[i], cmap='viridis')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_grad_cam(model, rgb, target_layer_name='layer4'):
        model.eval()

        # Get the target layer
        target_layer = None
        for name, module in model.model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found in the model.")

        # Prepare storage for activations and gradients
        activations = []
        gradients = []

        # Hook for forward pass to get activations
        def forward_hook(module, input, output):
            activations.append(output)

        # Hook for backward pass to get gradients
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Register hooks
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        output = model(rgb)

        # Predicted class (assuming single output)
        predicted_class = output.argmax()

        # Zero gradients
        model.zero_grad()

        # Backward pass with respect to the predicted class
        one_hot = th.zeros_like(output)
        one_hot[0, predicted_class] = 1
        output.backward(gradient=one_hot)

        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()

        # Get the activations and gradients from the hooks
        activation = activations[0].detach()  # Shape: (1, C, H, W)
        gradient = gradients[0].detach()  # Shape: (1, C, H, W)

        # Pool the gradients across the spatial dimensions
        pooled_gradients = th.mean(gradient, dim=[0, 2, 3])  # Shape: (C,)

        # Weight the channels by corresponding gradients
        for i in range(activation.shape[1]):
            activation[:, i, :, :] *= pooled_gradients[i]

        # Create the heatmap
        heatmap = th.mean(activation, dim=1).squeeze()  # Shape: (H, W)
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap -= heatmap.min()
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        # Convert heatmap to numpy
        heatmap = heatmap.cpu().numpy()

        # Resize heatmap to match input image size
        heatmap_resized = cv2.resize(heatmap, (width, height))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Convert input image to numpy array and rescale to 0-255
        input_np = rgb.cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (H, W, C)
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        input_np = (input_np * 255).astype(np.uint8)

        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(input_np, 0.6, heatmap_colored, 0.4, 0)
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def increase_difficulty(self):
        if self.curriculum.laps_counter < self.curriculum.total_laps:
            self.last_results = deque(maxlen=sr_window_size)

            self.workspace.xMin = np.maximum(self.workspace.xMin - (self.workspace.max_workspace_area-self.workspace.min_workspace_area) / 2 / self.curriculum.total_laps,
                                             0.4 - self.workspace.max_workspace_area / 2)
            self.workspace.xMax = np.minimum(self.workspace.xMax + (self.workspace.max_workspace_area-self.workspace.min_workspace_area) / 2 / self.curriculum.total_laps,
                                             0.4 + self.workspace.max_workspace_area / 2)
            self.workspace.yMin = np.maximum(self.workspace.yMin - (self.workspace.max_workspace_area-self.workspace.min_workspace_area) / 2 / self.curriculum.total_laps,
                                             0.0 - self.workspace.max_workspace_area / 2)
            self.workspace.yMax = np.minimum(self.workspace.yMax + (self.workspace.max_workspace_area-self.workspace.min_workspace_area) / 2 / self.curriculum.total_laps,
                                             0.0 + self.workspace.max_workspace_area / 2)

            new_gripper_height = np.minimum(
                self.gripper_start_pos[2] + (max_gripper_height - min_gripper_height) / self.curriculum.total_laps, max_gripper_height)
            self.gripper_start_pos = [self.gripper_start_pos[0], self.gripper_start_pos[1], new_gripper_height]

            new_lifting_height = np.minimum(self.lifting_height + (max_lifting_height - min_lifting_height) / self.curriculum.total_laps,
                                          max_lifting_height)
            self.lifting_height = new_lifting_height

            self.curriculum.laps_counter += 1
            print(f"Curriculum Level {self.curriculum.laps_counter} entered!")

    def increase_dataset(self):
        if self.dataset == "YCB":
            if self.curriculum.laps_counter <= self.curriculum.total_laps:
                difficulties = ['01_easy', '02_medium', '03_hard', '03_hard', '03_hard', '03_hard', '03_hard', '99_big'] #01_very_easy wird in main gesetzt
                source_folder = os.path.join('rl_gripper/resources/models/YCB/urdf_models/', difficulties[self.curriculum.laps_counter])
                active_models_folder = 'rl_gripper/resources/models/YCB/urdf_models/active_models'

                # Delete the active models folder if it exists
                if os.path.exists(active_models_folder):
                    shutil.rmtree(active_models_folder)

                # Copy all files from the source folder to the active models folder
                shutil.copytree(source_folder, active_models_folder, dirs_exist_ok=False)
                print(f"Current YCB-Dataset: {difficulties[self.curriculum.laps_counter]}")

    @staticmethod
    def image_augmentation(image, padding=6, sigma=0.1):
        """
        Apply DrQ-style image augmentation: random shift and bilinear interpolation.

        Args:
            image (numpy.ndarray): Input image array of shape (128, 128, 3)
            padding (int): Number of pixels to pad on each side (default: 6)
            sigma (float): Standard deviation for Gaussian kernel (default: 1.0)

        Returns:
            numpy.ndarray: Augmented image array of shape (128, 128, 3)
        """
        assert image.shape == (128, 128, 3), "Input image must be 128x128x3"

        # Pad the image
        padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='edge')

        # Generate random shifts (floats for interpolation)
        shift_x = np.random.uniform(-padding, padding)
        shift_y = np.random.uniform(-padding, padding)

        # Create meshgrid for interpolation
        x, y = np.meshgrid(np.arange(128), np.arange(128))

        # Apply shifts and adjust for padding
        x = x + shift_x + padding
        y = y + shift_y + padding

        # Flatten coordinates for map_coordinates
        coords = np.array([y.flatten(), x.flatten()])

        # Perform bilinear interpolation for each channel
        augmented = np.stack([
            map_coordinates(padded[:, :, c], coords, order=1, mode='reflect').reshape(128, 128)
            for c in range(3)
        ], axis=-1)

        # Apply Gaussian blur to each channel
        blurred = np.stack([
            gaussian_filter(augmented[:, :, c], sigma=sigma)
            for c in range(3)
        ], axis=-1)

        # fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        # # Plot the original image
        # axes[0].imshow(image)
        # axes[0].set_title("Original Image")
        # axes[0].axis('off')  # Turn off axis
        # # Plot the augmented image
        # axes[1].imshow(augmented)
        # axes[1].set_title("Augmented Image")
        # axes[1].axis('off')  # Turn off axis
        # # Plot the augmented image
        # axes[2].imshow(blurred)
        # axes[2].set_title("Blurred Image")
        # axes[2].axis('off')  # Turn off axis
        #
        #
        # plt.tight_layout()
        # plt.show()

        return blurred.astype(image.dtype)
