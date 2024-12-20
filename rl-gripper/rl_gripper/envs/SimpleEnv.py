import gymnasium as gym
import pybullet as p
from gymnasium.spaces import Box
import numpy as np
import random
import math

from rl_gripper.resources.classes.cube import Cube
from rl_gripper.resources.classes.plane import Plane
from rl_gripper.resources.classes.robot import Robot

sim_length = 256

class GripperEnv(gym.Env):
    #metadata = {'render_modes': ['GUI', 'DIRECT']}

    def __init__(self, render_mode='GUI'):
        # ACTION SPACE

        self.action_space = Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float16),
            high=np.array([1, 1, 1, 1], dtype=np.float16))

        # OBSERVATION SPACE (Greyscale Depth Image Input, Later also Gripper Width)
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
        # self.observation_space = Box(
        #     low=np.array([-100, -100, -100, -100, -100, -100], dtype=np.float32),
        #     high=np.array([100, 100, 100, 100, 100, 100], dtype=np.float32))

        # MULTIPLE FEATURES AS OBSERVATION... RL ALGORITHM HAS TO SUPPORT DICTS/TUPELS
        # spaces = {
        #     'depth': Box(low=np.array(0), high=np.array(255), shape=(im_width, im_height, 1), dtype=np.uint8),
        #     'gripper_width': Box(np.array([0, 1]), dtype=np.float32)
        # }
        # dict_space = gym.spaces.Dict(spaces)

        # self.state = [0, 0, 0, 0]   # [Yaw, Joint2, Joint3, Gripper]
        self.sim_length = sim_length  # ALSO IN RESET() !!!
        self.prev_dist_to_goal = None
        self.np_random, _ = gym.utils.seeding.np_random()
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False

        if render_mode == 'GUI':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        # p.setTimeStep(1/240, self.client)

        self.robot = None
        self.plane = None
        self.cube = None

        # self.reset()

    def step(self, action):
        #print("IN STEP FUNCTION:", action)
        ### ACTION ###
        self.robot.apply_action_xyz(action)
        p.stepSimulation()

        ### OBSERVATION ###
        depth, tcp, rgb_flat = self.robot.get_observation()
        obs = depth

        ### REWARD ###
        reward = self.calculate_reward(depth, tcp, rgb_flat)

        self.sim_length -= 1
        # print(self.sim_length)
        # if self.sim_length == 27:
        #     test = 1
        if self.sim_length == 0:
            self.terminated = True

        return obs, reward, self.terminated, False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)

        self.sim_length = sim_length
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False

        self.plane = Plane(self.client)
        self.robot = Robot(self.client)
        self.cube = Cube(self.client)
        # self.robot.print_joint_info()

        # Observation to start
        depth, tcp, rgb_flat = self.robot.get_observation()
        obs = depth

        # goal_xyz = self.cube.get_pos()  # Position of Goal
        # self.prev_dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
        #                                     (tcp[1] - goal_xyz[1]) ** 2 +
        #                                     (tcp[2] - goal_xyz[2]) ** 2))

        return obs, dict()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)

    def check_for_collisions(self):
        # p.performCollisionDetection(self.client)
        collision_robot_plane = p.getContactPoints(self.robot.id, self.plane.id)
        collision_robot_robot = p.getContactPoints(self.robot.id, self.robot.id)
        if len(collision_robot_plane) != 0 or len(collision_robot_robot) != 0:
            self.COLLISION_FLAG = True
            # print("Collision")

    def check_for_grasping(self):
        right_finger_and_cube = p.getContactPoints(self.robot.id, self.cube.id, 12)
        left_finger_and_cube = p.getContactPoints(self.robot.id, self.cube.id, 9)
        if len(right_finger_and_cube) != 0 and len(left_finger_and_cube) != 0:
            self.GRASPING_FLAG = True
            # print("Grasping detected")
        else:
            self.GRASPING_FLAG = False

    def calculate_reward(self, depth, tcp, rgb_flat):
        ### SHAPED REWARD ###
        reward = -2.0  # Time penalty

        self.check_for_grasping()
        self.check_for_collisions()

        if self.COLLISION_FLAG:
            reward -= 200
            self.terminated = True

        # REWARDING GREEN PIXELS
        # green_values = np.array(rgb_flat[1::4], dtype=np.int16)
        # blue_values = np.array(rgb_flat[2::4], dtype=np.int16)
        # true_green = green_values - blue_values  # otherwise white will also trigger the reward since it is [255, 255, 255]
        # true_green_count = len([pixel for pixel in true_green if pixel > 150])
        # # print("GreenPixelCount: {}".format(true_green_count))
        # # reward += np.clip(math.ceil(true_green_count / 50), 0, 49)
        # reward += np.clip(true_green_count / 50, 0, 0.9)
        # # print("GreenReward: {}".format(math.ceil(true_green_count / 10)))


        ''' # # tcp below z axis
        # if tcp[2] < 0.3:
        #     reward += 1
        #     # print("TCP below 30cm")

        # camera facing downwards
        # cam_z = p.getLinkState(self.robot.id, 14)[0][2]
        # cam_target_z = p.getLinkState(self.robot.id, 15)[0][2]
        # if cam_z - cam_target_z > 0.17:
        #     reward += 2
        # else:
        #     reward -= 2
        #     # print("Camera facing downwards")
        '''

        # distance to goal (L2 Norm) NICHT OPTIMAL DA CUBE POS NOTWENDIG
        goal_xyz = self.cube.get_pos()
        dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
                                  (tcp[1] - goal_xyz[1]) ** 2 +
                                  (tcp[2] - goal_xyz[2]) ** 2))
        if dist_to_goal < self.prev_dist_to_goal:
            reward += 1
        else:
            reward -= -1

        self.prev_dist_to_goal = dist_to_goal

        if dist_to_goal < 0.07:
            reward += 200
            self.terminated = True


        return reward
