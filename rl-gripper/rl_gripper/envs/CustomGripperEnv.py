import gym
import pybullet as p
from gym.spaces import Discrete, Box
import numpy as np
import random
import math

from rl_gripper.resources.classes.cube import Cube
from rl_gripper.resources.classes.plane import Plane
from rl_gripper.resources.classes.robot import Robot


class GripperEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # ACTION SPACE

        self.action_space = Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32))

        # ABSOLUTE POSITION CONTROL
        # self.action_space = Box(
        #     low=np.array([-6.283, -2.059, -3.927, 0.0000], dtype=np.float32),
        #     high=np.array([6.283, 2.094, 0.191, 0.850], dtype=np.float32))

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
        self.sim_length = 512   # Max simulation length 10s??
        self.prev_dist_to_goal = None
        self.np_random, _ = gym.utils.seeding.np_random()
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False

        self.client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        #p.setTimeStep(1/120, self.client)

        self.robot = None
        self.plane = None
        self.cube = None

        self.reset()

    def step(self, action):
        ### ACTION ###
        self.robot.apply_action(action)
        p.stepSimulation()

        ### OBSERVATION ###
        depth, tcp, rgb_flat = self.robot.get_observation()
        obs = depth

        ### REWARD ###
        reward = 0
        self.check_for_collisions()
        if self.COLLISION_FLAG:
            self.terminated = True
            reward -= 1000

        # distance to goal (L2 Norm)
        goal_xyz = self.cube.get_pos()
        dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
                                  (tcp[1] - goal_xyz[1]) ** 2 +
                                  (tcp[2] - goal_xyz[2]) ** 2))
        if dist_to_goal < self.prev_dist_to_goal:
            reward += 4
        else:
            reward -= 4
        self.prev_dist_to_goal = dist_to_goal

        # tcp below z axis
        if tcp[2] < 0.3:
            reward += 2

        # camera facing downwards
        cam_z = p.getLinkState(self.robot.id, 14)[0][2]
        cam_target_z = p.getLinkState(self.robot.id, 15)[0][2]
        if cam_z - cam_target_z > 0.15:
            reward += 2

        # goal or termination
        if dist_to_goal < 0.1:     # Goal achieved (5cm in range)
            self.terminated = True
            reward = 1000
        elif self.sim_length == 0:  # Time over
            self.terminated = True
            reward = -1000

        self.sim_length -= 1

        return obs, reward, self.terminated, False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)

        self.sim_length = 600
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False

        self.plane = Plane(self.client)
        self.robot = Robot(self.client)
        self.cube = Cube(self.client)
        #self.robot.print_joint_info()

        # Observation to start
        depth, tcp, rgb_flat = self.robot.get_observation()
        obs = depth

        goal_xyz = self.cube.get_pos()  # Position of Goal
        self.prev_dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
                                            (tcp[1] - goal_xyz[1]) ** 2 +
                                            (tcp[2] - goal_xyz[2]) ** 2))

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
