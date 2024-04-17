import gymnasium as gym
import pybullet as p
from gymnasium.spaces import Box
import numpy as np
import random
import math
from collections import deque

from rl_gripper.resources.classes.cube import Cube
from rl_gripper.resources.classes.plane import Plane
from rl_gripper.resources.classes.robot import Robot

sim_length = 82
curr_laps = 10


class GripperEnv(gym.Env):
    # metadata = {'render_modes': ['GUI', 'DIRECT']
    #            'cube_position' ['FIX', 'RANDOM']}

    def __init__(self, cube_position='FIX', render_mode='GUI'):

        # ACTION SPACE
        self.action_space = Box(
            low=np.array([-1, -1, -1], dtype=np.float16),
            high=np.array([1, 1, 1], dtype=np.float16))


        # OBSERVATION SPACE (Greyscale Depth Image Input, Later also Gripper Width)
        #self.observation_space = Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)

        # OBSERVATION SPACE (tcp_x, tcp_y, tcp_z, goal_x, goal_y, goal_z)
        self.observation_space = Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float16),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float16))

        self.sim_length = sim_length
        self.prev_dist_to_goal = 100
        self.np_random, _ = gym.utils.seeding.np_random()
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False
        self.dist_to_goal = 100

        #CURRICULUM
        self.cube_position = cube_position

        self.minGripperHeight = 0.15
        self.maxGripperHeight = 0.40
        self.gripper_start_pos = [0.35, 0, 0.15]

        self.last_results = deque(maxlen=10) #Results of the last 10 Episodes

        if render_mode == 'GUI':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240, self.client)

        self.robot = None
        self.plane = None
        self.cube = None

        # self.reset()

    def step(self, action):
        # print("IN STEP FUNCTION:", action)
        ### ACTION ###
        self.robot.apply_action_xyz(action, self.dist_to_goal)
        p.stepSimulation()

        ### OBSERVATION ###
        obs = self.get_full_observation()

        ### REWARD ###
        tcp_world = self.robot.get_tcp_world()
        #reward = self.calculate_reward_thesis()
        reward = self.calculate_reward_mlp(tcp_world)

        self.sim_length -= 1

        if self.sim_length == 0:
            self.last_results.append(0)
            self.terminated = True

        return obs, reward, self.terminated, False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)

        self.sim_length = sim_length
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False

        self.plane = Plane(self.client)
        self.robot = Robot(self.client, self.gripper_start_pos)
        self.cube = Cube(self.client, self.cube_position)

        # Observation to start
        obs = self.get_full_observation()

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
        ### SHAPED REWARD PERSONAL ###
        reward = -3  # Time penalty

        self.check_for_grasping()
        self.check_for_collisions()

        if self.COLLISION_FLAG:
            reward -= 200
            self.last_results.append(0)
            self.terminated = True

        # distance to goal (L2 Norm) NICHT OPTIMAL DA CUBE POS NOTWENDIG
        goal_xyz = self.cube.get_pos()
        self.dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
                                       (tcp[1] - goal_xyz[1]) ** 2 +
                                       (tcp[2] - goal_xyz[2]) ** 2))

        # reward -= 10 * self.dist_to_goal

        if self.dist_to_goal < 0.04:
            reward += 1
            # print("CUBE Z:", self.cube.get_pos()[2])
            # print("GRASPING!")
            if self.GRASPING_FLAG:
                reward += 1
                # over starting high of 2cm
                if self.cube.get_pos()[2] > 0.02:
                    reward += (self.cube.get_pos()[2] - 0.02) * 10

                # Goal, über 9cm
                if self.cube.get_pos()[2] > 0.09:
                    reward += 200
                    self.terminated = True
                    self.last_results.append(1)

        return reward

    def calculate_reward_mlp(self, tcp):
        ### SHAPED REWARD PERSONAL ###
        reward = -2  # Time penalty

        self.check_for_grasping()
        self.check_for_collisions()

        if self.COLLISION_FLAG:
            reward -= 200
            self.terminated = True
            self.last_results.append(0)


        # distance to goal (L2 Norm) NICHT OPTIMAL DA CUBE POS NOTWENDIG
        goal_xyz = self.cube.get_pos()
        self.dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
                                       (tcp[1] - goal_xyz[1]) ** 2 +
                                       (tcp[2] - goal_xyz[2]) ** 2))

        reward -= 10 * self.dist_to_goal

        if self.dist_to_goal < 0.04:
            reward += 1
            # print("CUBE Z:", self.cube.get_pos()[2])
            # print("GRASPING!")
            if self.GRASPING_FLAG:
                #reward += 1
                # over starting high of 2cm
                if self.cube.get_pos()[2] > 0.02:
                    reward += (self.cube.get_pos()[2] - 0.02) * 10

                # Goal, über 9cm
                if self.cube.get_pos()[2] > 0.09:
                    reward += 200
                    self.last_results.append(1)
                    self.terminated = True


        return reward

    def calculate_reward_thesis(self):
        ### SHAPED REWARD THESIS ###
        reward = -200  # Time penalty

        self.check_for_grasping()
        self.check_for_collisions()

        if self.COLLISION_FLAG:
            reward -= 10000
            self.last_results.append(0)
            self.terminated = True

        # distance to goal (L2 Norm) NICHT OPTIMAL DA CUBE POS NOTWENDIG

        if self.GRASPING_FLAG:
            reward += 100
            # print("CUBE Z:", self.cube.get_pos()[2])

            # over starting high of 2cm
            if self.cube.get_pos()[2] > 0.02:
                reward += (self.cube.get_pos()[2] - 0.02) * 1000

            # Goal, über 9cm
            if self.cube.get_pos()[2] > 0.09:
                reward += 10000
                self.terminated = True
                self.last_results.append(1)

        return reward

    def get_full_observation(self):
        tcp_world = self.robot.get_tcp_world()
        goal_world = self.cube.get_pos()
        obs = np.array([*tcp_world, *goal_world], dtype=np.float32)
        return obs

    @property
    def success_rate(self):
        if not self.last_results:
            return 0
        return sum(self.last_results) / len(self.last_results)

    def increase_difficulty(self):

        newGripperHeight = np.clip(self.gripper_start_pos[2] + (self.maxGripperHeight - self.minGripperHeight) / curr_laps, self.minGripperHeight, self.maxGripperHeight)
        self.gripper_start_pos = [0.35, 0, newGripperHeight]

