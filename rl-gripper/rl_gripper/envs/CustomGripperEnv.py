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
from rl_gripper.resources.classes.randomObject import RandomObject
from rl_gripper.resources.classes.workspace import Workspace
from rl_gripper.resources.functions.helper import load_config

sim_length = 82
curr_laps = 6
window_size = 50

height, width = 48, 48

min_gripper_height = 0.09
max_gripper_height = 0.25

min_picking_height = 0.01
max_picking_height = 0.1

class GripperEnv(gym.Env):
    # metadata = {'render_modes': ['GUI', 'DIRECT']
    #            'cube_position': ['FIX', 'RANDOM']
    #            'curriculum': [True, False]}

    def __init__(self, cube_position='FIX', render_mode='GUI', curriculum=False):

        # ACTION SPACE
        self.action_space = Box(
            low=np.array([-1, -1, -1], dtype=np.float16),
            high=np.array([1, 1, 1], dtype=np.float16))


        # OBSERVATION SPACE (Greyscale Depth Image Input, Later also Gripper Width)
        self.observation_space = Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

        # OBSERVATION SPACE (tcp_x, tcp_y, tcp_z, goal_x, goal_y, goal_z)
        # self.observation_space = Box(
        #     low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float16),
        #     high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float16))

        self.sim_length = sim_length
        self.np_random, _ = gym.utils.seeding.np_random()
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False
        self.dist_to_goal = 100

        self.last_results = deque(maxlen=window_size)  # Results of the last 50 Episodes
        self.curr_counter = 0

        if curriculum:
            self.gripper_start_pos = [0.42, 0.0, min_gripper_height]
            self.picking_height = min_picking_height
        else:
            self.gripper_start_pos = [0.42, 0.0, max_gripper_height]
            self.picking_height = max_picking_height

        self.workspace = Workspace()
        self.workspace.define_workspace(cube_position, curriculum)

        self.client = p.connect(p.GUI if render_mode == 'GUI' else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240, self.client)

        self.robot = None
        self.plane = None
        self.cube = None
        #self.randomObject = None

    def step(self, action):
        ### ACTION ###
        self.robot.apply_action_xyz(action, self.dist_to_goal)
        p.stepSimulation()

        ### OBSERVATION ###
        obs = self.get_full_observation()

        ### REWARD ###
        tcp_world = self.robot.get_tcp_world()
        reward = self.calculate_reward_full(tcp_world)

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
        p.setGravity(0, 0, -9.81)

        self.sim_length = sim_length
        self.terminated = False
        self.truncated = False
        self.COLLISION_FLAG = False
        self.GRASPING_FLAG = False

        self.plane = Plane(self.client)
        self.robot = Robot(self.client, self.gripper_start_pos)
        self.cube = Cube(self.client, self.workspace)
        #self.cube = RandomObject(self.client, self.workspace)

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

    def calculate_reward_simple(self, tcp):
        ### SHAPED REWARD PERSONAL ###
        reward = -1  # Time penalty

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

        reward -= 10 * self.dist_to_goal

        if self.dist_to_goal < 0.04:
            reward += 200
            self.last_results.append(1)
            self.terminated = True

        return reward

    def calculate_reward_full(self, tcp):
        ### SHAPED REWARD PERSONAL ###
        reward = -2.4  # Time penalty

        self.check_for_grasping()
        self.check_for_collisions()

        if self.COLLISION_FLAG:
            reward -= 200
            self.terminated = True
            self.last_results.append(0)
            print("CRASH")

        # distance to goal (L2 Norm) NICHT OPTIMAL DA CUBE POS NOTWENDIG
        goal_xyz = self.cube.get_pos()
        self.dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
                                       (tcp[1] - goal_xyz[1]) ** 2 +
                                       (tcp[2] - goal_xyz[2]) ** 2))

        reward -= 3 * self.dist_to_goal

        if self.dist_to_goal < 0.04:
            #print("FINDING THE CUBE")
            reward += 0.4
            # print("CUBE Z:", self.cube.get_pos()[2])
            # print("GRASPING!")
            if self.GRASPING_FLAG:
                #print("GRASPING")
                reward += 0.4

                # Lifting reward
                if self.cube.get_pos()[2] > 0.02:
                    reward += (self.cube.get_pos()[2] - 0.02) * 20

                # Terminal state bei self.picking_height
                if self.cube.get_pos()[2] > 0.02 + self.picking_height:
                    reward += 300
                    self.last_results.append(1)
                    self.terminated = True
                    print("DONE")

        return reward

    def calculate_reward_gpt(self, tcp):
        ### SHAPED REWARD PERSONAL ###
        reward = -1  # Moderately reduced time penalty to encourage efficiency

        self.check_for_grasping()
        self.check_for_collisions()

        if self.COLLISION_FLAG:
            reward -= 100  # Reduced collision penalty to balance with the benefits of successful behavior
            self.terminated = True
            self.last_results.append(0)
            print("CRASH")

        # Calculate distance to goal
        goal_xyz = self.cube.get_pos()
        self.dist_to_goal = math.sqrt(((tcp[0] - goal_xyz[0]) ** 2 +
                                       (tcp[1] - goal_xyz[1]) ** 2 +
                                       (tcp[2] - goal_xyz[2]) ** 2))

        reward -= 5 * self.dist_to_goal  # Reduced scaling factor for distance to goal

        if self.dist_to_goal < 0.04:
            reward += 1  # Increase reward for proximity
            if self.GRASPING_FLAG:
                reward += 1  # Increase reward for grasping
                # Incremental reward for lifting higher
                lifting_reward = max(0, (self.cube.get_pos()[2] - 0.02) * 50)
                reward += lifting_reward

                # Major reward for reaching target height
                if self.cube.get_pos()[2] > 0.02 + self.picking_height:
                    reward += 300  # Increased terminal reward to significantly incentivize reaching this goal
                    self.last_results.append(1)
                    self.terminated = True
                    print("DONE")

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
            if self.cube.get_pos()[2] > 0.02 + self.picking_height:
                reward += 10000
                self.terminated = True
                self.last_results.append(1)

        return reward

    def get_full_observation(self):
        ### Coordinate Observation ###
        # tcp_world = self.robot.get_tcp_world()
        # goal_world = self.cube.get_pos()
        # obs = np.array([*tcp_world, *goal_world], dtype=np.float32)

        rgb, depth, mask = self.robot.get_camera_data()

        #Filter depth img
        #depth[mask == self.plane.id] = 0.0  #Filter the Plane
        depth[mask == self.robot.id] = 0.0  #Filter the Robot

        obs = depth

        return obs

    @property
    def success_rate(self):
        if not self.last_results:
            return 0
        return sum(self.last_results) / len(self.last_results)

    def increase_difficulty(self):
        if self.curr_counter < curr_laps:
            print("New curriculum stage!")
            self.last_results = deque(maxlen=window_size)

            self.workspace.xMin = np.maximum(self.workspace.xMin - self.workspace.max_workspace_area / 2 / curr_laps, 0.4 - self.workspace.max_workspace_area/2)
            self.workspace.xMax = np.minimum(self.workspace.xMax + self.workspace.max_workspace_area / 2 / curr_laps, 0.4 + self.workspace.max_workspace_area/2)
            self.workspace.yMin = np.maximum(self.workspace.yMin - self.workspace.max_workspace_area / 2 / curr_laps, 0.0 - self.workspace.max_workspace_area/2)
            self.workspace.yMax = np.minimum(self.workspace.yMax + self.workspace.max_workspace_area / 2 / curr_laps, 0.0 + self.workspace.max_workspace_area/2)

            newGripperHeight = np.minimum(self.gripper_start_pos[2] + (max_gripper_height - min_gripper_height)/curr_laps, max_gripper_height)
            self.gripper_start_pos = [self.gripper_start_pos[0], self.gripper_start_pos[1], newGripperHeight]

            newPickingHeight = np.minimum(self.picking_height + (max_picking_height - min_picking_height)/curr_laps, max_picking_height)
            self.picking_height = newPickingHeight

            self.curr_counter += 1





