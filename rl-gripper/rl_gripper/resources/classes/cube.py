import pybullet as p
import numpy as np
import os
import random


class Cube:
    def __init__(self, client, cube_start_pos):
        self.client = client
        f_path = "rl_gripper/resources/models/cube.urdf"
        # DEBUG_CUBE self.id = p.loadURDF(f_path, [0.21, 0, 0.025], p.getQuaternionFromEuler([0, 0, 0]))
        self.id = p.loadURDF(f_path, cube_start_pos, p.getQuaternionFromEuler([0, 0, 0]))

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id)[0])

