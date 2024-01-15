import pybullet as p
import numpy as np
import os
import random


class Cube:
    def __init__(self, client):
        self.client = client
        f_path = "rl_gripper/resources/models/cube.urdf"
        # DEBUG_CUBE self.id = p.loadURDF(f_path, [0.21, 0, 0.025], p.getQuaternionFromEuler([0, 0, 0]))
        # startPosCube = [random.uniform(0.43, 0.63), random.uniform(-.1, .1), 0.03]
        startPosCube = [0.315, 0, 0.03]
        self.id = p.loadURDF(f_path, startPosCube, p.getQuaternionFromEuler([0, 0, 0]))

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return p.getBasePositionAndOrientation(self.id)[0]
