import pybullet as p
import numpy as np
import os
import random


class Cube:
    def __init__(self, client, workspace):
        self.client = client
        f_path = "rl_gripper/resources/models/cube.urdf"
        # DEBUG_CUBE self.id = p.loadURDF(f_path, [0.21, 0, 0.025], p.getQuaternionFromEuler([0, 0, 0]))

        self.start_pos = self.get_start_pos(workspace)
        self.id = p.loadURDF(f_path, self.start_pos, p.getQuaternionFromEuler([0, 0, 0]))

    @staticmethod
    def get_start_pos(workspace):
        return [random.uniform(workspace.xMin, workspace.xMax),
                random.uniform(workspace.yMin, workspace.yMax),
                0.02]

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id)[0])

