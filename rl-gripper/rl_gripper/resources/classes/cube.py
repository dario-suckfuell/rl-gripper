import pybullet as p
import numpy as np
import os
import random


class Cube:
    def __init__(self, client, cube_position='RANDOM'):
        self.client = client
        f_path = "rl_gripper/resources/models/cube.urdf"
        # DEBUG_CUBE self.id = p.loadURDF(f_path, [0.21, 0, 0.025], p.getQuaternionFromEuler([0, 0, 0]))

        if cube_position == 'FIX':
            startPosCube = [0.35, 0, 0.02]
        elif cube_position == 'RANDOM':
            startPosCube = [random.uniform(0.3, 0.5), random.uniform(-.15, .15), 0.02]
            # startPosCube = [random.uniform(0.4, 0.5), random.uniform(-.05, .05), 0.02]   #Kleiner Radius

        self.id = p.loadURDF(f_path, startPosCube, p.getQuaternionFromEuler([0, 0, 0]))

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id)[0])

