import pybullet as p
import numpy as np
import os
import random
import math
from rl_gripper.resources.models.YCB.urdf_models import models_data as md


class Cube:
    def __init__(self, client, workspace):
        self.client = client
        f_path = "rl_gripper/resources/models/cube.urdf"
        # DEBUG_CUBE self.id = p.loadURDF(f_path, [0.21, 0, 0.025], p.getQuaternionFromEuler([0, 0, 0]))
        self.model_name = 'Cube'
        self.start_pos = self.get_start_pos(workspace)
        self.start_orn = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)], physicsClientId=self.client)
        # self.start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.id = p.loadURDF(f_path, self.start_pos, self.start_orn, physicsClientId=self.client)

    @staticmethod
    def get_start_pos(workspace):
        return [random.uniform(workspace.xMin, workspace.xMax),
                random.uniform(workspace.yMin, workspace.yMax),
                0.02]

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)[0])


#Randrom geometric object
class RandomObject:
    def __init__(self, client, workspace, mode):
        self.client = client
        f_path, model_number = self.generate_path_for_random_object(mode)
        self.model_name = str(model_number)

        self.start_pos = self.get_start_pos(workspace)
        self.start_orn = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)], physicsClientId=self.client)

        self.id = p.loadURDF(f_path, self.start_pos, self.start_orn, globalScaling=1.0, physicsClientId=self.client)

    @staticmethod
    def get_start_pos(workspace):
        return [random.uniform(workspace.xMin, workspace.xMax),
                random.uniform(workspace.yMin, workspace.yMax),
                0.04]

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)[0])

    def generate_path_for_random_object(self, mode):

        # Generiert eine zufällige Zahl zwischen 000 und 99
        if mode == 'TRAIN':
            model_number = random.randint(0, 899)
        elif mode == 'EVAL':
            model_number = random.randint(900, 999)
        elif mode == 'TEST':
            model_number = random.randint(85, 99)

        formatted_number = f"{model_number:03}"

        #Path erstellen
        path = f"{'rl_gripper/resources/models/random_objects/'}{formatted_number}{'/'}{formatted_number}{'.urdf'}"
        return path, model_number


#Daily supplies, partly from ycb
class YCB:
    def __init__(self, client, workspace, mode):
        self.client = client
        self.mode = mode

        self.id, self.model_name = self.load_urdf(self.get_start_pos(workspace), p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)], physicsClientId=self.client))
        #print(self.get_height())

    def get_height(self):
        aabb = p.getAABB(self.id, physicsClientId=self.client)
        # Extract the min and max z-values
        min_z = aabb[0][2]  # The z-value of the minimum corner
        max_z = aabb[1][2]  # The z-value of the maximum corner

        # Calculate the z-size (height) of the object
        return max_z - min_z

    @staticmethod
    def get_start_pos(workspace):
        return [random.uniform(workspace.xMin, workspace.xMax),
                random.uniform(workspace.yMin, workspace.yMax),
                0.04]

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)[0])

    def load_urdf(self, pos, orn):
        models_lib = md.model_lib(self.mode)
        f_path, model_name = models_lib.random
        tmp_id= p.loadURDF(f_path, pos, orn, globalScaling=0.9, physicsClientId=self.client)

        return tmp_id, model_name