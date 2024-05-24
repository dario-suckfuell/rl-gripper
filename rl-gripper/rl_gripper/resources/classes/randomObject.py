import pybullet as p
import random
import math


class RandomObject:
    def __init__(self, client, workspace, dataset):
        self.client = client
        f_path = self.generate_path_for_random_object(dataset)

        self.start_pos = self.get_start_pos(workspace)
        self.start_orn = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])

        self.id = p.loadURDF(f_path, self.start_pos, self.start_orn, globalScaling=0.8)

    @staticmethod
    def get_start_pos(workspace):
        return [random.uniform(workspace.xMin, workspace.xMax),
                random.uniform(workspace.yMin, workspace.yMax),
                0.04]

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id)[0])

    def generate_path_for_random_object(self, dataset):

        # Generiert eine zufällige Zahl zwischen 000 und 999
        if dataset == 'TRAINING':
            number = random.randint(0, 69)
        elif dataset == 'VALIDATION':
            number = random.randint(70, 84)
        elif dataset == 'TEST':
            number = random.randint(85, 99)


        formatted_number = f"{number:03}"

        #Path erstellen
        path = f"{'rl_gripper/resources/models/random_objects/'}{formatted_number}{'/'}{formatted_number}{'.urdf'}"
        return path

