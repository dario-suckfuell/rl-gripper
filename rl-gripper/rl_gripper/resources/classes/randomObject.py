import pybullet as p
import random


class RandomObject:
    def __init__(self, client, workspace):
        self.client = client
        f_path = self.generate_path_for_random_object()

        self.start_pos = self.get_start_pos(workspace)
        self.id = p.loadURDF(f_path, self.start_pos, p.getQuaternionFromEuler([0, 0, 0]))

    @staticmethod
    def get_start_pos(workspace):
        return [random.uniform(workspace.xMin, workspace.xMax),
                random.uniform(workspace.yMin, workspace.yMax),
                0.04]

    def get_ids(self):
        return self.client, self.id

    def get_pos(self):
        return list(p.getBasePositionAndOrientation(self.id)[0])

    def generate_path_for_random_object(self):

        # Generiert eine zufällige Zahl zwischen 000 und 999
        number = random.randint(0, 999)
        formatted_number = f"{number:03}"

        #Path erstellen
        path = f"{'rl_gripper/resources/models/random_objects/'}{formatted_number}{'/'}{formatted_number}{'.urdf'}"
        return path

