import pybullet as p

class Plane:

    def __init__(self, client):
        f_path = "rl_gripper/resources/models/plane.urdf"

        self.client = client
        self.id = p.loadURDF(f_path)

    def get_ids(self):
        return self.client, self.id

