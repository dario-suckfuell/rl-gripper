import pybullet as p

class Plane:

    def __init__(self, client):
        f_path = "rl_gripper/resources/models/Plane/plane.urdf"

        self.client = client
        self.id = p.loadURDF(f_path, physicsClientId=self.client)
        #_ = p.loadURDF("/home/dsuckfuell/rl-gripper/venv/lib/python3.10/site-packages/pybullet_data/table/table.urdf", [0.4, 0, 0],  physicsClientId=self.client)

    def get_ids(self):
        return self.client, self.id

