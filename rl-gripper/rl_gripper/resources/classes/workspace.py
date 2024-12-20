from rl_gripper.resources.functions.helper import load_config


config = load_config()

min_workspace_area = config['curriculum']['min_workspace_area']
max_workspace_area = config['curriculum']['max_workspace_area'] #Maximale Seitenlänge der rechteckigen Arbeitsfläche (Mittelpunkt bei (0.4, 0.0))

class Workspace:
    def __init__(self):
        self.min_workspace_area = min_workspace_area
        self.max_workspace_area = max_workspace_area
        self.xMin = None
        self.xMax = None
        self.yMin = None
        self.yMax = None

    def define_workspace(self, curriculum):
        if curriculum.enabled:
            self.xMin = 0.4 - self.min_workspace_area/2
            self.xMax = 0.4 + self.min_workspace_area/2
            self.yMin = 0.0 - self.min_workspace_area/2
            self.yMax = 0.0 + self.min_workspace_area/2
        else:
            self.xMin = 0.4 - self.max_workspace_area/2
            self.xMax = 0.4 + self.max_workspace_area/2
            self.yMin = 0.0 - self.max_workspace_area/2
            self.yMax = 0.0 + self.max_workspace_area/2

