from rl_gripper.resources.functions.helper import load_config

config = load_config()

class Curriculum:
    def __init__(self, curriculum_enabled):
        self.enabled = curriculum_enabled

        self.laps_counter = 0

        self.total_laps = config['curriculum']['curr_laps']
        self.threshold_for_increase = config['curriculum']['threshold_for_increase']


