from gymnasium.envs.registration import register
register(
    id="Gripper-v0",
    entry_point="rl_gripper.envs:GripperEnv"
)


