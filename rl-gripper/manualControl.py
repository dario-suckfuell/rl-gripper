import gymnasium as gym
import os
import pybullet as p
import numpy as np
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

log_path = os.path.join('rl_gripper', 'training', 'logs')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'PPO_Model_1_500000')
#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_1
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs

### LOAD ENVIRONMENT ###
env = gym.make("Gripper-v0")
# env = DummyVecEnv([lambda: env]) # Zerstört step function
# env = VecFrameStack(env, n_stack=4)
# env = VecMonitor(env)

yaw = p.addUserDebugParameter("1", -1, 1, 0)
param2 = p.addUserDebugParameter("2", -1, 1, 0)
param3 = p.addUserDebugParameter("3", -1, 1, 0)
paramGripper = p.addUserDebugParameter("4", -1.0, 1, 0)

### MANUAL CONTROL ###
print("\nManual Control:")
while True:
    obs = env.reset()
    terminated = False
    truncated = False
    score = 0

    while not terminated:
        # userParam0 = p.readUserDebugParameter(param0)
        userYaw = p.readUserDebugParameter(yaw)
        userParam2 = p.readUserDebugParameter(param2)
        userParam3 = p.readUserDebugParameter(param3)
        # userParam4 = p.readUserDebugParameter(param4)
        # userParam5 = p.readUserDebugParameter(param5)
        # userParam6 = p.readUserDebugParameter(param6)
        # userParam7 = p.readUserDebugParameter(param7)
        userParamGripper = p.readUserDebugParameter(paramGripper)

        action = np.array([userYaw, userParam2, userParam3, userParamGripper])

        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        #obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        #print("\nStepreward: {}".format(reward))
    print('\nScore: {}'.format(score))
env.close()

