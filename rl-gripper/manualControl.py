import gymnasium as gym
import os
import pybullet as p
import numpy as np
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env

log_path = os.path.join('rl_gripper', 'training', 'logs')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'PPO_Model_1_500000')
#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_1
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs

### LOAD ENVIRONMENT ###
### LOAD TRAINING ENVIRONMENT ###

env = gym.make("Gripper-v0", cube_position='RANDOM', curriculum=False)

x_fader = p.addUserDebugParameter("X", -1, 1, 0)
y_fader = p.addUserDebugParameter("Y", -1, 1, 0)
z_fader = p.addUserDebugParameter("Z", -1, 1, 0)
Gw_fader = p.addUserDebugParameter("Gw", -1, 1, 0)

### MANUAL CONTROL ###
print("\nManual Control:")
while True:
    obs = env.reset()
    terminated = False
    truncated = False
    score = 0

    while not terminated:
        x = p.readUserDebugParameter(x_fader)
        y = p.readUserDebugParameter(y_fader)
        z = p.readUserDebugParameter(z_fader)
        Gw = p.readUserDebugParameter(Gw_fader)

        action = np.array([x, y, z, Gw])

        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        score += reward
        #print("\nStepreward: {}".format(reward))
    print('\nScore: {}'.format(score))
env.close()

