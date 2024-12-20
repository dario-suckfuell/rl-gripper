import gymnasium as gym
import os
import pybullet as p
import numpy as np
import shutil
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

#Reset Dataset
# Delete the old active models folder and create a new one with the YCB 01_very_easy Dataset
# if os.path.exists('rl_gripper/resources/models/YCB/urdf_models/active_models'):
#     shutil.rmtree('rl_gripper/resources/models/YCB/urdf_models/active_models')
# shutil.copytree('rl_gripper/resources/models/YCB/urdf_models/00_test', 'rl_gripper/resources/models/YCB/urdf_models/active_models', dirs_exist_ok=False)

env = gym.make("Gripper-v0", curriculum_enabled=False, dataset='YCB', mode='TRAIN_easy', reward_scale=10) #dummy for 00_dummy

x_fader = p.addUserDebugParameter("X", -1, 1, 0)
y_fader = p.addUserDebugParameter("Y", -1, 1, 0)
z_fader = p.addUserDebugParameter("Z", -1, 1, 0)
yaw_fader = p.addUserDebugParameter("Yaw", -1, 1, 0)
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
        yaw = p.readUserDebugParameter(yaw_fader)
        Gw = p.readUserDebugParameter(Gw_fader)

        action = np.array([x, y, z, yaw, Gw])

        obs, reward, terminated, truncated, info = env.step(action)
        #print(reward)
        score += reward
        #print("\nStepreward: {}".format(reward))
    print('\nScore: {}'.format(score))
env.close()

