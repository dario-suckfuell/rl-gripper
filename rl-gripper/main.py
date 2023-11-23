import gymnasium as gym
import os
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import numpy as np

log_path = os.path.join('rl_gripper', 'training', 'logs')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'SAC_Model_1_')
#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_1
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs

### LOAD ENVIRONMENT ###
env = gym.make("Gripper-v0")
# check_env(env)
env = DummyVecEnv([lambda: env])    # Für eval nur das verwenden
# env = VecNormalize(env, norm_obs=False, norm_reward=True)   #Obs wird von CnnPolicy normalisiert // norm_reward not possible for SAC
env = VecFrameStack(env, n_stack=4)
env = VecMonitor(env)

### TRAINING ###
# model = SAC.load(save_path, env=env)
model = SAC("CnnPolicy", env,
            verbose=1,
            buffer_size=50000,
            tensorboard_log=log_path)

# model = PPO.load(save_path, env=env)
# model = PPO('CnnPolicy', env, learning_rate=0.0003,
#             n_steps=50000,
#             batch_size=128,
#             ent_coef=0.02,      # exploration (0.1) vs convergence (0.01)
#             verbose=1,
#             tensorboard_log=log_path)

for episode in range(8):     # total episodes
    model.learn(total_timesteps=200000)
    checkpoint_path = f"{save_path}{(episode+1)*200000}"
    model.save(checkpoint_path)




