import gymnasium as gym
import os
import numpy as np
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env


log_path = os.path.join('rl_gripper', 'training', 'logs')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'best_model')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'SAC_Vergleich_noCurr')
#save_path = os.path.join('rl_gripper', 'training', 'checkpoints', 'SAC_FR_RP_200k_2048_1580000_steps')

#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_1
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs/coords_input

### LOAD ENVIRONMENT ###
env = gym.make("Gripper-v0", cube_position='RANDOM', curriculum=False)
env = DummyVecEnv([lambda: env])    # Für eval nur das verwenden
env = VecMonitor(env)
#env = VecFrameStack(env, n_stack=8)

# env_kwargs = {'render_mode': 'GUI'}
# env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
# env = VecTransposeImage(env)
# env = VecFrameStack(env, n_stack=4)

model = SAC.load(save_path, env=env)

print("Evaluation:")
print(evaluate_policy(model, env, n_eval_episodes=8, render=False))


### TESTING ###
# print("\nTesting:")
# episodes = 5
# for episode in range(1, episodes+1):
#     obs = env.reset()
#     terminated = False
#     truncated = False
#     score = 0
#
#     while not terminated:
#         action = model.predict(obs)
#         # action = env.action_space.sample()
#         obs, reward, terminated, truncated = env.step(np.array(action[0][0]))
#         #obs, reward, terminated, truncated, info = env.step(action)
#         score += reward
#         #print("\nStepreward: {}".format(reward))
#     print('\nEpisode: {} --- Score: {}'.format(episode, score))
# env.close()