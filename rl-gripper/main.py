import gym
import os
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

log_path = os.path.join('rl_gripper', 'training', 'logs')
PPO_path = os.path.join('rl_gripper', 'training', 'saved_models', 'PPO_Model_1_200000')
#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_2


### LOAD ENVIRONMENT ###
env = gym.make("Gripper-v0")
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

### TRAINING ###
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model = PPO('MlpPolicy', env, learning_rate=0.0003,
            n_steps=4096,
            batch_size=128,
            ent_coef=0.01,      # exploration (0.1) vs convergence (0.01)
            verbose=1,
            tensorboard_log=log_path)

model.learn(total_timesteps=200000)
model.save(PPO_path)
del model

### EVALUATION ###
# model = PPO.load(PPO_path, env=env)
# evaluate_policy(model, env, n_eval_episodes=10, render=False)

### TESTING ###
# episodes = 10
# for episode in range(1, episodes+1):
#     obs = env.reset()
#     terminated = False
#     truncated = False
#     score = 0
#
#     while not terminated:
#         action = model.predict(obs)
#         obs, reward, terminated, truncated = env.step(action[0].flatten())
#         score += reward
#     print('Episode: {} --- Score: {}'.format(episode, score))
# env.close()

