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


#save_path = os.path.join('rl_gripper', 'training', 'saved_models')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'Working', 'gripper_test')
norm_path = os.path.join(save_path, 'best_model_vec_normalize.pkl')
norm_path = os.path.join(save_path, 'SAC_GripperTest_vec_normalize.pkl')

# Load the trained model
model = SAC.load(os.path.join(save_path, 'SAC_GripperTest'))

### LOAD ENVIRONMENT ###
env_kwargs = {'render_mode': 'GUI',
              'cube_position': 'RANDOM',
              'curriculum': False,
              'dataset': 'YCB'}

eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
eval_env = VecTransposeImage(eval_env)

# Load and apply the saved normalization parameters
eval_env = VecNormalize.load(norm_path, eval_env)
eval_env.training = False  # Turn off training mode for normalization
eval_env.norm_reward = True  # Assuming you want to normalize rewards

# Wrap the environment if needed
eval_env = VecMonitor(eval_env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
print(f"Evaluation: Mean reward: {mean_reward} +/- {std_reward}")
eval_env.close()




#
# import gymnasium as gym
# import os
# import numpy as np
# from rl_gripper.envs.CustomGripperEnv import GripperEnv
# from stable_baselines3 import SAC, PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
# from stable_baselines3.common.env_util import make_vec_env
#
#
# log_path = os.path.join('rl_gripper', 'training', 'logs')
# save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'best_model')
# #save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'SAC_AttentionCNN_SimpleTestToCheckCurr')
# #save_path = os.path.join('rl_gripper', 'training', 'checkpoints', 'SAC_Final_GripperHeight_und_WorkspaceArea_165000_steps')
#
# #tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_1
# #tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs/coords_input
#
# ### LOAD ENVIRONMENT ###
# env_kwargs = {'render_mode': 'GUI',
#               'cube_position': 'RANDOM',
#               'curriculum': False}
# eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
# eval_env = VecTransposeImage(eval_env)
# eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)
# eval_env = VecMonitor(eval_env)
#
#
# # env_kwargs = {'render_mode': 'GUI'}
# # env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
# # env = VecTransposeImage(env)
# # env = VecFrameStack(env, n_stack=4)
#
# model = SAC.load(save_path, env=eval_env)
#
# print("Evaluation:")
# print(evaluate_policy(model, eval_env, n_eval_episodes=8, render=False))
#
#
# ### TESTING ###
# # print("\nTesting:")
# # episodes = 5
# # for episode in range(1, episodes+1):
# #     obs = env.reset()
# #     terminated = False
# #     truncated = False
# #     score = 0
# #
# #     while not terminated:
# #         action = model.predict(obs)
# #         # action = env.action_space.sample()
# #         obs, reward, terminated, truncated = env.step(np.array(action[0][0]))
# #         #obs, reward, terminated, truncated, info = env.step(action)
# #         score += reward
# #         #print("\nStepreward: {}".format(reward))
# #     print('\nEpisode: {} --- Score: {}'.format(episode, score))
# env.close()