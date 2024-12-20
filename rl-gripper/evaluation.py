import gymnasium as gym
import os
import shutil
import numpy as np
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env


#Reset Dataset
#Delete the old active models folder and create a new one with the YCB 01_very_easy Dataset
# if os.path.exists('rl_gripper/resources/models/YCB/urdf_models/active_models'):
#     shutil.rmtree('rl_gripper/resources/models/YCB/urdf_models/active_models')
# shutil.copytree('rl_gripper/resources/models/YCB/urdf_models/04_hard', 'rl_gripper/resources/models/YCB/urdf_models/active_models', dirs_exist_ok=False)


### MODEL 4.06M ###

### BEST MODEL ###
# save_path = os.path.join('rl_gripper', 'training', 'saved_models')
# norm_path = os.path.join(save_path, 'best_model_vec_normalize.pkl')
# model = SAC.load(os.path.join(save_path, 'best_model'))

### TRAIN MODEL ###
model = SAC.load('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_4060000_steps.zip')
norm_path = os.path.join('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_vecnormalize_4060000_steps.pkl')

### EVAL MODEL ###
# model = SAC.load('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_8000000_steps.zip')
# norm_path = os.path.join('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_vecnormalize_8000000_steps.pkl')

### AVG MODEL ###
# model = SAC.load('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/averaged_model.zip')
# norm_path = os.path.join('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/averaged_vecnormalize.pkl')

# Load the trained model timestep
# model = SAC.load('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/FINAL/02_ycb_hard_5M/02_checkpoints/ycb-full_run_hard_3040000_steps.zip')
# norm_path = os.path.join('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/FINAL/02_ycb_hard_5M/02_checkpoints/ycb-full_run_hard_vecnormalize_3040000_steps.pkl')

# Load the trained model timestep
# model = SAC.load('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/FINAL/01_ycb_easy_5M/02_checkpoints/ycb-full_run_easy_4450000_steps.zip')
# norm_path = os.path.join('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/FINAL/01_ycb_easy_5M/02_checkpoints/ycb-full_run_easy_vecnormalize_4450000_steps.pkl')


# Load the trained model checkpoint
# model = SAC.load('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/FINAL/03_rng_5M/checkpoints/ycb-full_run_hard_3040000_steps.zip')
# norm_path = os.path.join('/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/FINAL/03_rng_5M/checkpoints/ycb-full_run_hard_vecnormalize_3040000_steps.pkl')


### LOAD ENVIRONMENT ###
env_kwargs = {'render_mode': 'GUI',
              'curriculum_enabled': False,
              'dataset': 'YCB',
              'mode': 'TRAIN',
              'reward_scale': 10}

eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)


# Load and apply the saved normalization parameters
eval_env = VecNormalize.load(norm_path, eval_env)
eval_env.training = False  # Turn off training mode for normalization
eval_env.norm_reward = True  # Assuming you want to normalize rewards
print(f"Normalization statistics (reward mean, variance): {eval_env.ret_rms.mean}, {eval_env.ret_rms.var}")

# Reset the environment to apply normalization statistics
#eval_env.reset()

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, render=False)
print(f"Evaluation 4.06M: Mean reward: {mean_reward} +/- {std_reward}")
eval_env.close()


