import gymnasium as gym
import os
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from rl_gripper.resources.classes.customClasses import TensorboardCallback, CustomCNN, CustomResNetFeatureExtractor
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch

torch.cuda.empty_cache()

# 2M timesteps in 39h

log_path = os.path.join('rl_gripper', 'training', 'logs')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'SAC_Model_2M')
#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_1
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs

### LOAD TRAINING ENVIRONMENT ###
env_kwargs = {'render_mode': 'GUI'}
train_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
train_env = VecTransposeImage(train_env)
train_env = VecFrameStack(train_env, n_stack=4)

### LOAD EVAL ENVIRONMENT ###
env_kwargs = {'render_mode': 'DIRECT'}
eval_env = make_vec_env("Gripper-v0", n_envs=1)
eval_env = VecTransposeImage(eval_env)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join('rl_gripper', 'training', 'saved_models'),
                             eval_freq=5000,
                             deterministic=True, render=False)

### TRAINING ###
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = SAC("CnnPolicy", train_env,
            verbose=1,
            buffer_size=200000,
            batch_size=500,
            ent_coef='auto',
            learning_rate=0.0003,
            device='cuda',
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_path)
model.learn(total_timesteps=2000000, callback=[eval_callback], progress_bar=True)
model.save(save_path)



