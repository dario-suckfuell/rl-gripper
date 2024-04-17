import gymnasium as gym
import os
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from rl_gripper.resources.classes.customClasses import TensorboardCallback, CustomCNN, CustomCNN_attention, CustomCNN_big, CustomCNN_maxPooling
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_1
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs/coords_input


### BASE ###

torch.cuda.empty_cache()
log_path = os.path.join('rl_gripper', 'training', 'logs', 'coords_input')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'SAC_Vergleich_Batch')

### LOAD TRAINING ENVIRONMENT ###
env_kwargs = {'render_mode': 'DIRECT', 'cube_position': 'FIX'}
train_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
#train_env = VecTransposeImage(train_env)
train_env = VecNormalize(train_env, training=True, norm_obs=False, norm_reward=True, clip_obs=10.)
train_env = VecMonitor(train_env)

### LOAD EVAL ENVIRONMENT ###
env_kwargs = {'render_mode': 'DIRECT', 'cube_position': 'FIX'}
eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
#eval_env = VecTransposeImage(eval_env)
eval_env = VecNormalize(eval_env, training=True, norm_obs=False, norm_reward=True, clip_obs=10.)
eval_env = VecMonitor(eval_env)


### CALLBACKS ### (mitverantwortlich für FPS)
eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join('rl_gripper', 'training', 'saved_models'),
                             eval_freq=5000,    #eval_freq = eval_freq * n_envs
                             deterministic=True, render=False, n_eval_episodes=10)
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=os.path.join('rl_gripper', 'training', 'checkpoints'),
                                         name_prefix='SAC_Vergleich_Batch')

### TRAINING ###
# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=512),
#     #activation_fn=torch.nn.Tanh,
#     #net_arch,
# )

model = SAC("MlpPolicy", train_env,
            verbose=1,
            buffer_size=1000000,
            batch_size=256,
            ent_coef='auto',
            learning_rate=0.0003,
            learning_starts=1000,
            gamma=0.99,
            device='cuda',
            #policy_kwargs=policy_kwargs,
            tensorboard_log=log_path)

model.learn(total_timesteps=1000000, callback=[eval_callback, checkpoint_callback], progress_bar=True)
model.save(save_path)
del model
del train_env
del eval_env

