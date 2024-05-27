import gymnasium as gym
import os
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from rl_gripper.resources.classes.customFeaturesExtractor import CustomCNN, CustomCNN_attention, CustomCNN_attentionBIG
from rl_gripper.resources.classes.customCallbacks import TensorboardCallback, CurriculumCallback, SaveNormalizationCallback
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
import numpy as np
from rl_gripper.resources.functions.helper import load_config

#TODO
#Config File
#(Pretrained) Vision Transformer Model
#Dropout for better generalisierung
#Attention Num Heads opt
#Adaptive Noise Scaling
#SDG
#VecFrameStack?
#NAdam Optm?
#Curr Threshold auf 0.6?
#obj scaling
#Picking Height continuierlich


#tensorboard --logdir=C:\Users\Dario\Desktop\rl-gripper\rl-gripper\rl_gripper\training\logs\depth_input
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs/depth_input

config = load_config()


### 01 ###

torch.cuda.empty_cache()
log_path = os.path.join('rl_gripper', 'training', 'logs', 'depth_input')
save_path = os.path.join('rl_gripper', 'training', 'saved_models')

### LOAD TRAINING ENVIRONMENT ###
env_kwargs = {'render_mode': 'DIRECT',
              'cube_position': 'RANDOM',
              'curriculum': True,
              'dataset': 'TRAINING'}

train_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
train_env = VecTransposeImage(train_env)
#train_env = VecFrameStack(train_env, n_stack=4)
train_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.)
train_env = VecMonitor(train_env)

### LOAD EVALUATION ENVIRONMENT ###
env_kwargs = {'render_mode': 'DIRECT',
              'cube_position': 'RANDOM',
              'curriculum': False,
              'dataset': 'VALIDATION'}

eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
eval_env = VecTransposeImage(eval_env)
#eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)
eval_env = VecMonitor(eval_env)

### TRAINING ###

policy_kwargs = dict(
    features_extractor_class=CustomCNN_attentionBIG,
    features_extractor_kwargs=dict(features_dim=1024),
    net_arch=[512, 512, 256]
)

# Configure the Ornstein-Uhlenbeck action noise
ou_noise_mean = np.zeros(5)
ou_noise_sigma = np.array([0.25, 0.25, 0.25, 0.20, 0.20])
ou_noise_theta = np.array([0.2, 0.2, 0.2, 0.15, 0.15])   # how "fast" the noise variable reverts towards the mean; increase for more exploration
ou_noise_dt = 1e-2  # time step size
action_noise = OrnsteinUhlenbeckActionNoise(mean=ou_noise_mean, sigma=ou_noise_sigma, theta=ou_noise_theta, dt=ou_noise_dt)

model = SAC("CnnPolicy", train_env,
            verbose=1,
            buffer_size=1000000,
            batch_size=128,
            ent_coef='auto',
            learning_rate=0.00035,
            learning_starts=5000,
            gamma=0.99,
            device='cuda',
            policy_kwargs=policy_kwargs,
            gradient_steps=4,
            tau=0.005, #increase for faster updates (faster adaption of new policy)
            train_freq=4,
            action_noise=action_noise,
            tensorboard_log=log_path)

# Customize the optimizer
model.policy.optimizer_class = torch.optim.Adam
#model.policy.optimizer_class = torch.optim.nadam
model.policy.optimizer_kwargs = dict(lr=0.00035, betas=(0.78, 0.9))

### CALLBACKS ###
save_vec_normalize = SaveNormalizationCallback(train_env, save_path)
eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                             eval_freq=5000,    #eval_freq = eval_freq * n_envs
                             deterministic=True, render=False, n_eval_episodes=10,
                             callback_on_new_best=save_vec_normalize)
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=os.path.join('rl_gripper', 'training', 'checkpoints'), name_prefix='SAC_FullRun_RandomObject',
                                         save_replay_buffer=False,
                                         save_vecnormalize=True)
curriculum_callback = CurriculumCallback(model)
tensorboard_callback = TensorboardCallback(model)

model.learn(total_timesteps=10000000, callback=[eval_callback, checkpoint_callback, tensorboard_callback, curriculum_callback], progress_bar=True)
model.save(os.path.join(save_path, "SAC_FullRun_RandomObject.zip"))
train_env.save(os.path.join(save_path, "SAC_FullRun_RandomObject.pkl"))

del model
del train_env
del eval_env

