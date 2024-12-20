import gymnasium as gym
import os
import shutil
import stable_baselines3.common.type_aliases
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from rl_gripper.resources.classes.customFeaturesExtractor import AdapterNetworkMHP
from rl_gripper.resources.classes.customCallbacks import TensorboardCallback, CurriculumCallback, \
    SaveNormalizationCallback, ActionNoiseDecayCallback
from rl_gripper.resources.classes.customPolicy import CustomSACPolicy
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import MlpExtractor
import torch
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import numpy as np
from rl_gripper.resources.functions.helper import load_config, save_parameters, linear_schedule, cosine_schedule, cosine_schedule_with_warmup
import torch.profiler

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()

#tensorboard --logdir=C:\Users\Dario\Desktop\rl-gripper\rl-gripper\rl_gripper\training\logs
#tensorboard --logdir=/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/logs
#tensorboard --logdir=/home/dsuckfuell/Desktop/rl-gripper/rl-gripper/rl_gripper/training/logs

config = load_config()

### NAME AND PATHS ###
name = "YCB_10M"
log_path = os.path.join('rl_gripper', 'training', 'logs')
save_path = os.path.join('rl_gripper', 'training', 'saved_models')

### LOAD TRAINING ENVIRONMENT ###
env_kwargs = {'render_mode': 'DIRECT',
              'curriculum_enabled': True,
              'dataset': 'YCB',
              'mode': 'TRAIN',
              'reward_scale': 10}

train_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
train_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.)
train_env = VecMonitor(train_env)

### LOAD EVALUATION ENVIRONMENT ###
env_kwargs = {'render_mode': 'DIRECT',
              'curriculum_enabled': False,
              'dataset': 'YCB',
              'mode': 'EVAL',
              'reward_scale': 10}

eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)
eval_env = VecMonitor(eval_env)

policy_kwargs = dict(
    features_extractor_class=AdapterNetworkMHP,
    net_arch=dict(pi=[256], qf=[512, 512]),
    n_critics=2,
)

# TOTAL TIMESTEPS
total_timesteps = 10000000

# ACTION NOISE
ou_noise_mean = np.zeros(5)
ou_noise_initial_sigma = np.array([0.55, 0.55, 0.35, 0.35, 0.25])
ou_noise_target_sigma = np.array([0.05, 0.05, 0.05, 0.03, 0.03])
ou_noise_theta = np.array([0.15, 0.15, 0.15, 0.15, 0.15])  # how "fast" the noise variable reverts towards the mean; increase for more exploration
ou_noise_dt = 1e-2  # time step size
action_noise = OrnsteinUhlenbeckActionNoise(mean=ou_noise_mean, sigma=ou_noise_initial_sigma, theta=ou_noise_theta,
                                            dt=ou_noise_dt)

### SAC ALGORITHM AND HYPERPARAMETERS ###
SAC.policy_aliases["CustomSACPolicy"] = CustomSACPolicy
model = SAC("CustomSACPolicy", train_env,
            verbose=1,
            buffer_size=2*1048576, #2**20
            batch_size=4096,
            ent_coef='auto',
            learning_rate=cosine_schedule(0.0008, 0.00008),
            learning_starts=2*8192,
            gamma=0.99,
            device='cuda',
            policy_kwargs=policy_kwargs,
            gradient_steps=3,
            tau=0.01,  #increase for faster updates (faster adaption of new policy) [0.01 - 0.001]
            train_freq=3,
            action_noise=action_noise,
            tensorboard_log=log_path,
            )

### OPTIMIZER ###
model.policy.optimizer_class = torch.optim.NAdam
model.policy.optimizer_kwargs = dict(lr=cosine_schedule(0.0008, 0.00008), betas=(0.9, 0.99), weight_deceay=0.1)  #Beta2 nicht zu niedrig (0.99)

### CALLBACKS ###
save_vec_normalize = SaveNormalizationCallback(train_env, save_path)
eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                             eval_freq=10000,  #eval_freq = eval_freq * n_envs
                             deterministic=True, render=False, n_eval_episodes=10,
                             callback_on_new_best=save_vec_normalize)
checkpoint_callback = CheckpointCallback(save_freq=20000,
                                         save_path=os.path.join('rl_gripper', 'training', 'checkpoints'),
                                         name_prefix=name,
                                         save_replay_buffer=False,
                                         save_vecnormalize=True)
curriculum_callback = CurriculumCallback(model, threshold_for_increase=config['curriculum']['threshold_for_increase'])
tensorboard_callback = TensorboardCallback(model, log_path)
action_noise_decay_callback = ActionNoiseDecayCallback(action_noise, ou_noise_initial_sigma, ou_noise_target_sigma, total_timesteps)

save_parameters(name, model, config) # Save all important Parameters

model.learn(total_timesteps=total_timesteps,
                callback=[eval_callback, checkpoint_callback, tensorboard_callback, curriculum_callback, action_noise_decay_callback],
                progress_bar=False)

model.save(os.path.join(save_path, f"{name}.zip"))
train_env.save(os.path.join(save_path, f"{name}.pkl"))

del model
del train_env
del eval_env

