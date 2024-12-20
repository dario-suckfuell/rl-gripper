import optuna
from optuna.storages import RDBStorage
import gymnasium as gym
import os
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from rl_gripper.resources.classes.customFeaturesExtractor import CustomCNN_attention, CustomAdapterNet
from rl_gripper.resources.classes.customCallbacks import TensorboardCallback, CurriculumCallback, SaveNormalizationCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecTransposeImage, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import torch
import numpy as np
from rl_gripper.resources.functions.helper import load_config

torch.cuda.empty_cache()
log_path = os.path.join('rl_gripper', 'training', 'logs', 'depth_input')
save_path = os.path.join('rl_gripper', 'training', 'saved_models')

config = load_config()


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    tau = trial.suggest_uniform('tau', 0.003, 0.006)
    gradient_steps = trial.suggest_int('gradient_steps', 3, 10)
    train_freq = trial.suggest_int('train_freq', 3, 10)
    ou_noise_sigma = trial.suggest_uniform('ou_noise_sigma', 0.10, 0.35)
    ou_noise_theta = trial.suggest_uniform('ou_noise_theta', 0.05, 0.25)
    beta1 = trial.suggest_uniform('beta1', 0.50, 0.90)
    beta2 = trial.suggest_uniform('beta2', 0.9, 0.999)

    ### LOAD TRAINING ENVIRONMENT ###
    env_kwargs = {'render_mode': 'DIRECT',
                  'cube_position': 'RANDOM',
                  'curriculum': True,
                  'dataset': 'CUBE'}

    train_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
    # train_env = VecTransposeImage(train_env)
    # train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.)
    train_env = VecMonitor(train_env)

    ### LOAD EVALUATION ENVIRONMENT ###
    env_kwargs = {'render_mode': 'DIRECT',
                  'cube_position': 'RANDOM',
                  'curriculum': False,
                  'dataset': 'CUBE'}

    eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)
    # eval_env = VecTransposeImage(eval_env)
    # eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env = VecMonitor(eval_env)


    policy_kwargs = dict(
        features_extractor_class=CustomAdapterNet,
        net_arch=[512, 512, 256]
    )

    # Configure the Ornstein-Uhlenbeck action noise
    ou_noise_mean = np.zeros(5)
    ou_noise_sigma = np.ones(5) * ou_noise_sigma
    ou_noise_theta = np.ones(5) * ou_noise_theta  # how "fast" the noise variable reverts towards the mean; increase for more exploration
    ou_noise_dt = 1e-2  # time step size
    action_noise = OrnsteinUhlenbeckActionNoise(mean=ou_noise_mean, sigma=ou_noise_sigma, theta=ou_noise_theta,
                                                dt=ou_noise_dt)

    model = SAC("MlpPolicy", train_env,
                verbose=1,
                buffer_size=500000,
                batch_size=512,
                ent_coef='auto',
                learning_rate=learning_rate,
                learning_starts=1000,
                gamma=0.995,
                device='cuda',
                policy_kwargs=policy_kwargs,
                gradient_steps=gradient_steps,
                tau=tau,  # increase for faster updates (faster adaption of new policy)
                train_freq=train_freq,
                action_noise=action_noise,
                tensorboard_log=log_path)

    # Customize the optimizer
    # model.policy.optimizer_class = torch.optim.Adam
    model.policy.optimizer_class = torch.optim.NAdam
    model.policy.optimizer_kwargs = dict(lr=learning_rate, betas=(beta1, beta2))  # Beta2 nicht zu niedrig (0.99)

    ### CALLBACKS ###
    save_vec_normalize = SaveNormalizationCallback(train_env, save_path)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                                 eval_freq=5000,  # eval_freq = eval_freq * n_envs
                                 deterministic=True, render=False, n_eval_episodes=10,
                                 callback_on_new_best=save_vec_normalize)
    checkpoint_callback = CheckpointCallback(save_freq=20000,
                                             save_path=os.path.join('rl_gripper', 'training', 'checkpoints'),
                                             name_prefix='SAC_ResNet_Vergleich',
                                             save_replay_buffer=False,
                                             save_vecnormalize=True)
    curriculum_callback = CurriculumCallback(model,
                                             threshold_for_increase=config['curriculum']['threshold_for_increase'])
    tensorboard_callback = TensorboardCallback(model)

    model.learn(total_timesteps=400000,
                callback=[eval_callback, checkpoint_callback, tensorboard_callback, curriculum_callback],
                progress_bar=True)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)

    return mean_reward

# Storage, Sampler and Pruner
storage = RDBStorage(url="sqlite:///optuna_study.db")
sampler = optuna.samplers.TPESampler()
pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)

study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner, storage=storage, study_name="SAC_ResNetOpt_01")
study.optimize(objective, n_trials=50)

print("Best hyperparameters: ", study.best_params)
