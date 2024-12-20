import torch
from stable_baselines3 import SAC  # Replace with your algorithm if different

import torch
from stable_baselines3 import PPO  # Replace with your algorithm if different
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env


def average_parameters_with_vecnormalize(model_paths, vecnormalize_paths, save_path, vecnormalize_save_path):
    """
    Average the parameters of multiple Stable-Baselines3 models and their VecNormalize statistics.

    Args:
        model_paths (list of str): List of paths to the model checkpoints.
        vecnormalize_paths (list of str): List of paths to the VecNormalize objects corresponding to the models.
        save_path (str): Path to save the averaged model.
        vecnormalize_save_path (str): Path to save the averaged VecNormalize statistics.
    """
    # Load all models
    models = [SAC.load(path) for path in model_paths]

    # Get the first model's parameters as a reference
    avg_params = {key: torch.zeros_like(param) for key, param in models[0].policy.state_dict().items()}

    # Sum the parameters from all models
    for model in models:
        for key, param in model.policy.state_dict().items():
            avg_params[key] += param

    # Divide by the number of models to get the average
    num_models = len(models)
    for key in avg_params:
        avg_params[key] /= num_models

    # Load the averaged parameters into the first model
    models[0].policy.load_state_dict(avg_params)

    # Save the averaged model
    models[0].save(save_path)
    print(f"Averaged model saved to {save_path}")

    ### LOAD ENVIRONMENT ###
    env_kwargs = {'render_mode': 'GUI',
                  'curriculum_enabled': False,
                  'dataset': 'YCB',
                  'mode': 'EVAL',
                  'reward_scale': 10}

    eval_env = make_vec_env("Gripper-v0", n_envs=1, env_kwargs=env_kwargs)

    # Average VecNormalize statistics
    vec_normalizes = [VecNormalize.load(path, eval_env) for path in vecnormalize_paths]
    avg_obs_mean = sum(v.obs_rms.mean for v in vec_normalizes) / num_models
    avg_obs_var = sum(v.obs_rms.var for v in vec_normalizes) / num_models
    avg_ret_mean = sum(v.ret_rms.mean for v in vec_normalizes) / num_models
    avg_ret_var = sum(v.ret_rms.var for v in vec_normalizes) / num_models

    # Use the first VecNormalize as a reference to update and save averaged stats
    reference_vecnormalize = vec_normalizes[0]
    reference_vecnormalize.obs_rms.mean = avg_obs_mean
    reference_vecnormalize.obs_rms.var = avg_obs_var
    reference_vecnormalize.ret_rms.mean = avg_ret_mean
    reference_vecnormalize.ret_rms.var = avg_ret_var

    # Save the updated VecNormalize
    reference_vecnormalize.save(vecnormalize_save_path)
    print(f"Averaged VecNormalize statistics saved to {vecnormalize_save_path}")


# Example usage
model_paths = ["/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_8000000_steps.zip",
               "/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_8200000_steps.zip"]

vecnormalize_paths = ["/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_vecnormalize_8000000_steps.pkl",
                      "/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/checkpoints/YCB_refined_10M_vecnormalize_8200000_steps.pkl"]

save_path = "/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/averaged_model.zip"
vecnormalize_save_path = "/home/dsuckfuell/rl-gripper/rl-gripper/rl_gripper/training/averaged_vecnormalize.pkl"
average_parameters_with_vecnormalize(model_paths, vecnormalize_paths, save_path, vecnormalize_save_path)

