from PIL import Image
import pybullet as p
import random
import yaml
import json
import os
from typing import Callable
import math
from contextlib import contextmanager
import time

def render_rgba_flat(width, height, rgba_flat_array):
    image = Image.new("RGBA", (width, height))

    rgba_array = [(rgba_flat_array[i], rgba_flat_array[i + 1], rgba_flat_array[i + 2], rgba_flat_array[i + 3])
                  for i in range(0, len(rgba_flat_array), 4)]

    image.putdata(rgba_array)
    image.show()


def spawn_random_cubes(count):
    for i in range(count):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        cubeID = p.loadURDF("model/cube.urdf", [x, y, .8], p.getQuaternionFromEuler([0, 0, 0]))


def load_config(path='rl_gripper/config/config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_parameters(name, model, config):
    hyperparams = {
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "buffer_size": model.buffer_size,
        "batch_size": model.batch_size,
        "ent_coef": model.ent_coef,
        "learning_rate": str(model.learning_rate),
        "lr_start": model.ent_coef_optimizer.param_groups[0]['lr'],
        "learning_starts": model.learning_starts,
        "gamma": model.gamma,
        "device": str(model.device),
        "gradient_steps": model.gradient_steps,
        "tau": model.tau,
        "train_freq": str(model.train_freq),
        "action_noise": str(model.action_noise),
    }

    # Combine into a single dictionary with separate sections
    params = {
        "HYPERPARAMETER": hyperparams,
        "POLICY_KWARGS": str(model.policy_kwargs),
        "OPTIMIZER": str(model.policy.optimizer_class),
            "OPTIMIZER_KWARGS": str(model.policy.optimizer_kwargs),
        "CONFIG": config
    }

    # Save the dictionary to a file
    with open(os.path.join('rl_gripper/training/saved_models', f"{name}.json"), 'w') as file:
        json.dump(params, file, indent=1)


def linear_schedule(initial_value: float, target_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value + (target_value - initial_value) * (1 - progress_remaining)

    return func

def cosine_schedule(initial_value: float, target_value: float) -> Callable[[float], float]:
    """
    Cosine decay learning rate schedule.

    :param initial_value: Initial learning rate.
    :param target_value: Final learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        progress = 1 - progress_remaining
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return target_value + (initial_value - target_value) * cosine_decay

    return func

def cosine_schedule_with_warmup(initial_value: float, target_value: float, warmup_duration: float = 0.01) -> Callable[[float], float]:
    """
    Cosine decay learning rate schedule with a warm-up phase.

    :param initial_value: Initial learning rate after warm-up.
    :param target_value: Final learning rate at the end of training.
    :param warmup_duration: Fraction of total time used for warm-up (default is 1%).
    :return: A schedule function that computes the current learning rate based on remaining progress.
    """
    def func(progress_remaining: float) -> float:
        """
        Computes the learning rate at a given point in the training process.

        :param progress_remaining: Remaining progress (from 1.0 to 0.0).
        :return: Current learning rate.
        """
        progress = 1 - progress_remaining  # Progress increases from 0 to 1

        if progress < warmup_duration:
            # Warm-up phase: linearly increase from target_value to initial_value
            warmup_progress = progress / warmup_duration
            lr = target_value + (initial_value - target_value) * warmup_progress
        else:
            # Cosine decay phase
            # Adjust the progress to account for the time spent in warm-up
            adjusted_progress = (progress - warmup_duration) / (1 - warmup_duration)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_progress))
            lr = target_value + (initial_value - target_value) * cosine_decay

        return lr

    return func

@contextmanager
def tictoc():
    t0 = time.time()  # Start the timer
    yield
    t1 = time.time()  # End the timer
    print(f"dt: {(t1 - t0) * 1000:.4f} ms")