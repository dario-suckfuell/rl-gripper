import gym
import os
from rl_gripper.envs.CustomGripperEnv import GripperEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

log_path = os.path.join('rl_gripper', 'training', 'logs')
save_path = os.path.join('rl_gripper', 'training', 'saved_models', 'PPO_Model_1_10000')
#tensorboard --logdir=D:\projects\rl-gripper\rl_gripper\training\logs\PPO_2


### LOAD ENVIRONMENT ###
env = gym.make("Gripper-v0")
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env = VecMonitor(env)

model = PPO.load(save_path, env=env)

### TRAINING ###
# model = PPO('CnnPolicy', env, learning_rate=0.0004,
#             n_steps=1024,
#             batch_size=512,
#             ent_coef=0.05,      # exploration (0.1) vs convergence (0.01)
#             verbose=1,
#             tensorboard_log=log_path)
#
# for episode in range(1):     # total episodes
#     model.learn(total_timesteps=10000)
#     checkpoint_path = f"{save_path}{(episode+1)*10000}"
#     model.save(checkpoint_path)

### EVALUATION ###
# model = PPO.load(save_path, env=env)
# print("Evaluation:")
# print(evaluate_policy(model, env, n_eval_episodes=3, render=False))

# del model

### TESTING ###
print("\nTesting:")
episodes = 3
for episode in range(1, episodes+1):
    obs = env.reset()
    terminated = False
    truncated = False
    score = 0

    while not terminated:
        # action = model.predict(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated = env.step(action[0].flatten())
        # obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        #print("\nStepreward: {}".format(reward))
    print('\nEpisode: {} --- Score: {}'.format(episode, score))
env.close()

