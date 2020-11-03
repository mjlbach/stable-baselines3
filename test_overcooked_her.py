import gym
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, Overcooked
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, AgentFromPolicy, FixedPlanAgent, RandomAgent
from os import system

from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
import stable_baselines3.common.env_checker

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import numpy as np

model_class = TD3  # works also with SAC, DDPG and TD3

mdp = OvercookedGridworld.from_layout_name("large_room_single")
base_env = OvercookedEnv.from_mdp(mdp, horizon=1e4)
env = gym.make('Overcooked-single-v0')
env.custom_init(base_env, base_env.lossless_state_encoding_mdp_single)

# Available strategies (cf paper): future, final, episode
# goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = 50

action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.3 * np.ones(1))

# Initialize the model
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, action_noise=action_noise, goal_selection_strategy=goal_selection_strategy, online_sampling=online_sampling,
                        verbose=1, max_episode_length=max_episode_length, tensorboard_log="./her_overcooked/")

model = HER.load('./her_bit_env40.zip', env=env)

obs = env.reset()
for i in range(1000):
    action, _ = model.model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    import time
    time.sleep(0.5)
    system("clear")

    if done or i % 20 == 0:
        obs = env.reset()


