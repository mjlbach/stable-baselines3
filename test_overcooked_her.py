import gym
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, Overcooked
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, AgentFromPolicy, FixedPlanAgent, RandomAgent

from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.common.monitor import Monitor
import stable_baselines3.common.env_checker

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import numpy as np

model_class = DQN  # works also with SAC, DDPG and TD3

mdp = OvercookedGridworld.from_layout_name("cramped_room_single")
start_state_fn = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.5)
base_env = OvercookedEnv.from_mdp(mdp, start_state_fn=start_state_fn, horizon=1e4)
env = gym.make('Overcooked-single-v0')
env.custom_init(base_env, base_env.lossless_state_encoding_mdp_single)
env = gym.wrappers.TimeLimit(env, max_episode_steps=10)
env = Monitor(env, "./her_overcooked/", allow_early_resets=True)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True

model = HER.load(f"./her_overcooked/saves/her_model_300000", env)

episode_reward=0
obs = env.reset()
import time
from os import system
for _ in range(100):
   action, _ = model.predict(obs, deterministic=True)
   obs, reward, done, info = env.step(action)
   print(env.env.env.base_env)
   print(reward)
   episode_reward += reward
   if done or info.get("is_success", False):
       print("Number of steps:", episode_reward, "Success?", reward == 0.0)
       episode_reward = 0.0
       obs = env.reset()
   else:
       print("Not yet finished")
   time.sleep(0.5)
   system("clear")
