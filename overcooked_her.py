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

import sys

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)

sys.excepthook = info

import numpy as np

model_class = DQN  # works also with SAC, DDPG and TD3

mdp = OvercookedGridworld.from_layout_name("cramped_room_single")
base_env = OvercookedEnv.from_mdp(mdp, horizon=1e4)
env = gym.make('Overcooked-single-v0')
env.custom_init(base_env, base_env.lossless_state_encoding_mdp_single)
env = gym.wrappers.TimeLimit(env, max_episode_steps=10)
env = Monitor(env, "./her_overcooked/", allow_early_resets=True)

# Available strategies (cf paper): future, final, episode
# goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
# max_episode_length = 50

# action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.3 * np.ones(1))

# Initialize the model
model = HER(
   "MlpPolicy",
   env,
   model_class,
   n_sampled_goal=4,
   goal_selection_strategy=goal_selection_strategy,
   # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
   # we have to manually specify the max number of steps per episode
   # max_episode_length=max_episode_length,
   verbose=1,
   buffer_size=int(1e6),
   learning_rate=1e-3,
   gamma=0.95,
   tensorboard_log="./her_overcooked",
   batch_size=256,
   online_sampling=online_sampling,
   # action_noise = action_noise,
   # policy_kwargs=dict(net_arch=[256, 256, 256]),
)

# Train the model
# for i in range(1000):
model.learn(100000)
model.save(f"./her_overcooked/saves/her_model_100000")

model = HER.load(f"./her_overcooked/saves/her_model_100000", env)

episode_reward=0
obs = env.reset()
import time
from os import system
for _ in range(100):
   # action, _ = model.predict(obs, deterministic=True)
   action, _ = model.predict(obs)
   obs, reward, done, info = env.step(action)
   print(env.env.env.base_env)
   print(reward)
   episode_reward += reward
   if done or info.get("is_success", False):
       print("Reward:", episode_reward, "Success?", info.get("is_success", False))
       episode_reward = 0.0
       obs = env.reset()
   time.sleep(0.25)
   system("clear")

# for _ in range(100):
#    action, _ = model.predict(obs, deterministic=True)
#    obs, reward, done, info = env.step(action)
#    env.render()
#    episode_reward += reward
#    if done or info.get("is_success", False):
#        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
#        episode_reward = 0.0
#        obs = env.reset()

