import os, sys
sys.path.insert(1, os.getcwd() + '/envs')
sys.path.insert(1, os.getcwd() + '/ppo')
sys.path.insert(1, os.getcwd() + '/ddpg')
sys.path.insert(1, os.getcwd() + '/td3')

import numpy as np

import logging
from simple import make_env
from envhandler import EnvHandler
from dummy_vec_env import DummyVecEnv
from subproc_vec_env import SubprocVecEnv
#from ppo import learn, view
#from ddpg import learn, view
from td3 import learn
from gym.wrappers import Monitor


logger = logging.getLogger(__name__)


def train_ppo():
    env = SubprocVecEnv([lambda: EnvHandler(make_env())])
    learn(env=env, eval_env=None, total_timesteps=3e7, nsteps=128, nminibatches=1,
          cliprange=0.2, ent_coef=0.01, vf_coef=0.5, lam=0.95, gamma=0.99, noptepochs=4, lr=2.5e-4,
          save_interval=100, save_dir=".", load_path=None,
          normalize_observations=False, normalize_returns=False)


def view_policy_ppo():
    env = DummyVecEnv([lambda: EnvHandler(make_env())])
    view(env=env, episodes=100, total_timesteps=1000000, nsteps=200, nminibatches=1,
          cliprange=0.2, ent_coef=0.0, lam=0.95, gamma=0.99, noptepochs=4,
          save_interval=100, save_dir=".", load_path="./checkpoints/00500",
          normalize_observations=False, normalize_returns=False)


def train_ddpg():
    env = SubprocVecEnv([lambda: EnvHandler(make_env()) for _ in range(1)])
    #env = SubprocVecEnv([lambda: EnvHandler(make_env(env_no=0)), lambda: EnvHandler(make_env(env_no=1))])
    learn(env=env, seed=None, total_timesteps=1e5, nb_epochs=None, nb_epoch_cycles=10, nb_rollout_steps=100, reward_scale=1.0,
          render=False, render_eval=False, noise_type='ou-param_0.2', normalize_returns=False, normalize_observations=False,
          critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3, popart=False, gamma=0.99, clip_norm=None,
          nb_train_steps=50, nb_eval_steps=100, batch_size=64, tau=0.01, eval_env=None, param_noise_adaption_interval=50,
          nb_save_epochs=1, save_dir=".", load_path=None)


def view_policy_ddpg():
    env = DummyVecEnv([lambda: EnvHandler(make_env())])
    view(env, seed=None, total_timesteps=10000, reward_scale=1.0, render=True, render_eval=False, noise_type=None,
         normalize_returns=False, normalize_observations=False, critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3, popart=False,
         gamma=0.99, clip_norm=None, nb_train_steps=50, nb_eval_steps=100, nb_save_epochs=None, batch_size=64, 
         tau=0.01, action_range=(-250.0, 250.0), observation_range=(-5.0, 5.0), eval_env=None, load_path="./checkpoints/00007", save_dir=None,
         param_noise_adaption_interval=50)


def train_td3():
    env = SubprocVecEnv([lambda: EnvHandler(make_env()) for _ in range(1)])
    learn(env, total_timesteps=1e6, nb_epochs=None, nb_rollout_steps=100, max_ep_len=250, reward_scale=1.0,
          render=False, render_eval=False, noise_type='adaptive-param_0.2',
          normalize_returns=False, normalize_observations=True, actor_lr=1e-4, critic_lr=1e-3,
          popart=False, gamma=0.99, clip_norm=None, start_steps=10000, nb_train_steps=50,
          nb_eval_steps=100, nb_log_steps=100, nb_save_steps=None, batch_size=64,
          polyak=0.01, action_range=(-250.0, 250.0), observation_range=(-5.0, 5.0),
          target_noise=0.2, noise_clip=0.5, policy_delay=2,
          load_path=None, save_dir=None)


if __name__ == "__main__":
    #train_ddpg()
    #view_policy_ddpg()
    #train_ppo()
    #view_policy_ppo()
    train_td3()
