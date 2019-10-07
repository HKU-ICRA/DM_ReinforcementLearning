import os, sys
sys.path.insert(1, os.getcwd() + '/envs')
sys.path.insert(1, os.getcwd() + '/ppo')

import logging
from base import  make_env
from envhandler import EnvHandler
from dummy_vec_env import DummyVecEnv
from subproc_vec_env import SubprocVecEnv
from ppo import learn, view
from gym.wrappers import Monitor

import numpy as np

logger = logging.getLogger(__name__)

def train():
    #env = SubprocVecEnv([lambda: EnvHandler(make_env(env_no=0)), lambda: EnvHandler(make_env(env_no=1))])
    env = SubprocVecEnv([lambda: EnvHandler(make_env(env_no=0))])

    learn(env=env, eval_env=None, total_timesteps=1e6, nsteps=200, nminibatches=1,
          cliprange=0.2, ent_coef=0.01, vf_coef=0.5, lam=0.95, gamma=0.99, noptepochs=10, lr=3e-4,
          save_interval=100, save_dir=".", load_path=None)

def view_policy():
    env = DummyVecEnv([lambda: EnvHandler(make_env(env_no=0))])

    view(env=env, episodes=100, total_timesteps=1000000, nsteps=200, nminibatches=1,
          cliprange=0.2, ent_coef=0.0, lam=0.95, gamma=0.99, noptepochs=4,
          save_interval=100, save_dir=".", load_path="./checkpoints/00400")

if __name__ == "__main__":
    train()
    #view_policy()
