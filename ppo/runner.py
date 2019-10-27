import os, sys
from copy import deepcopy
sys.path.insert(1, os.getcwd() + "/common")
sys.path.insert(1, os.getcwd() + "/policy")

import numpy as np
from abc import ABC, abstractmethod
from util_algo import convert_maObs_to_saObs, obs_reduce_dim, flatten_obs, obs_to_listObs
from util_p import listdict2dictnp

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError 

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.n_actors = env.n_actors

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        n_agents = self.obs['observation_self'][0].shape[0]

        mb_rewards = [[] for _ in range(n_agents)]
        mb_values = [[] for _ in range(n_agents)]
        mb_neglogpacs = [[] for _ in range(n_agents)]
        mb_obs = [{k: [] for k in self.env.observation_space.spaces.keys()} for _ in range(n_agents)]
        mb_actions = [{k: [] for k in self.env.action_space.spaces.keys()} for _ in range(n_agents)]
        mb_dones = [[] for _ in range(n_agents)]
        mb_states = self.states

        epinfos = []
        n_batches = self.obs['observation_self'].shape[0]

        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            agent_actions = []

            for i in range(n_agents):
                sa_obs = {k: v[0:, i] for k, v in self.obs.items()}
                actions, info = self.model.step(sa_obs)
                agent_actions.append(actions)
                mb_values[i].append(list(info['vpred']))
                mb_neglogpacs[i].append(list(info['ac_neglogp']))
                self.state = info['state']
                for k, v in actions.items():
                    mb_actions[i][k].append(v)
                for k, v in sa_obs.items():
                    mb_obs[i][k].append(v)
            
            for i in range(n_agents):
                mb_dones[i].append(self.dones)  
            
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            nenvs_actions = []
            for e in range(self.nenv):
                per_act = {k: [] for k in agent_actions[0].keys()}
                for z in range(n_agents):
                    for k, v in agent_actions[z].items():
                        per_act[k].append(v[e])
                for k, v in per_act.items():
                    per_act[k] = np.array(v[e])
                nenvs_actions.append(per_act)
                
            self.obs, rewards, self.dones, infos = self.env.step(nenvs_actions)

            for i in range(n_agents):
                mb_rewards[i].append(rewards[:, i])
        
        #batch of steps to batch of rollouts
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        last_values = [[] for _ in range(n_agents)]
        for i in range(n_agents):
            sa_obs = {k: v[0:, i] for k, v in self.obs.items()}
            last_values[i] = np.squeeze(self.model.value(sa_obs), -1)

        infos = [{'r': np.mean(mb_rewards), 'l': 250}]
        epinfos += infos

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = [0 for _ in range(n_agents)]

        for i in range(n_agents):
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values[i]
                else:
                    nextnonterminal = 1.0 - mb_dones[i][t+1]
                    nextvalues = mb_values[i][t+1]
                delta = mb_rewards[i][t] + self.gamma * nextvalues * nextnonterminal - mb_values[i][t]
                mb_advs[i][t] = lastgaelam[i] = delta + self.gamma * self.lam * nextnonterminal * lastgaelam[i]
            mb_returns[i] = mb_advs[i] + mb_values[i]

            mb_obs[i], mb_actions[i] = dsf01(mb_obs[i]), dsf01(mb_actions[i])

        mb_returns, mb_dones, mb_values, mb_neglogpacs = sf01(mb_returns, n_agents), sf01(mb_dones, n_agents), sf01(mb_values, n_agents), sf01(mb_neglogpacs, n_agents)
        return mb_obs, mb_actions, mb_returns, mb_dones, mb_values, mb_neglogpacs, mb_states, epinfos
    
    def record_render(self, eval_env):
        obs = eval_env.reset()
        while True:
            actions, info = self.model.step(obs_reduce_dim(obs, 0))
            obs, rewards, dones, infos = eval_env.step([actions])
            eval_env.render()
            if True in dones:
                break
    
    def render(self, episodes):
        obs = self.env.reset()
        n_batches = obs['observation_self'].shape[0]
        n_agents = obs['observation_self'][0].shape[0]
        for _ in range(episodes):
            while True:
                #actions, info = self.model.step(flatten_obs(obs), n_agents, n_batches)
                actions, info = self.model.step(obs)
                nenvs_actions = [{'action_movement' : actions['action_movement'][i*n_agents:(i + 1)*n_agents]} for i in range(self.nenv)]
                obs, rewards, dones, infos = self.env.step(nenvs_actions)
                self.env.render()
                if True in dones:
                    break


def sf01(arr, n_agents):
    """
    swap and then flatten axes 0 and 1
    """
    new_arr = []
    for a in arr:
        s = a.shape
        new_arr.append(a.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:]))
    return np.asarray(new_arr)


def dsf01(dic):
    """
    swap and then flatten axes 0 and 1 of dict
    """
    for k, v in dic.items():
        v = np.array(v)
        s = v.shape
        v = v.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        v = np.expand_dims(v, 1)
        dic[k] = v
    return dic
