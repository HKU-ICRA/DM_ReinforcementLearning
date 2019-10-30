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
                obs_dc = deepcopy(self.obs)
                sa_obs = {k: v[:, i:i+1, :] for k, v in obs_dc.items()}
                actions, info = self.model.step(sa_obs)
                actions_dc = deepcopy(actions)
                info_dc = deepcopy(info)
                agent_actions.append(actions_dc)
                mb_values[i].append(info_dc['vpred'])
                mb_neglogpacs[i].append(info_dc['ac_neglogp'])
                self.state = info_dc['state']
                mb_dones[i].append(deepcopy(self.dones))  
                for k, v in actions_dc.items():
                    mb_actions[i][k].append(v)
                for k, v in sa_obs.items():
                    mb_obs[i][k].append(v)
                
            # Take actions in env and look the results
            nenvs_actions = []
            for e in range(self.nenv):
                per_act = {k: [] for k in agent_actions[0].keys()}
                for z in range(n_agents):
                    for k, v in agent_actions[z].items():
                        per_act[k].append(v[e])
                for k, v in per_act.items():
                    per_act[k] = np.array(v)
                nenvs_actions.append(per_act)
            
            self.obs, rewards, self.dones, infos = self.env.step(nenvs_actions)

            for i in range(n_agents):
                mb_rewards[i].append(deepcopy(rewards)[:, i:i+1])
        
        #batch of steps to batch of rollouts
        last_values = [[] for _ in range(n_agents)]
        for i in range(n_agents):
            mb_rewards[i] = np.squeeze(np.asarray(mb_rewards[i], dtype=np.float32), -1)
            mb_values[i] = np.asarray(mb_values[i], dtype=np.float32)
            mb_neglogpacs[i] = np.asarray(mb_neglogpacs[i], dtype=np.float32)
            mb_dones[i] = np.asarray(mb_dones[i], dtype=np.bool)
            sa_obs = {k: v[:, i:i+1, :] for k, v in deepcopy(self.obs).items()}
            last_values[i] = np.squeeze(deepcopy(self.model.value(sa_obs)), -1)

        infos = [{'r': np.mean(mb_rewards), 'l': 250}]
        epinfos += infos

        # discount/bootstrap off value fn
        mb_returns = [np.zeros_like(mb_rewards[i]) for i in range(n_agents)]
        mb_advs = [np.zeros_like(mb_rewards[i]) for i in range(n_agents)]
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
            mb_returns[i], mb_dones[i], mb_values[i], mb_neglogpacs[i] = sf01(mb_returns[i]), sf01(mb_dones[i]), sf01(mb_values[i]), sf01(mb_neglogpacs[i])

        mb_obs = dconcat(np.asarray(mb_obs))
        mb_actions = dconcat(np.asarray(mb_actions))

        mb_returns = f01(np.asarray(mb_returns))
        mb_dones = f01(np.asarray(mb_dones))
        mb_values = f01(np.asarray(mb_values))
        mb_neglogpacs = f01(np.asarray(mb_neglogpacs))

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
                agent_actions = []
                for i in range(n_agents):
                    sa_obs = {k: v for k, v in self.obs.items()}
                    actions, info = self.model.step(sa_obs)
                    agent_actions.append(actions)

                nenvs_actions = []
                for e in range(self.nenv):
                    per_act = {k: [] for k in agent_actions[0].keys()}
                    for z in range(n_agents):
                        for k, v in agent_actions[z].items():
                            per_act[k].append(v[e])
                    for k, v in per_act.items():
                        per_act[k] = np.array([v[e]])
                    nenvs_actions.append(per_act)
                    nenvs_actions = [{'action_movement' : actions['action_movement'][i*n_agents:(i + 1)*n_agents]} for i in range(self.nenv)]
                    obs, rewards, dones, infos = self.env.step(nenvs_actions)
                    self.env.render()
                    if True in dones:
                        break


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def f01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.reshape(s[0] * s[1], *s[2:])

def dsf01(dic):
    """
    swap and then flatten axes 0 and 1 of dict
    """
    for k, v in dic.items():
        dic[k] = sf01(np.asarray(v))
    return dic

def dconcat(arrdics):
    """
    Concat arrays along axes 0
    """
    new_arrdics = arrdics[0]
    for i in range(1, len(arrdics)):
        for k, v in arrdics[i].items():
            new_arrdics[k] = np.concatenate((new_arrdics[k], v), axis=0)
    return new_arrdics
