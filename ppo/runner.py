import os, sys
from copy import deepcopy
sys.path.insert(1, os.getcwd() + "/common")

import numpy as np
from abc import ABC, abstractmethod
from util_algo import convert_maObs_to_saObs, obs_reduce_dim, flatten_obs, obs_to_listObs

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1

        #for k, v in env.observation_space.items():
        #    self.batch_ob_shape[k] = (nenv*nsteps,) + v.shape
        self.obs = env.reset()#[env.reset() for _ in range(nenv)]
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
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        all_nsteps = int(self.nsteps / self.env.nremotes)
        last_nsteps = self.nsteps - all_nsteps * self.env.nremotes

        n_batches = self.obs['observation_self'].shape[0]
        n_agents = self.obs['observation_self'][0].shape[0]

        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            #obs = flatten_obs(self.obs)

            #actions, info = self.model.step(obs, n_agents, n_batches)
            actions, info = self.model.step(obs_reduce_dim(deepcopy(self.obs), 0))
            info = deepcopy(info)
            actions = deepcopy(actions)
            #mb_values += list(info['vpred'])
            list_values = list(info['vpred'])
            mb_values.append(list_values)
            self.state = info['state']
            #mb_neglogpacs += list(info['ac_logp'])
            list_neglogpacs = list(info['ac_neglogp'])
            mb_neglogpacs.append(list_neglogpacs)
            mb_actions.append(convert_maObs_to_saObs(actions, 0))
            #mb_dones += [d for d in self.dones for _ in range(n_agents)]
            mb_dones.append(self.dones)
            #mb_obs += obs_to_listObs(obs, n_agents, n_batches)
            obs_cpy = deepcopy(self.obs)
            mb_obs.append(convert_maObs_to_saObs(obs_reduce_dim(obs_cpy, 0), 0))
            
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            nenvs_actions = []
            for i in range(self.nenv):
                nenv_action = {'action_movement' : actions['action_movement'][i*n_agents:(i + 1)*n_agents]}
                nenvs_actions.append(nenv_action)

            #self.env.step_async(nenvs_actions)
            #self.obs, rewards, self.dones, infos = self.env.step_wait()
            self.obs, rewards, self.dones, infos = self.env.step(nenvs_actions)

            #mb_rewards += [r for r_list in rewards for r in r_list]
            mb_rewards.append(rewards[0])
        
        #batch of steps to batch of rollouts
        #mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        #mb_actions = np.asarray(mb_actions)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float64)
        mb_values = np.asarray(mb_values, dtype=np.float64)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float64)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(obs_reduce_dim(deepcopy(self.obs), 0))[0]

        infos = [{'r': np.mean(mb_rewards), 'l': 250}]
        epinfos += infos

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values
        return np.asarray(mb_obs), np.asarray(mb_actions), (*map(sf01, (mb_returns, mb_dones, mb_values, mb_neglogpacs))), mb_states, epinfos
    
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
                actions, info = self.model.step(flatten_obs(obs))
                nenvs_actions = []
                for i in range(self.nenv):
                    nenv_action = {'action_movement' : actions['action_movement'][i*n_agents:(i + 1)*n_agents]}
                    nenvs_actions.append(nenv_action)
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
