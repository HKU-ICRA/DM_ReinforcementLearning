import numpy as np
from abc import ABC, abstractmethod
from util_ppo import convert_maObs_to_saObs

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


def obs_reduce_dim(obs, axis):
    reduced_obs = dict()
    for k, v in obs.items():
        reduced_obs[k] = np.array(v[axis])
    return reduced_obs

def flatten_obs(obs):
    flat_obs = dict()
    for k, v in obs.items():
        vals = []
        for e in v:
            for f in e:
                vals.append(f)
        flat_obs[k] = np.array(vals)
    return flat_obs

def obs_to_listObs(obs, n_agents, n_batches):
    new_obs = []
    for i in range(n_agents * n_batches):
        ob = dict()
        for k, v in obs.items():
            ob[k] = v[i]
        new_obs.append(ob)
    return new_obs    

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

        for _ in range(all_nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            #obs = flatten_obs(self.obs)

            #actions, info = self.model.step(obs, n_agents, n_batches)
            actions, info = self.model.step(obs_reduce_dim(self.obs, 0))
            #mb_values += list(info['vpred'])
            list_values = list(info['vpred'])
            mb_values.append(list_values[0])
            self.state = info['state']
            #mb_neglogpacs += list(info['ac_logp'])
            list_neglogpacs = list(info['ac_logp'])
            mb_neglogpacs.append(list_neglogpacs[0])
            mb_actions.append(convert_maObs_to_saObs(actions, 0))
            #mb_dones += [d for d in self.dones for _ in range(n_agents)]
            mb_dones.append(self.dones[0])
            #mb_obs += obs_to_listObs(obs, n_agents, n_batches)
            mb_obs.append(convert_maObs_to_saObs(obs_reduce_dim(self.obs, 0), 0))
            
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
            mb_rewards.append(rewards[0][0])
        
        #batch of steps to batch of rollouts
        #mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        #mb_actions = np.asarray(mb_actions)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float64)
        mb_values = np.asarray(mb_values, dtype=np.float64)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float64)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = 0 # last value is just zero

        infos = [{'r': np.mean(mb_rewards), 'l': 250}]
        epinfos += infos

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        '''
        for b in range(n_batches):
            for t in reversed(range(b, all_nsteps - n_batches + b + 1, n_batches)):
                for a in range(n_agents):
                    if n_agents * t + a == n_agents * (all_nsteps - n_batches + b) + a:
                        nextnonterminal = 1.0 - mb_dones[n_agents * t + a]
                        nextvalues = last_values
                    else:
                        nextnonterminal = 1.0 - mb_dones[n_agents * t + a + 1]
                        nextvalues = mb_values[n_agents * t + a + 1]
                    delta = mb_rewards[n_agents * t + a] + self.gamma * nextvalues * nextnonterminal - mb_values[n_agents * t + a]
                    mb_advs[n_agents * t + a] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam\
        '''
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
