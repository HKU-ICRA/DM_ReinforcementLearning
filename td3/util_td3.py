import numpy as np

def convert_maObs_to_saObs(ma_obs, actor_no):
    sa_obs = dict()
    for k, v in ma_obs.items():
        sa_obs[k] = np.array([np.array(v)[actor_no]])
    return sa_obs

def all_maObs_to_saObs(ma_obs):
    all_obs = []
    for k, v in ma_obs.items():
        n_agents = np.array(v).shape[0]
    for i in range(n_agents):
        sa_obs = dict()
        for k, v in ma_obs.items():
            sa_obs[k] = np.array([np.array(v)[i]])
        all_obs.append(sa_obs)
    return all_obs

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

def batch_obs_to_dictObs(obs):
    new_obs = dict()
    for k, v in obs[0][0].items():
        new_obs[k] = []
    for ob in obs:
        for k, v in ob[0].items():
            new_obs[k].append(v[0])
    for k, v in obs[0][0].items():
        new_obs[k] = np.array(new_obs[k])
    return new_obs

def batch_act_to_dictAct(acts):
    new_acts = dict()
    for k, v in acts[0][0].items():
        new_acts[k] = []
    for act in acts:
        for k, v in act[0].items():
            new_acts[k].append(v)
    for k, v in acts[0][0].items():
        new_acts[k] = np.array(new_acts[k])
    return new_acts
