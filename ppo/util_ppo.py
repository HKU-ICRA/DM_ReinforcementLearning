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
