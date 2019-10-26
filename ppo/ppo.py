import os, sys
import time
import numpy as np
import os.path as osp

from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from runner import Runner
from model import Model


def constfn(val):
    def f(_):
        return val
    return f

def learn(env, total_timesteps, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, save_dir=None, model_fn=None, update_fn=None, init_fn=None,
            normalize_observations=True, normalize_returns=True,
            mpi_rank_weight=1, comm=None, use_tensorboard=False, tb_log_dir=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)
    Parameters:
    ----------
    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.
    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)
    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)
    ent_coef: float                   policy entropy coefficient in the optimization objective
    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.
    vf_coef: float                    value function loss coefficient in the optimization objective
    max_grad_norm: float or None      gradient norm clipping coefficient
    gamma: float                      discounting factor
    lam: float                        advantage estimation discounting factor (lambda in the paper)
    log_interval: int                 number of timesteps between logging events
    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.
    noptepochs: int                   number of training epochs per update
    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training
    save_interval: int                number of timesteps between saving events
    load_path: str                    path to load the model from
    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    model_fn = Model
    model = model_fn(ob_space=ob_space, ac_space=ac_space,
                    ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight,
                    normalize_observations=normalize_observations, normalize_returns=normalize_returns,
                    use_tensorboard=use_tensorboard, tb_log_dir=tb_log_dir)

    if load_path is not None:
        model.load(load_path)
    '''
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)
    '''

    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    #if eval_env is not None:
    #    eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, actions, returns, dones, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        #if eval_env is not None:
        #    eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)
        #if eval_env is not None:
        #    eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        '''
        mblossvals = []
        assert nenvs % nminibatches == 0
        envsperbatch = nenvs // nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        for _ in range(noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                sl_obs = [obs[i] for i in mbflatinds]
                sl_returns = np.array([returns[i] for i in mbflatinds])
                sl_actions = [actions[i] for i in mbflatinds]
                sl_dones = np.array([dones[i] for i in mbflatinds])
                sl_values = np.array([values[i] for i in mbflatinds])
                sl_neglogpacs = np.array([neglogpacs[i] for i in mbflatinds])
                mbstates = states
                mblossvals.append(model.train(lrnow, cliprangenow, sl_obs, sl_returns, sl_actions, sl_dones, sl_values, sl_neglogpacs, mbstates))
        '''
        mblossvals = []
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (returns, values, neglogpacs))
                mbstates = states
                slice_obs = {k: v[mbinds] for k, v in obs.items()}
                slice_actions = {k: v[mbinds] for k, v in actions.items()}
                mblossvals.append(model.train(lrnow, cliprangenow, slice_obs, slice_actions, *slices, mbstates))
        
        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)

        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)
            '''
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            '''
            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            if save_dir == None:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
            else:
                checkdir = osp.join(save_dir, 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            if eval_env != None:
                runner.record_render(eval_env)

    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def view(env, total_timesteps, episodes=5, seed=None, nsteps=2048, 
        load_path=None, model_fn=None, update_fn=None, init_fn=None,
        normalize_observations=False, normalize_returns=False):
  
    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Instantiate the model object (that creates act_model and train_model)
    model = Model(ob_space=ob_space, ac_space=ac_space, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    normalize_observations=normalize_observations, normalize_returns=normalize_returns)

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=0.99, lam=0.95)
    runner.render(episodes)
