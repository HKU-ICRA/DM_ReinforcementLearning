import os, sys
import os.path as osp
import time
from collections import deque
import pickle

import numpy as np
import functools

from ddpg_learner import DDPG
from ddpgpolicy import DdpgPolicy
from memory import Memory
from noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from baselines.common.tf_util import save_variables, load_variables
from baselines import logger
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def learn(env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          nb_save_epochs=None,
          batch_size=64, # per MPI worker
          tau=0.01,
          action_range=(-250.0, 250.0),
          observation_range=(-5.0, 5.0),
          eval_env=None,
          load_path=None,
          save_dir=None,
          param_noise_adaption_interval=50,
          **network_kwargs):

    set_global_seeds(seed)

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    
    memory = Memory(limit=int(1e6))

    network_spec = [
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            },
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            }
        ]
    
    vnetwork_spec = [
            {
                'layer_type': 'concat',
                'nodes_in': ['action_movement', 'observation_self'],
                'nodes_out': ['main']
            },
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            },
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            }
        ]

    network = DdpgPolicy(scope="ddpg", ob_space=env.observation_space, ac_space=env.action_space, network_spec=network_spec, v_network_spec=vnetwork_spec,
                 stochastic=False, reuse=False, build_act=True,
                 trainable_vars=None, not_trainable_vars=None,
                 gaussian_fixed_var=False, weight_decay=0.0, ema_beta=0.99999,
                 normalize_observations=normalize_observations, normalize_returns=normalize_returns,
                 observation_range=observation_range)

    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                action_noise = dict()
                for k, v in env.action_space.spaces.items():
                    act_size = v.spaces[0].shape[-1]
                    _, stddev = current_noise_type.split('_')
                    action_noise[k] = NormalActionNoise(mu=np.zeros(act_size), sigma=float(stddev) * np.ones(act_size))
            elif 'ou' in current_noise_type:
                action_noise = dict()
                for k, v in env.action_space.spaces.items():
                    act_size = v.spaces[0].shape[-1]
                    _, stddev = current_noise_type.split('_')
                    action_noise[k] = OrnsteinUhlenbeckActionNoise(mu=np.zeros(act_size), sigma=float(stddev) * np.ones(act_size))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_action = action_range[1]
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    agent = DDPG(network, memory, env.observation_space, env.action_space,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    sess = U.get_session()

    saver = functools.partial(save_variables, sess=sess)
    loader = functools.partial(load_variables, sess=sess)
    if load_path != None:
        loader(load_path)

    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()

    agent.reset()
    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = env.num_envs
    n_agents = obs['observation_self'].shape[0]

    episode_reward = np.zeros((nenvs, n_agents), dtype = np.float32) #vector
    episode_step = np.zeros(nenvs, dtype = int) # vector
    episodes = 0 #scalar
    t = 0 # scalar

    epoch = 0

    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                # of the environments, so resetting here instead
                agent.reset()
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)

                # Execute next action.
                if rank == 0 and render:
                    env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                for k, v in action.items():
                    action[k] *= max_action
                
                nenvs_actions = []
                for i in range(nenvs):
                    nenv_action = {'action_movement' : action['action_movement'][i*n_agents:(i + 1)*n_agents]}
                    nenvs_actions.append(nenv_action)
                new_obs, r, done, info = env.step(nenvs_actions)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                # note these outputs are batched from vecenv

                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(obs, action, r, new_obs, done) #the batched data will be unrolled in memory.py's append.

                obs = new_obs

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        if nenvs == 1:
                            agent.reset()

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/return_history_std'] = np.std(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)
        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)
        
        if nb_save_epochs != None and (epoch + 1) % nb_save_epochs == 0:
            if save_dir == None:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
            else:
                checkdir = osp.join(save_dir, 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%epoch)
            print('Saving to', savepath)
            saver(savepath)

    return agent

def view(env,
          seed=None,
          total_timesteps=None,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          nb_save_epochs=None,
          batch_size=64, # per MPI worker
          tau=0.01,
          action_range=(-250.0, 250.0),
          observation_range=(-5.0, 5.0),
          eval_env=None,
          load_path=None,
          save_dir=None,
          param_noise_adaption_interval=50,
          **network_kwargs):

    set_global_seeds(seed)

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    memory = Memory(limit=int(1e6))
    
    network_spec = [
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            },
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            }
        ]
    
    vnetwork_spec = [
            {
                'layer_type': 'concat',
                'nodes_in': ['action_movement', 'observation_self'],
                'nodes_out': ['main']
            },
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            },
            {
                'layer_type': 'dense',
                'units': int (128),
                'activation': 'relu',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            }
        ]

    network = DdpgPolicy(scope="ddpg", ob_space=env.observation_space, ac_space=env.action_space, network_spec=network_spec, v_network_spec=vnetwork_spec,
                 stochastic=False, reuse=False, build_act=True,
                 trainable_vars=None, not_trainable_vars=None,
                 gaussian_fixed_var=False, weight_decay=0.0, ema_beta=0.99999,
                 normalize_observations=normalize_observations, normalize_returns=normalize_returns,
                 observation_range=observation_range)

    max_action = action_range[1]
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    agent = DDPG(network, memory, env.observation_space, env.action_space,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=None, param_noise=None, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    sess = U.get_session()

    loader = functools.partial(load_variables, sess=sess)
    if load_path != None:
        loader(load_path)

    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()

    nenvs = env.num_envs
    obs = env.reset()
    n_agents = obs['observation_self'].shape[0]

    for epoch in range(total_timesteps):
        agent.reset()
        obs = env.reset()
        while True:
            action, q, _, _ = agent.step(obs, apply_noise=False, compute_Q=False)
            # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
            for k, v in action.items():
                action[k] *= max_action
            nenvs_actions = []
            for i in range(nenvs):
                nenv_action = {'action_movement' : action['action_movement'][i*n_agents:(i + 1)*n_agents]}
                nenvs_actions.append(nenv_action)
            obs, r, done, info = env.step(nenvs_actions)
            env.render()
            if True in done:
                break
