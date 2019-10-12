import os, sys
import os.path as osp
import time
from collections import deque
import pickle

import numpy as np
import functools

from td3_learner import TD3
from td3policy import Td3Policy
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
          total_timesteps=1e6,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_rollout_steps=100,
          max_ep_len=250,
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
          start_steps=10000,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          nb_log_steps=None,
          nb_save_steps=None,
          batch_size=64, # per MPI worker
          polyak=0.01,
          action_range=(-250.0, 250.0),
          observation_range=(-5.0, 5.0),
          target_noise=0.2,
          noise_clip=0.5,
          policy_delay=2,
          eval_env=None,
          load_path=None,
          save_dir=None,
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
                'units': int (256),
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
            },
            {
                'layer_type': 'dense',
                'units': int (1),
                'activation': 'tanh',
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
                'units': int (256),
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
            },
            {
                'layer_type': 'dense',
                'units': int (1),
                'activation': '',
                'nodes_in': ['main'],
                'nodes_out': ['main']
            }
        ]

    network = Td3Policy(scope="td3", ob_space=env.observation_space, ac_space=env.action_space, network_spec=network_spec,
                 v_network_spec=vnetwork_spec,
                 stochastic=False, reuse=False, build_act=True,
                 trainable_vars=None, not_trainable_vars=None,
                 gaussian_fixed_var=False, weight_decay=0.0, ema_beta=0.99999,
                 normalize_observations=normalize_observations, normalize_returns=normalize_returns,
                 observation_range=observation_range,
                 action_range=action_range,
                 target_noise=target_noise,
                 noise_clip=noise_clip)
    
    target_network = Td3Policy(scope="target", ob_space=env.observation_space, ac_space=env.action_space, network_spec=network_spec,
                 v_network_spec=vnetwork_spec,
                 stochastic=False, reuse=False, build_act=True,
                 trainable_vars=None, not_trainable_vars=None,
                 gaussian_fixed_var=False, weight_decay=0.0, ema_beta=0.99999,
                 normalize_observations=normalize_observations, normalize_returns=normalize_returns,
                 observation_range=observation_range,
                 action_range=action_range,
                 target_noise=target_noise,
                 noise_clip=noise_clip,
                 isTarget=True)

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

    agent = TD3(env, network, target_network, memory, env.action_space, env.observation_space, 
        steps_per_epoch=nb_rollout_steps, epochs=nb_epochs, gamma=gamma, 
        polyak=polyak, actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, start_steps=start_steps, 
        action_noise=action_noise, target_noise=target_noise, noise_clip=noise_clip, policy_delay=policy_delay)

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

    for t in range(int(total_timesteps)):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)
            nenvs_actions = []
            for i in range(nenvs):
                nenv_action = {'action_movement' : action['action_movement'][i*n_agents:(i + 1)*n_agents]}
                nenvs_actions.append(nenv_action)
        else:
            action, q = env.action_space.sample(), None
            nenvs_actions = []
            for i in range(nenvs):
                nenv_action = {'action_movement' : action['action_movement'][i*n_agents:(i + 1)*n_agents][0]}
                nenvs_actions.append(nenv_action)

        new_obs, r, done, info = env.step(nenvs_actions)

        episode_reward += r
        episode_step += 1

        for d in range(len(done)):
            done[d] = False if episode_step==max_ep_len else done[d]

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
        
        episode_actor_losses = []
        episode_critic_losses = []
        episode_critic = []
        episode_critic_twin = []
        if d or (episode_step[0] == max_ep_len):
            """
            Perform all TD3 updates at the end of the trajectory
            (in accordance with source code of TD3 published by
            original authors).
            """
            for j in range(episode_step[0]):
                critic_loss, critic, critic_twin, actor_loss = agent.train(episode_step[0])

                episode_critic_losses.append(critic_loss)
                episode_critic.append(critic)
                episode_critic_twin.append(critic_twin)
                if actor_loss is not None:
                    episode_actor_losses.append(actor_loss)

            obs, r, done, episode_reward, episode_step = env.reset(), 0, False, np.zeros((nenvs, n_agents), dtype = np.float32), np.zeros(nenvs, dtype = int)
        
        if (t + 1) % nb_log_steps == 0:
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
            combined_stats['train/loss_actor'] = np.mean(episode_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(episode_critic_losses)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
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

        if nb_save_steps != None and (t + 1) % nb_save_steps == 0:
            if save_dir == None:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
            else:
                checkdir = osp.join(save_dir, 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%t)
            print('Saving to', savepath)
            saver(savepath)

    return agent
