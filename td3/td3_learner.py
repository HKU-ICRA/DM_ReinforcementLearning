import numpy as np
import tensorflow as tf
import gym
import time

import baselines.common.tf_util as U
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
from util_td3 import obs_reduce_dim, batch_obs_to_dictObs, batch_act_to_dictAct
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines import logger

def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)

class TD3():
    
    def __init__(self, env, network, target_network, memory, ac_space, ob_space, ac_kwargs=dict(),
        steps_per_epoch=5000, epochs=100, gamma=0.99, 
        polyak=0.995, actor_lr=1e-3, critic_lr=1e-3, batch_size=100, start_steps=10000, 
        action_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):

        self.network = network
        self.target_network = target_network
        self.memory = memory
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
        self.batch_size = batch_size
        self.action_noise = action_noise
        self.policy_delay = policy_delay
        self.stats_sample = None

        # obs0
        self.obs0 = dict()
        for k, v in self.ob_space.spaces.items():
            self.obs0[k] = network.phs[k]

        # obs1
        self.obs1 = dict()
        for k, v in self.ob_space.spaces.items():
            self.obs1[k] = target_network.phs[k]

        # acts0
        self.acts0 = dict()
        for k, v in self.ac_space.spaces.items():
            self.acts0[k] = network.phs[k]

        # acts1
        self.acts1 = dict()
        for k, v in self.ac_space.spaces.items():
            self.acts1[k] = target_network.phs[k]

        self.terminals1 = terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name='rewards')

        # Network
        self.actor_tf = network.actions
        self.critic_tf = critic_tf = network.value0
        self.critic_twin_tf = critic_twin_tf = network.value0_twin
        self.critic_with_actor_tf = critic_with_actor_tf = network.value1

        # Target network
        self.target_actor_tf = target_network.actions
        self.target_critic_tf = target_critic_tf = target_network.value0
        self.target_critic_twin_tf = target_critic_twin_tf = target_network.value0_twin
        self.target_critic_with_target_actor_tf = target_network.value1

        # Bellman backup for Q functions, using Clipped Double-Q targets
        self.min_q_targ = min_q_targ = tf.minimum(target_critic_tf, target_critic_twin_tf)
        self.backup = tf.stop_gradient(rewards + gamma * (1 - terminals1) * min_q_targ)

        # Init
        self.setup_target_network_updates()
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        self.setup_stats()

    def setup_critic_optimizer(self):
        critic_loss = tf.reduce_mean((self.critic_tf - self.backup)**2)
        critic_twin_loss = tf.reduce_mean((self.critic_twin_tf - self.backup)**2)
        self.critic_losses = critic_losses = critic_loss + critic_twin_loss
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        self.train_critic_op = critic_optimizer.minimize(critic_losses, var_list=self.network.critic_variables)

    def setup_actor_optimizer(self):
        self.actor_loss = actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        self.train_actor_op = actor_optimizer.minimize(actor_loss, var_list=self.network.actor_variables)

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.network.actor_variables, self.target_network.actor_variables, self.polyak)
        critic_init_updates, critic_soft_updates = get_target_updates(self.network.critic_variables, self.target_network.critic_variables, self.polyak)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)

    def setup_stats(self):
        ops = []
        names = []

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']

        ops += [tf.reduce_mean(self.critic_twin_tf)]
        names += ['reference_Q_mean']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']

        ops += [self.actor_tf]
        names += ['reference_action_mean']

        self.stats_ops = ops
        self.stats_names = names

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        B = obs0['observation_self'].shape[0]
        for b in range(B):
            self.memory.append(obs0, action, reward, obs1, terminal1)

    def step(self, obs, apply_noise=True, compute_Q=True):
        actor_tf = self.actor_tf

        feed_dict = {self.obs0[k]: v for k, v in obs.items()}

        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        if self.action_noise is not None and apply_noise:
            for k, v in self.action_noise.items():
                noise = self.action_noise[k]()
                assert noise.shape == action[k][0][0].shape
                action[k] += noise
        
        for k, v in action.items():
            action[k] = np.clip(v, self.action_range[0], self.action_range[1])
        
        return action, q, None, None
    
    def train(self, episode_step):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)
        
        # Get all gradients and perform a synced update.
        ops = [self.critic_losses, self.critic_tf, self.critic_twin_tf, self.train_critic_op]
        
        br = []
        for r in batch['rewards']:
            br.append([r])

        feed_dict = {self.acts0[k] : v for k, v in batch_act_to_dictAct(batch['actions']).items()}
        acts1_feed_dict = {self.acts1[k] : v for k, v in batch_act_to_dictAct(batch['actions']).items()}
        obs0_feed_dict = {self.obs0[k] : v for k, v in batch_obs_to_dictObs(batch['obs0']).items()}
        obs1_feed_dict = {self.obs1[k] : v for k, v in batch_obs_to_dictObs(batch['obs1']).items()}
        extra_feed_dict = {
            self.rewards : br,
            self.terminals1 : batch['terminals1'].astype('float32'),
        }

        feed_dict.update(acts1_feed_dict)
        feed_dict.update(obs0_feed_dict)
        feed_dict.update(obs1_feed_dict)
        feed_dict.update(extra_feed_dict)

        critic_losses, critic, critic_twin, _ = self.sess.run(ops, feed_dict=feed_dict)

        if episode_step % self.policy_delay  == 0:
            outs = self.sess.run([self.actor_loss, self.train_actor_op], feed_dict)
            self.update_target_net()
            actor_loss = outs[0]
        else:
            actor_loss = None

        return critic_losses, critic, critic_twin, actor_loss

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)

        feed_dict = {self.acts0[k] : v for k, v in batch_act_to_dictAct(self.stats_sample['actions']).items()}
        acts1_feed_dict = {self.acts1[k] : v for k, v in batch_act_to_dictAct(self.stats_sample['actions']).items()}
        obs0_feed_dict = {self.obs0[k] : v for k, v in batch_obs_to_dictObs(self.stats_sample['obs0']).items()}
        obs1_feed_dict = {self.obs1[k] : v for k, v in batch_obs_to_dictObs(self.stats_sample['obs1']).items()}

        feed_dict.update(acts1_feed_dict)
        feed_dict.update(obs0_feed_dict)
        feed_dict.update(obs1_feed_dict)
        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        actions, actions_names = [], []
        cnt = 0
        for k, v in values[-1].items():
            num = np.sum(v)
            actions.append(num)
            actions_names.append("reference_action_mean" + str(cnt))
            cnt += 1

        names = self.stats_names[:]
        values[-1] = actions[0]
        if len(actions) > 1:
            values = values + actions[1:]
            names = names + actions_names
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        return stats

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            for k, v in self.action_noise.items():
                self.action_noise[k].reset()
