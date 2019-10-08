import os, sys
sys.path.insert(1, os.getcwd() + "/comoon")
from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import baselines.common.tf_util as U
from baselines.common.tf_util import get_session, save_variables, load_variables
from normalizers import EMAMeanStd
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common.distributions import make_pdtype
from variable_schema import VariableSchema, BATCH, TIMESTEPS
from baselines.common.mpi_running_mean_std import RunningMeanStd
from util_algo import obs_reduce_dim, batch_obs_to_dictObs, batch_act_to_dictAct
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

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


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    updates = []
    for var, perturbed_var in zip(actor.actor_variables, perturbed_actor.actor_variables):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class DDPG(object):
    def __init__(self, network, memory, ob_space, ac_space, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., ema_beta=0.99999):

        # Create target network.
        target_network = copy(network)
        self.target_network = target_network
        
        # obs0
        self.ob_space = ob_space
        self.obs0 = dict()
        for k, v in self.ob_space.spaces.items():
            self.obs0[k] = network.phs[k]

        # obs1
        self.obs1 = dict()
        for k, v in self.ob_space.spaces.items():
            self.obs1[k] = target_network.phs[k]

        # actions
        self.ac_space = ac_space
        self.ac_space.spaces = {k: v.spaces[0] for k, v in self.ac_space.spaces.items()}
        pdtypes = {k: make_pdtype(s) for k, s in ac_space.spaces.items()}
        self.actions = dict()
        for k, pdtype in pdtypes.items():
            self.actions[k] = VariableSchema(shape=[BATCH, TIMESTEPS] + pdtype.sample_shape(),
                              dtype=pdtype.sample_dtype())
            self.actions[k] = self.actions[k].placeholder(name=k)

        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.network = network
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self._ema_beta = ema_beta
        self._running_mean_stds = {}

        self.obs_rms = network.obs_rms
        self.ret_rms = network.value_rms

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = network.sampled_action
        self.normalized_critic_tf = network.value
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = network.value
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(target_network.value, self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(self.obs0)
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

        self.initial_state = None # recurrent architectures not supported yet

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.network.actor_variables, self.target_network.actor_variables, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.network.critic_variables, self.target_network.critic_variables, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.network)
        self.perturbed_actor_tf = param_noise_actor.sampled_action
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.network, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.network)
        adaptive_actor_tf = adaptive_param_noise_actor.sampled_action
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.network, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.network.actor_variables]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.network.actor_variables, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.network.actor_variables,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.network.critic_variables if var.name.endswith('/kernel:0') and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.network.critic_variables]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.network.critic_variables, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.network.critic_variables,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.network.critic_output_vars, self.target_network.critic_output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']
        
        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        self.stats_ops = ops
        self.stats_names = names

    def step(self, obs, apply_noise=True, compute_Q=True):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        if compute_Q:
            action = self.network.act(obs_reduce_dim(obs, 0))
            q = self.network.qvalue(self.critic_with_actor_tf, obs_reduce_dim(obs, 0), action)
        else:
            action = self.network.act(obs_reduce_dim(obs, 0))
            q = None

        if self.action_noise is not None and apply_noise:
            for k, v in self.action_noise.items():
                noise = self.action_noise[k]()
                assert noise.shape == action[k][0].shape
                action[k] += noise
        
        for k, v in action.items():
            action[k] = np.clip(v, self.action_range[0], self.action_range[1])
        
        return action, q, None, None

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale

        B = obs0['observation_self'].shape[0]
        for b in range(B):
            #self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])
            self.memory.append(obs0, action, reward, obs1, terminal1)

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })
        else:
            target_action = self.target_network.act(batch_obs_to_dictObs(batch['obs1']))
            extra_feed_dict = {
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32')
            }
            target_Q = self.target_network.qvalue(self.target_Q, batch_obs_to_dictObs(batch['obs1']), target_action, extra_feed_dict)
            target_Q = np.expand_dims(target_Q, -1)
        
        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        extra_feed_dict = {
            self.critic_target: target_Q
        }
        actor_grads, actor_loss, critic_grads, critic_loss = self.network.tensor_eval(ops, batch_obs_to_dictObs(batch['obs0']),
                                                                                      batch_act_to_dictAct(batch['actions']), extra_feed_dict)

        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.network.tensor_eval(self.stats_ops, batch_obs_to_dictObs(self.stats_sample['obs0']),
                                          action=batch_act_to_dictAct(self.stats_sample['actions']))

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            for k, v in self.action_noise.items():
                self.action_noise[k].reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })
    