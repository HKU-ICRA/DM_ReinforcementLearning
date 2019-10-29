import os, sys
sys.path.insert(1, os.getcwd() + "/policy")

import tensorflow as tf
import numpy as np
from collections import OrderedDict
from copy import deepcopy, copy

from ma_policy import MAPolicy
from util_p import listdict2dictnp, normc_initializer, shape_list, l2_loss
from graph_construct import construct_tf_graph

class SacPolicy(MAPolicy):
    
    def __init__(self, scope, *, ob_space, ac_space, network_spec, v_network_spec=None, v_network_spec2=None
                 stochastic=True, reuse=False, build_act=True,
                 trainable_vars=None, not_trainable_vars=None,
                 gaussian_fixed_var=True, weight_decay=0.0, ema_beta=0.99999,
                 normalize_observations=True, normalize_returns=True,
                 observation_range=(-5.0, 5.0),
                 LOG_STD_MIN=0.1, LOG_STD_MAX=0.9,
                 **kwargs):
        self.v_network_spec2 = v_network_spec2
        self.LOG_STD_MIN = LOG_STD_MIN
        self.LOG_STD_MAX = LOG_STD_MAX
        super(SacPolicy, self).__init__(scope=scope, ob_space=ob_space, ac_space=ac_space,
                 network_spec=network_spec, v_network_spec=v_network_spec,
                 stochastic=stochastic, reuse=reuse, build_act=build_act,
                 trainable_vars=trainable_vars, not_trainable_vars=not_trainable_vars,
                 gaussian_fixed_var=gaussian_fixed_var, weight_decay=weight_decay, ema_beta=ema_beta,
                 normalize_observations=normalize_observations, normalize_returns=normalize_returns,
                 observation_range=observation_range)

    def build(self):

        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.state_out = OrderedDict()

            if self.normalize_observations:
                all_obs = dict()
                for k, v in self.ob_space.spaces.items():
                    all_obs[k] = self.phs[k]
                self._normalize_inputs(all_obs)
            else:
                self.obs_rms = None

            # Policy net with tanh squashing and rescaling 
            (pi,
            pi_state_out,
            pi_reset_ops) = construct_tf_graph(self.phs, self.network_spec, scope='policy/net')
            self.mus, self.actions, self.logp_pis = self._init_policy_out(pi)

            processed_inp = {k: v for k, v in self.phs.items() if k not in self.ac_space.spaces.keys()}
            processed_inp.update(self.actions)

            # Value net q1
            (vpred,
            vpred_state_out,
            vpred_reset_ops) = construct_tf_graph(self.phs, self.v_network_spec, scope='vpred/q1/net')
            self.q1 = self._init_vpred_head(vpred, self.phs, "vpred/q1/head0", "vpred/q1/filter0")

            (vpred,
            vpred_state_out,
            vpred_reset_ops) = construct_tf_graph(processed_inp, self.v_network_spec, scope='vpred/q1/net', reuse=True)
            self.q1_pi = self._init_vpred_head(vpred, processed_inp, "vpred/q1/head1", "vpred/q1/filter1")

            # Value net q2
            (vpred,
            vpred_state_out,
            vpred_reset_ops) = construct_tf_graph(self.phs, self.v_network_spec, scope='vpred/q2/net')
            self.q2 = self._init_vpred_head(vpred, self.phs, "vpred/q2/head0", "vpred/q2/filter0")

            (vpred,
            vpred_state_out,
            vpred_reset_ops) = construct_tf_graph(processed_inp, self.v_network_spec, scope='vpred/q2/net', reuse=True)
            self.q2_pi = self._init_vpred_head(vpred, processed_inp, "vpred/q2/head1", "vpred/q2/filter1")

            # Value net v
            (vpred,
            vpred_state_out,
            vpred_reset_ops) = construct_tf_graph(self.phs, self.v_network_spec2, scope='vpred/v/net')
            self.v = self._init_vpred_head(vpred, self.phs, "vpred/v/head0", "vpred/v/filter0")

            self.state_out.update(vpred_state_out)
            self.state_out.update(pi_state_out)
            self._reset_ops += vpred_reset_ops + pi_reset_ops
            if self.weight_decay != 0.0:
                kernels = [var for var in self.get_trainable_variables() if 'kernel' in var.name]
                w_norm_sum = tf.reduce_sum([tf.nn.l2_loss(var) for var in kernels])
                w_norm_loss = w_norm_sum * self.weight_decay
                self.add_auxiliary_loss('weight_decay', w_norm_loss)

            # set state to zero state
            self.reset()

    def _init_policy_out(self, pi):
        mus, actions, logp_pis = dict(), dict(), dict()
        with tf.variable_scope('policy/out'):
            for k, v in self.ac_space.spaces.items():
                mus[k] = tf.layers.dense(pi["main"], v.shape[0], kernel_initializer=normc_initializer(0.01))
                log_std = tf.layers.dense(pi["main"], v.shape[0], kernel_initializer=normc_initializer(0.01),
                                          activation=tf.math.tanh)
                log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
                std = tf.exp(log_std)
                actions[k] = mus[k] + tf.random_normal(tf.shape(mus[k])) * std
                logp_pis[k] = self.gaussian_likelihood(actions[k], mus[k], log_std)
                mus[k], actions[k], logp_pis[k] = self.squash(mus[k], actions[k], logp_pis[k])
                action_scale = v.high[0]
                mus[k] *= action_scale
                actions[k] *= action_scale

        return mus, actions, logp_pis
    
    def _init_vpred_head(self, vpred, processed_inp, vpred_scope, feedback_name):
        with tf.variable_scope(vpred_scope):
            _vpred = vpred['main']
            #_vpred = tf.layers.dense(vpred['main'], 1, activation=None,
            #                         kernel_initializer=tf.contrib.layers.xavier_initializer())
            #_vpred = tf.squeeze(vpred['main'], -1)
            if self.normalize_returns:
                normalize_axes = (0, 1)
                loss_fn = partial(l2_loss, mask=processed_inp.get(feedback_name + "_mask", None))
                rms_class = partial(EMAMeanStd, beta=self._ema_beta)
                rms_shape = [dim for i, dim in enumerate(_vpred.get_shape()) if i not in normalize_axes]
                self.value_rms = rms_class(shape=rms_shape, scope='value0filter')
                scaled_value_tensor = self.value_rms.mean + _vpred * self.value_rms.std
                self.add_running_mean_std(rms=self.value_rms, name='feedback.value0', axes=normalize_axes)
            else:
                self.value_rms = None
                scaled_value_tensor = _vpred
        return scaled_value_tensor

    def prepare_input(self, observation, state_in, taken_action=None):
        ''' Add in time dimension to observations, assumes that first dimension of observation is
            already the batch dimension and does not need to be added.'''
        obs = copy(observation)
        
        obs.update(state_in)
        if taken_action is not None:
            obs.update(taken_action)
        return obs
    
    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

    def squash(self, mu, pi, logp_pi):
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
        return mu, pi, logp_pi

    @property
    def actor_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/policy")
    
    @property
    def critic_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/vpred")

    @property
    def critic_output_vars(self):
        output_vars = [var for var in self.critic_variables if 'output' in var.name]
        return output_vars
