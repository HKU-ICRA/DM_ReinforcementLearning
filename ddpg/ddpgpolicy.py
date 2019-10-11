import os, sys
sys.path.insert(1, os.getcwd() + "/policy")

import tensorflow as tf
import numpy as np
from collections import OrderedDict
from copy import deepcopy, copy

from ma_policy import MAPolicy
from util_p import listdict2dictnp, normc_initializer, shape_list, l2_loss
from graph_construct import construct_tf_graph

class DdpgPolicy(MAPolicy):
    
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

            (pi,
            pi_state_out,
            pi_reset_ops) = construct_tf_graph(self.phs, self.network_spec, scope='policy/net')

            self.actions = self._init_policy_out(pi)

            (vpred,
            vpred_state_out,
            vpred_reset_ops) = construct_tf_graph(self.phs, self.v_network_spec, scope='vpred/net')

            self.value0 = self._init_vpred_head(vpred, self.phs, "vpred/head0", "vpred/filter0")

            processed_inp = {k: v for k, v in self.phs.items() if k not in self.ac_space.spaces.keys()}
            processed_inp.update(self.actions)

            (vpred,
            vpred_state_out,
            vpred_reset_ops) = construct_tf_graph(processed_inp, self.v_network_spec, scope='vpred/net', reuse=True)

            self.value1 = self._init_vpred_head(vpred, processed_inp, "vpred/head1", "vpred/filter1")

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
        actions = dict()
        with tf.variable_scope('policy/out'):
            for k, v in self.ac_space.spaces.items():
                actions[k] = pi["main"]
                #tf.layers.dense(pi["main"], v.shape[0],
                #           kernel_initializer=normc_initializer(0.01),
                #           activation=tf.math.tanh)
        return actions
    
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
