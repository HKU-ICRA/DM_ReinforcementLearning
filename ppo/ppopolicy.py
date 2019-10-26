import os, sys
sys.path.insert(1, os.getcwd() + "/policy")

import tensorflow as tf
import numpy as np
from collections import OrderedDict
from functools import partial
from copy import deepcopy
import gym

from ma_policy import MAPolicy
from util_p import listdict2dictnp, normc_initializer, shape_list, l2_loss
from variable_schema import VariableSchema, BATCH, TIMESTEPS
from normalizers import EMAMeanStd
from graph_construct import construct_tf_graph, construct_schemas_zero_state

class PpoPolicy(MAPolicy):
    
    def build(self):
        inputs = self.phs
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.full_scope_name = tf.get_variable_scope().name
            self._init(inputs, **self.kwargs)

    def _init(self, inputs, gaussian_fixed_var=True, **kwargs):
        '''
            Args:
                inputs (dict): input dictionary containing tf tensors
                gaussian_fixed_var (bool): If True the policies variance won't be conditioned on state
        '''
        taken_actions = {k: inputs[k] for k in self.pdtypes.keys()}

        #  Copy inputs to not overwrite. Don't need to pass actions to policy, so exclude these
        processed_inp = {k: v for k, v in inputs.items() if k not in self.pdtypes.keys()}

        if self.normalize_observations:
            self._normalize_inputs(processed_inp)
        else:
            self.obs_rms = None

        self.state_out = OrderedDict()

        # Value network
        (vpred,
         vpred_state_out,
         vpred_reset_ops) = construct_tf_graph(processed_inp, self.v_network_spec, scope='vpred_net', reuse=self.reuse)

        self._init_vpred_head(vpred, processed_inp, 'vpred_out0', "value0")

        # Policy network
        (pi,
         pi_state_out,
         pi_reset_ops) = construct_tf_graph(processed_inp, self.network_spec, scope='policy_net', reuse=self.reuse)

        self.state_out.update(vpred_state_out)
        self.state_out.update(pi_state_out)
        self._reset_ops += vpred_reset_ops + pi_reset_ops
        self._init_policy_out(pi, taken_actions)

        if self.weight_decay != 0.0:
            kernels = [var for var in self.get_trainable_variables() if 'kernel' in var.name]
            w_norm_sum = tf.reduce_sum([tf.nn.l2_loss(var) for var in kernels])
            w_norm_loss = w_norm_sum * self.weight_decay
            self.add_auxiliary_loss('weight_decay', w_norm_loss)

        # set state to zero state
        self.reset()

    def _init_policy_out(self, pi, taken_actions):
        with tf.variable_scope('policy_out', reuse=self.reuse):
            self.pdparams = {}
            for k in self.pdtypes.keys():
                with tf.variable_scope(k, reuse=self.reuse):
                    if self.gaussian_fixed_var and isinstance(self.ac_space.spaces[k], gym.spaces.Box):
                        mean = tf.layers.dense(pi["main"],
                                               self.pdtypes[k].param_shape()[0] // 2,
                                               kernel_initializer=normc_initializer(0.01),
                                               activation=None)
                        logstd = tf.get_variable(name="logstd",
                                                 shape=[1, self.pdtypes[k].param_shape()[0] // 2],
                                                 initializer=tf.zeros_initializer())
                        self.pdparams[k] = tf.concat([mean, mean * 0.0 + logstd], axis=2)
                    elif k in pi:
                        # This is just for the case of entity specific actions
                        if isinstance(self.ac_space.spaces[k], (gym.spaces.Discrete)):
                            assert pi[k].get_shape()[-1] == 1
                            self.pdparams[k] = pi[k][..., 0]
                        elif isinstance(self.ac_space.spaces[k], (gym.spaces.MultiDiscrete)):
                            assert np.prod(pi[k].get_shape()[-2:]) == self.pdtypes[k].param_shape()[0],\
                                f"policy had shape {pi[k].get_shape()} for action {k}, but required {self.pdtypes[k].param_shape()}"
                            new_shape = shape_list(pi[k])[:-2] + [np.prod(pi[k].get_shape()[-2:]).value]
                            self.pdparams[k] = tf.reshape(pi[k], shape=new_shape)
                        else:
                            assert False
                    else:
                        self.pdparams[k] = tf.layers.dense(pi["main"],
                                                           self.pdtypes[k].param_shape()[0],
                                                           kernel_initializer=normc_initializer(0.01),
                                                           activation=None)

            with tf.variable_scope('pds', reuse=self.reuse):
                self.pds = {k: pdtype.pdfromflat(self.pdparams[k]) for k, pdtype in self.pdtypes.items()}

            with tf.variable_scope('sampled_action', reuse=self.reuse):
                self.sampled_action = {k: pd.sample() if self.stochastic else pd.mode() for k, pd in self.pds.items()}

            with tf.variable_scope('sampled_action_neglogp', reuse=self.reuse):
                self.sampled_action_neglogp = sum([self.pds[k].neglogp(self.sampled_action[k]) for k in self.pdtypes.keys()])

            with tf.variable_scope('entropy', reuse=False):
                self.entropy = sum([pd.entropy() for pd in self.pds.values()])

            with tf.variable_scope('taken_action_neglogp', reuse=False):
                self.taken_action_neglogp = sum([self.pds[k].neglogp(taken_actions[k]) for k in self.pdtypes.keys()])

    def _init_vpred_head(self, vpred, processed_inp, vpred_scope, feedback_name):
        with tf.variable_scope(vpred_scope, reuse=self.reuse):
            _vpred = tf.layers.dense(vpred['main'], 1, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            _vpred = tf.squeeze(_vpred, -1)
            if self.normalize_returns:
                normalize_axes = (0, 1)
                loss_fn = partial(l2_loss, mask=processed_inp.get(feedback_name + "_mask", None))
                rms_class = partial(EMAMeanStd, beta=self._ema_beta)
                rms_shape = [dim for i, dim in enumerate(_vpred.get_shape()) if i not in normalize_axes]
                self.value_rms = rms_class(shape=rms_shape, scope='value0filter')
                self.scaled_value_tensor = self.value_rms.mean + _vpred * self.value_rms.std
                self.add_running_mean_std(rms=self.value_rms, name='feedback.value0', axes=normalize_axes)
            else:
                self.value_rms = None
                self.scaled_value_tensor = _vpred

    def value(self, observation, states=None, masks=None):
        outputs = {'vpred': self.scaled_value_tensor}
        
        obs = deepcopy(observation)
        n_agents = observation['observation_self'].shape[0]

        # Make sure that there are as many states as there are agents.
        # This should only happen with the zero state.
        for k, v in self.state.items():
            assert v.shape[0] == 1 or v.shape[0] == n_agents
            if v.shape[0] == 1 and v.shape[0] != n_agents:
                self.state[k] = np.repeat(v, n_agents, 0)

        # Add time dimension to obs
        inputs = self.prepare_input(observation=obs, state_in=self.state)
        feed_dict = {self.phs[k]: v for k, v in inputs.items()}

        val = tf.get_default_session().run(outputs, feed_dict)
        return val['vpred']
    
    def act(self, observation, extra_feed_dict={}):
        outputs = {
            'ac': self.sampled_action,
            'ac_neglogp': self.sampled_action_neglogp,
            'vpred': self.scaled_value_tensor,
            'state': self.state_out}
        # Add timestep dimension to observations
        obs = deepcopy(observation)
        n_agents = observation['observation_self'].shape[0]

        # Make sure that there are as many states as there are agents.
        # This should only happen with the zero state.
        for k, v in self.state.items():
            assert v.shape[0] == 1 or v.shape[0] == n_agents
            if v.shape[0] == 1 and v.shape[0] != n_agents:
                self.state[k] = np.repeat(v, n_agents, 0)

        # Add time dimension to obs
        inputs = self.prepare_input(observation=obs, state_in=self.state)
        feed_dict = {self.phs[k]: v for k, v in inputs.items()}
        feed_dict.update(extra_feed_dict)

        outputs = tf.get_default_session().run(outputs, feed_dict)
        self.state = outputs['state']

        # Remove time dimension from outputs
        def preprocess_act_output(act_output):
            if isinstance(act_output, dict):
                return {k: np.squeeze(v, 1) for k, v in act_output.items()}
            else:
                return np.squeeze(act_output, 1)

        info = {'vpred': preprocess_act_output(outputs['vpred']),
                'ac_neglogp': preprocess_act_output(outputs['ac_neglogp']),
                'state': outputs['state']}

        return preprocess_act_output(outputs['ac']), info
    
    def act_in_parallel(self, observation, n_agents, n_batches, extra_feed_dict={}):
        outputs = {
            'ac': self.sampled_action,
            'ac_neglogp': self.sampled_action_neglogp,
            'vpred': self.scaled_value_tensor,
            'state': self.state_out}
        # Add timestep dimension to observations
        obs = deepcopy(observation)

        # Make sure that there are as many states as there are agents.
        # This should only happen with the zero state.
        for k, v in self.state.items():
            assert v.shape[0] == 1 or v.shape[0] == n_agents * n_batches
            if v.shape[0] == 1 and v.shape[0] != n_agents:
                self.state[k] = np.repeat(v, n_agents * n_batches, 0)
        
        # Add time dimension to obs
        for k, v in obs.items():
            obs[k] = np.expand_dims(v, 1)
        inputs = self.prepare_input(observation=obs, state_in=self.state)
        feed_dict = {self.phs[k]: v for k, v in inputs.items()}
        feed_dict.update(extra_feed_dict)
        
        outputs = tf.get_default_session().run(outputs, feed_dict)

        self.state = outputs['state']

        # Remove time dimension from outputs
        def preprocess_act_output(act_output):
            if isinstance(act_output, dict):
                return {k: np.squeeze(v, 1) for k, v in act_output.items()}
            else:
                return np.squeeze(act_output, 1)

        info = {'vpred': preprocess_act_output(outputs['vpred']),
                'ac_neglogp': preprocess_act_output(outputs['ac_neglogp']),
                'state': outputs['state']}

        return preprocess_act_output(outputs['ac']), info
