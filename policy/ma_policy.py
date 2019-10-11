import tensorflow as tf
import numpy as np
import gym
import logging
import sys
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from baselines.common.distributions import make_pdtype

from util_p import listdict2dictnp, normc_initializer, shape_list, l2_loss
from variable_schema import VariableSchema, BATCH, TIMESTEPS
from normalizers import EMAMeanStd
from graph_construct import construct_tf_graph, construct_schemas_zero_state


class MAPolicy(object):
    '''
        Args:
            ob_space: gym observation space of a SINGLE agent. Expects a dict space.
            ac_space: gym action space. Expects a dict space where each item is a tuple of action
                spaces
            network_spec: list of layers. See construct_tf_graph for details.
            v_network_spec: optional. If specified it is the network spec of the value function.
            trainable_vars: optional. List of variable name segments that should be trained.
            not_trainable_vars: optional. List of variable name segements that should not be
                trained. trainable_vars supercedes this if both are specified.
    '''
    def __init__(self, scope, *, ob_space, ac_space, network_spec, v_network_spec=None,
                 stochastic=True, reuse=False, build_act=True,
                 trainable_vars=None, not_trainable_vars=None,
                 gaussian_fixed_var=True, weight_decay=0.0, ema_beta=0.99999,
                 normalize_observations=True, normalize_returns=True,
                 observation_range=(-5.0, 5.0),
                 **kwargs):
        self.reuse = reuse
        self.scope = scope
        self.ob_space = ob_space
        self.ac_space = deepcopy(ac_space)
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.observation_range = observation_range
        self.network_spec = network_spec
        self.v_network_spec = v_network_spec or self.network_spec
        self.stochastic = stochastic
        self.trainable_vars = trainable_vars
        self.not_trainable_vars = not_trainable_vars
        self.gaussian_fixed_var = gaussian_fixed_var
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        self.build_act = build_act
        self._reset_ops = []
        self._auxiliary_losses = []
        self._running_mean_stds = {}
        self._ema_beta = ema_beta
        self.training_stats = []

        assert isinstance(self.ac_space, gym.spaces.Dict)
        assert isinstance(self.ob_space, gym.spaces.Dict)
        assert 'observation_self' in self.ob_space.spaces

        # Action space will come in as a MA action space. Convert to a SA action space.
        self.ac_space.spaces = {k: v.spaces[0] for k, v in self.ac_space.spaces.items()}

        self.pdtypes = {k: make_pdtype(s) for k, s in self.ac_space.spaces.items()}

        # Create input schemas for each action type
        self.input_schemas = {
            k: VariableSchema(shape=[BATCH, TIMESTEPS] + pdtype.sample_shape(),
                              dtype=pdtype.sample_dtype())
            for k, pdtype in self.pdtypes.items()
        }

        # Creat input schemas for each observation
        for k, v in self.ob_space.spaces.items():
            self.input_schemas[k] = VariableSchema(shape=[BATCH, TIMESTEPS] + list(v.shape),
                                                   dtype=tf.float32)

        # Setup schemas and zero state for layers with state
        v_state_schemas, v_zero_states = construct_schemas_zero_state(
            self.v_network_spec, self.ob_space, 'vpred_net')
        pi_state_schemas, pi_zero_states = construct_schemas_zero_state(
            self.network_spec, self.ob_space, 'policy_net')

        self.state_keys = list(v_state_schemas.keys()) + list(pi_state_schemas.keys())
        self.input_schemas.update(v_state_schemas)
        self.input_schemas.update(pi_state_schemas)
        self.zero_state = {}
        self.zero_state.update(v_zero_states)
        self.zero_state.update(pi_zero_states)

        if build_act:
            with tf.variable_scope(self.scope, reuse=self.reuse):
                self.phs = {name: schema.placeholder(name=name)
                            for name, schema in self.get_input_schemas().items()}
            self.build()

    def build(self):
        pass

    def _normalize_inputs(self, processed_inp):
        with tf.variable_scope('normalize_self_obs'):
            self.obs_rms = ob_rms_self = EMAMeanStd(shape=self.ob_space.spaces['observation_self'].shape,
                                        scope="obsfilter", beta=self._ema_beta, per_element_update=False)
            self.add_running_mean_std("observation_self", ob_rms_self, axes=(0, 1))
            normalized = (processed_inp['observation_self'] - ob_rms_self.mean) / ob_rms_self.std
            clipped = tf.clip_by_value(normalized, self.observation_range[0], self.observation_range[1])
            processed_inp['observation_self'] = clipped

        for key in self.ob_space.spaces.keys():
            if key == 'observation_self':
                continue
            elif 'mask' in key:  # Don't normalize observation masks
                pass
            else:
                with tf.variable_scope(f'normalize_{key}'):
                    ob_rms = EMAMeanStd(shape=self.ob_space.spaces[key].shape[1:],
                                        scope=f"obsfilter/{key}", beta=self._ema_beta, per_element_update=False)
                    normalized = (processed_inp[key] - ob_rms.mean) / ob_rms.std
                    processed_inp[key] = tf.clip_by_value(normalized, self.observation_range[0], self.observation_range[1])
                    self.add_running_mean_std(key, ob_rms, axes=(0, 1, 2))

    def get_input_schemas(self):
        return self.input_schemas.copy()

    def process_state_batch(self, states):
        '''
            Batch states together.
            args:
                states -- list (batch) of dicts of states with shape (n_agent, dim state).
        '''
        new_states = listdict2dictnp(states, keepdims=True)
        return new_states

    def process_observation_batch(self, obs):
        '''
            Batch obs together.
            Args:
                obs -- list of lists (batch, time), where elements are dictionary observations
        '''

        new_obs = deepcopy(obs)
        # List tranpose -- now in (time, batch)
        new_obs = list(map(list, zip(*new_obs)))
        # Convert list of list of dicts to dict of numpy arrays
        new_obs = listdict2dictnp([listdict2dictnp(batch, keepdims=True) for batch in new_obs])
        # Flatten out the agent dimension, so batches look like normal SA batches
        new_obs = {k: self.reshape_ma_observations(v) for k, v in new_obs.items()}

        return new_obs

    def reshape_ma_observations(self, obs):
        # Observations with shape (time, batch)
        if len(obs.shape) == 2:
            batch_first_ordering = (1, 0)
        # Observations with shape (time, batch, dim obs)
        elif len(obs.shape) == 3:
            batch_first_ordering = (1, 0, 2)
        # Observations with shape (time, batch, n_entity, dim obs)
        elif len(obs.shape) == 4:
            batch_first_ordering = (1, 0, 2, 3)
        else:
            raise ValueError(f"Obs dim {obs.shape}. Only supports dim 3 or 4")
        new_obs = obs.copy().transpose(batch_first_ordering)  # (n_agent, batch, time, dim obs)

        return new_obs

    def prepare_input(self, observation, state_in, taken_action=None):
        ''' Add in time dimension to observations, assumes that first dimension of observation is
            already the batch dimension and does not need to be added.'''
        obs = deepcopy(observation)
        obs.update(state_in)
        if taken_action is not None:
            obs.update(taken_action)
        return obs

    def get_variables(self):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.full_scope_name + '/')
        return variables

    def get_trainable_variables(self):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.full_scope_name + '/')
        if self.trainable_vars is not None:
            variables = [v for v in variables
                         if any([tr_v in v.name for tr_v in self.trainable_vars])]
        elif self.not_trainable_vars is not None:
            variables = [v for v in variables
                         if not any([tr_v in v.name for tr_v in self.not_trainable_vars])]
        variables = [v for v in variables if 'not_trainable' not in v.name]
        return variables

    def reset(self):
        self.state = deepcopy(self.zero_state)
        if tf.get_default_session() is not None:
            tf.get_default_session().run(self._reset_ops)

    def set_state(self, state):
        self.state = deepcopy(state)

    def auxiliary_losses(self):
        """ Any extra losses internal to the policy, automatically added to the total loss."""
        return self._auxiliary_losses

    def add_auxiliary_loss(self, name, loss):
        self.training_stats.append((name, 'scalar', loss, lambda x: x))
        self._auxiliary_losses.append(loss)

    def add_running_mean_std(self, name, rms, axes=(0, 1)):
        """
        Add a RunningMeanStd/EMAMeanStd object to the policy's list. It will then get updated during optimization.
        :param name: name of the input field to update from.
        :param rms: RMS object to update.
        :param axes: axes of the input to average over.
            RMS's shape should be equal to input's shape after axes are removed.
            e.g. if inputs is [5, 6, 7, 8] and axes is [0, 2], then RMS's shape should be [6, 8].
        :return:
        """
        self._running_mean_stds[name] = {'rms': rms, 'axes': axes}
