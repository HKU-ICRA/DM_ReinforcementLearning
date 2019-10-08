import os, sys
sys.path.insert(1, os.getcwd() + "/policy")

import tensorflow as tf
import numpy as np
from copy import deepcopy, copy

from ma_policy import MAPolicy
from graph_construct import construct_tf_graph

class DdpgPolicy(MAPolicy):
    
    def prepare_input(self, observation, state_in, taken_action=None):
        ''' Add in time dimension to observations, assumes that first dimension of observation is
            already the batch dimension and does not need to be added.'''
        obs = copy(observation)
        
        obs.update(state_in)
        if taken_action is not None:
            obs.update(taken_action)
        return obs
    
    def construct_vpred_net(self, dtype=tf.float32):
        processed_inp = {k: v for k, v in self.phs.items()}
        for k, v in processed_inp.items():
            processed_inp[k] = tf.dtypes.cast(v, dtype)
        return construct_tf_graph(processed_inp, self.v_network_spec, scope='vpred_net')

    def act(self, observation, extra_feed_dict={}):
        outputs = {'ac': self.sampled_action}
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
        for k, v in obs.items():
            obs[k] = np.expand_dims(v, 1)
        inputs = self.prepare_input(observation=obs, state_in=self.state)
        feed_dict = {self.phs[k]: v for k, v in inputs.items()}
        feed_dict.update(extra_feed_dict)

        outputs = tf.get_default_session().run(outputs, feed_dict)

        # Remove time dimension from outputs
        def preprocess_act_output(act_output):
            if isinstance(act_output, dict):
                return {k: np.squeeze(v, 1) for k, v in act_output.items()}
            else:
                return np.squeeze(act_output, 1)

        return preprocess_act_output(outputs['ac'])
    
    def qvalue(self, tensor, observation, action, extra_feed_dict={}):
        outputs = {'qval': tensor}
        # Add timestep dimension to observations
        obs = deepcopy(observation)
        act = deepcopy(action)
        n_agents = observation['observation_self'].shape[0]

        # Make sure that there are as many states as there are agents.
        # This should only happen with the zero state.
        for k, v in self.state.items():
            assert v.shape[0] == 1 or v.shape[0] == n_agents
            if v.shape[0] == 1 and v.shape[0] != n_agents:
                self.state[k] = np.repeat(v, n_agents, 0)

        # Add time dimension to obs
        for k, v in obs.items():
            obs[k] = np.expand_dims(v, 1)
        inputs = self.prepare_input(observation=obs, state_in=self.state)
        feed_dict = {self.phs[k]: v for k, v in inputs.items()}

        for k, v in act.items():
            act[k] = np.expand_dims(v, 1)
        act_feed_dict = {self.phs[k]: v for k, v in act.items()}
        feed_dict.update(act_feed_dict)
        feed_dict.update(extra_feed_dict)

        outputs = tf.get_default_session().run(outputs, feed_dict)

        # Remove time dimension from outputs
        def preprocess_act_output(act_output):
            if isinstance(act_output, dict):
                return {k: np.squeeze(v, 1) for k, v in act_output.items()}
            else:
                return np.squeeze(act_output, 1)

        return preprocess_act_output(outputs['qval'])
    
    def tensor_eval(self, tensor, observation, action=None, extra_feed_dict={}):
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
        for k, v in obs.items():
            obs[k] = np.expand_dims(v, 1)
        inputs = self.prepare_input(observation=obs, state_in=self.state)
        feed_dict = {self.phs[k]: v for k, v in inputs.items()}

        if action is not None:
            act = deepcopy(action)
            for k, v in act.items():
                act[k] = np.expand_dims(v, 1)
            act_feed_dict = {self.phs[k]: v for k, v in act.items()}
            feed_dict.update(act_feed_dict)

        feed_dict.update(extra_feed_dict)
        outputs = tf.get_default_session().run(tensor, feed_dict)

        # Remove time dimension from outputs
        def preprocess_act_output(act_output):
            if isinstance(act_output, dict):
                return {k: np.squeeze(v, 1) for k, v in act_output.items()}
            else:
                return np.squeeze(act_output, 1)

        return preprocess_act_output(outputs)

    @property
    def value(self):
        return self.scaled_value_tensor
    
    @property
    def actor_variables(self):
        return (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ddpg/policy_out")
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ddpg/policy_net"))
    
    @property
    def critic_variables(self):
        return (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ddpg/vpred_out0")
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ddpg/vpred_net"))

    @property
    def critic_output_vars(self):
        output_vars = [var for var in self.critic_variables if 'output' in var.name]
        return output_vars
