import os, sys
sys.path.insert(1, os.getcwd() + "/policy")

import tensorflow as tf
import numpy as np
from copy import deepcopy, copy

from ma_policy import MAPolicy

class DdpgPolicy(MAPolicy):
    def value(self):
        return self.scaled_value_tensor
    
    def act(self):
        return self.sampled_action
    
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
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=["policy_out", "policy_net"])
    
    @property
    def critic_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=["vpred_out0", "vpred_net"])

    @property
    def critic_output_vars(self):
        output_vars = [var for var in self.critic_variables if 'output' in var.name]
        return output_vars
