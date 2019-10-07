import os, sys
sys.path.insert(1, os.getcwd() + "/policy")

import tensorflow as tf
import numpy as np
from copy import deepcopy

from ma_policy import MAPolicy

class PpoPolicy(MAPolicy):
    def value(self, observation, states=None, masks=None):
        outputs = {
            'vpred': self.scaled_value_tensor}
        
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

        val = tf.get_default_session().run(outputs, feed_dict)
        return val['vpred']
