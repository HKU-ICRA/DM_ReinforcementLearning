import os, sys
sys.path.insert(1, os.getcwd() + "/policy")

import numpy as np
import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

from ppopolicy import PpoPolicy

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

def outer_scope_getter(scope, new_scope=""):
    """
    remove a scope layer for the getter
    :param scope: (str) the layer to remove
    :param new_scope: (str) optional replacement name
    :return: (function (function, str, ``*args``, ``**kwargs``): Tensorflow Tensor)
    """
    def _getter(getter, name, *args, **kwargs):
        name = name.replace(scope + "/", new_scope, 1)
        val = getter(name, *args, **kwargs)
        return val
    return _getter

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        # CREATE OUR TWO MODELS
        network_spec = [
            {
                'layer_type': 'dense',
                'units': int (128),
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
            }
        ]
        # Act model that is used for both sampling
        act_model = PpoPolicy(scope='ppo', ob_space=ob_space, ac_space=ac_space, network_spec=network_spec, v_network_spec=None,
                stochastic=True, reuse=False, build_act=True,
                trainable_vars=None, not_trainable_vars=None,
                gaussian_fixed_var=True, weight_decay=0.0, ema_beta=0.99999)

        # Train model for training
        train_model = PpoPolicy(scope='ppo', ob_space=ob_space, ac_space=ac_space, network_spec=network_spec, v_network_spec=None,
                    stochastic=True, reuse=True, build_act=True,
                    trainable_vars=None, not_trainable_vars=None,
                    gaussian_fixed_var=True, weight_decay=0.0, ema_beta=0.99999)
        
        with tf.variable_scope("loss", reuse=False):
            # CREATE THE PLACEHOLDERS
            self.A = A = train_model.phs['action_movement']
            self.ADV = ADV = tf.placeholder(tf.float32, [None])
            self.R = R = tf.placeholder(tf.float32, [None])
            # Keep track of old actor
            self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            # Keep track of old critic
            self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
            self.LR = LR = tf.placeholder(tf.float32, [])
            # Cliprange
            self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

            neglogpac = train_model.taken_action_logp

            # Calculate the entropy
            # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
            entropy = tf.reduce_mean(train_model.entropy)

            # CALCULATE THE LOSS
            # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

            # Clip the value to reduce variability during Critic training
            # Get the predicted value
            vpred = train_model.scaled_value_tensor
            #self.vpred = vpred
            vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - CLIPRANGE, CLIPRANGE)
            # Unclipped value
            vf_losses1 = tf.square(vpred - R)
            # Clipped value
            vf_losses2 = tf.square(vpredclipped - R)

            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            # Calculate ratio (pi current policy / pi old policy)
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

            # Defining Loss = - J is equivalent to max J
            pg_losses = -ADV * ratio

            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

            # Total loss
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
            #self.loss = loss

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables()#train_model.get_trainable_variables()
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model

        self.step = act_model.act#act_model.act_in_parallel
        self.value = act_model.value
        self.initial_state = act_model.zero_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)#, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values
        
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        # Turn the obs into correct format
        phs_obs_self = []
        phs_obs_agentqvelqpos = []
        phs_obs_lidar = []
        for o in obs:
            #phs_obs_self.append([o['observation_self']])
            #phs_obs_agentqvelqpos.append([o['agent_qpos_qvel']])
            #phs_obs_lidar.append([o['lidar']])
            phs_obs_self.append(o['observation_self'])
            #phs_obs_agentqvelqpos.append(o['agent_qpos_qvel'])
            #phs_obs_lidar.append(o['lidar'])
        
        phs_action = []
        for a in actions:
            phs_action.append(a['action_movement'])

        td_map = {
            self.train_model.phs['observation_self'] : phs_obs_self,
            #self.train_model.phs['agent_qpos_qvel'] : phs_obs_agentqvelqpos,
            #self.train_model.phs['lidar'] : phs_obs_lidar,
            self.A : phs_action,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
        }
        
        if states is not None:
            pass
            #td_map[self.train_model.phs['policy_net_lstm2_state_c']] = np.repeat([states['policy_net_lstm2_state_c'][0]], len(obs), 0)
            #td_map[self.train_model.phs['policy_net_lstm2_state_h']] = np.repeat([states['policy_net_lstm2_state_h'][0]], len(obs), 0)
            #td_map[self.train_model.phs['vpred_net_lstm2_state_c']] = np.repeat([states['vpred_net_lstm2_state_c'][0]], len(obs), 0)
            #td_map[self.train_model.phs['vpred_net_lstm2_state_h']] = np.repeat([states['vpred_net_lstm2_state_h'][0]], len(obs), 0)

        #var_check = self.sess.run(self.vpred, td_map)
        #print(var_check)
        return self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]

'''
network_spec = [
                {
                    'layer_type': 'circ_conv1d',
                    'nodes_in': ['lidar'],
                    'nodes_out': ['lidar_conv'],
                    'filters': 10,
                    'kernel_size': 3,
                    'activation': 'relu'
                },
                {
                    'layer_type': 'flatten_outer',
                    'nodes_in': ['lidar_conv'],
                    'nodes_out': ['lidar_conv']
                },
                {
                    'layer_type': 'concat',
                    'nodes_in': ['observation_self', 'lidar_conv'],
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
                    'layer_type': 'layernorm',
                    'nodes_in': ['main'],
                    'nodes_out': ['main']
                },
                {
                    'layer_type': 'concat',
                    'nodes_in': ['observation_self', 'agent_qpos_qvel'],
                    'nodes_out': ['agent_qpos_qvel_dense']
                },
                {
                    'layer_type': 'dense',
                    'units': int (128),
                    'activation': 'relu',
                    'nodes_in': ['agent_qpos_qvel_dense'],
                    'nodes_out': ['agent_qpos_qvel_dense']
                },
                {
                    'layer_type': 'layernorm',
                    'nodes_in': ['agent_qpos_qvel_dense'],
                    'nodes_out': ['agent_qpos_qvel_dense']
                },
                {
                    'layer_type': 'entity_concat',
                    'nodes_in': ['main', 'agent_qpos_qvel_dense'],
                    'nodes_out': ['main']
                },
                {
                    'layer_type': 'residual_sa_block',
                    'nodes_in': ['main'],
                    'nodes_out': ['main'],
                    'heads': int(4),
                    'n_embd': int(128)
                },
                {
                    'layer_type': 'entity_pooling',
                    'nodes_in': ['main'],
                    'nodes_out': ['main']
                },
                {
                    'layer_type': 'concat',
                    'nodes_in': ['main', 'observation_self'],
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
                    'layer_type': 'layernorm',
                    'nodes_in': ['main'],
                    'nodes_out': ['main']
                },
                {
                    'layer_type': 'lstm',
                    'units': int (128),
                    'nodes_in': ['main'],
                    'nodes_out': ['main']
                }
            ]
'''
