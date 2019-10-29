import os, sys, datetime
sys.path.insert(1, os.getcwd() + "/policy")

import numpy as np
import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables, initialize

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
    def __init__(self, ob_space, ac_space, ent_coef, vf_coef,
                max_grad_norm, mpi_rank_weight=1, comm=None,
                normalize_observations=True, normalize_returns=True,
                use_tensorboard=False, tb_log_dir=None):
        self.sess = sess = get_session()
        self.use_tensorboard = use_tensorboard

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        # CREATE OUR TWO MODELS
        network_spec = [
            {
                'layer_type': 'dense',
                'units': int (256),
                'activation': 'relu',
                'nodes_in': ['observation_self'],
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
        vnetwork_spec = [
            {
                'layer_type': 'dense',
                'units': int (256),
                'activation': 'relu',
                'nodes_in': ['observation_self'],
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
        act_model = PpoPolicy(scope='ppo', ob_space=ob_space, ac_space=ac_space, network_spec=network_spec, v_network_spec=vnetwork_spec,
                stochastic=True, reuse=False, build_act=True,
                trainable_vars=None, not_trainable_vars=None,
                gaussian_fixed_var=True, weight_decay=0.0, ema_beta=0.99999,
                normalize_observations=normalize_observations, normalize_returns=normalize_returns)

        # Train model for training
        train_model = PpoPolicy(scope='ppo', ob_space=ob_space, ac_space=ac_space, network_spec=network_spec, v_network_spec=vnetwork_spec,
                    stochastic=True, reuse=True, build_act=True,
                    trainable_vars=None, not_trainable_vars=None,
                    gaussian_fixed_var=True, weight_decay=0.0, ema_beta=0.99999,
                    normalize_observations=normalize_observations, normalize_returns=normalize_returns)
        
        # CREATE THE PLACEHOLDERS
        self.A = A = {k: v.sample_placeholder([None]) for k, v in train_model.pdtypes.items()}
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = sum([train_model.pds[k].neglogp(A[k]) for k in train_model.pdtypes.keys()])

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        #entropy = tf.reduce_mean(train_model.entropy)
        entropy = tf.reduce_mean(sum([train_model.pds[k].entropy() for k in train_model.pdtypes.keys()]))

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.scaled_value_tensor
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

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables(scope="ppo")
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

        self.step = act_model.act
        self.value = act_model.value
        self.initial_state = act_model.zero_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

        if self.use_tensorboard:
            self.attach_tensorboard(tb_log_dir)
            self.tb_step = 0

    def train(self, lr, cliprange, obs, actions, returns, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values
        
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Turn the obs into correct format
        td_map = {
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
        }

        obs_map = {self.train_model.phs[k]: v for k, v in obs.items()}
        td_map.update(obs_map)
        actions_map = {self.A[k]: v for k, v in actions.items()}
        td_map.update(actions_map)

        if states is not None:
            pass
            #td_map[self.train_model.phs['policy_net_lstm2_state_c']] = np.repeat([states['policy_net_lstm2_state_c'][0]], len(obs), 0)
            #td_map[self.train_model.phs['policy_net_lstm2_state_h']] = np.repeat([states['policy_net_lstm2_state_h'][0]], len(obs), 0)
            #td_map[self.train_model.phs['vpred_net_lstm2_state_c']] = np.repeat([states['vpred_net_lstm2_state_c'][0]], len(obs), 0)
            #td_map[self.train_model.phs['vpred_net_lstm2_state_h']] = np.repeat([states['vpred_net_lstm2_state_h'][0]], len(obs), 0)

        if self.use_tensorboard:
            losses = self.sess.run(self.stats_list + [self._train_op, self.merged], td_map)
            self.tb_writer.add_summary(losses.pop(), self.tb_step)
            self.tb_step += 1
            losses = losses[:-1]
        else:
            losses = self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]

        return losses
    
    def attach_tensorboard(self, logdir):
        for i in range(len(self.stats_list)):
            tf.summary.scalar(self.loss_names[i], self.stats_list[i])
        self.merged = tf.summary.merge_all()
        logdir = os.path.join(os.getcwd(), logdir)
        logdir = os.path.join(logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.tb_writer = tf.summary.FileWriter(logdir, self.sess.graph)
