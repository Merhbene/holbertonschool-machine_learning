#!/usr/bin/env python3
"Momentum Upgraded"
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):

    "create the training operation for a neural network in tensorflow"
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)

    """ Add operations to minimize loss by updating var_list,
     This method simply combines calls compute_gradients() and
     apply_gradients()"""

    return optimizer.minimize(loss)
